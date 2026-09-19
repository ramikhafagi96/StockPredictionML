"""Local HTTP API: uvicorn api:app --host 127.0.0.1 --port 8000."""

import logging
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from download_historical_prices import build_sp500_dataset, build_stock_dataset
from stock_prediction import BENCHMARK, CHECKPOINT, build_data_set, create_forecaster, forecast_prices

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    tickers: list[str] = Field(min_length=1, max_length=30)
    horizon: int = Field(default=24, ge=1, le=128, strict=True)
    context_len: int = Field(default=128, ge=32, le=512, strict=True)
    source: Literal["live", "local"] = "live"
    outperformance: float = Field(default=10, ge=0)

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, values):
        tickers = [value.strip().upper() for value in values]
        if any(not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=-]{0,19}", t) for t in tickers):
            raise ValueError("Use valid ticker symbols, e.g. AAPL, BRK-B, or RELIANCE.NS")
        if BENCHMARK in tickers:
            raise ValueError("MARKET is reserved for the benchmark")
        if len(set(tickers)) != len(tickers):
            raise ValueError("Tickers must be unique (case insensitive)")
        return tickers


class SeriesPrediction(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    ticker: str
    last_price: float
    forecast_return_pct: float
    forecast_excess_return_pp: float
    selected: bool
    buy: bool
    sell: bool
    buy_signal: int = Field(ge=0, le=1)
    sell_signal: int = Field(ge=0, le=1)
    recommendation: Literal["BUY", "SELL", "HOLD"]
    forecast: list[float]
    quantiles: dict[str, list[float]]


class PredictionResponse(BaseModel):
    model: str
    source: Literal["live", "local"]
    as_of: date
    context_len: int
    horizon: int
    session_steps: list[int]
    predictions: list[SeriesPrediction]
    benchmark: SeriesPrediction
    ranking: list[str]
    selected_tickers: list[str]


def load_data(request):
    if request.source == "local":
        return build_data_set(
            os.getenv("STOCK_PRICES_PATH", str(ROOT / "stock_prices.csv")),
            os.getenv("STOCK_BENCHMARK_PATH", str(ROOT / "sp500_index.csv")),
            request.tickers,
        )
    # Exclude today's potentially unfinished US session. No shared CSV mutations.
    end = datetime.now(ZoneInfo("America/New_York")).date()
    start = end - timedelta(days=request.context_len * 2 + 60)
    with TemporaryDirectory(prefix="stock-forecast-") as directory:
        prices_path = Path(directory) / "prices.csv"
        market_path = Path(directory) / "market.csv"
        build_stock_dataset(str(start), str(end), request.tickers, prices_path)
        build_sp500_dataset(str(start), str(end), market_path)
        prices, covariates = build_data_set(prices_path, market_path, request.tickers)
    if (end - prices.index[-1].date()).days > 7:
        raise ValueError("Provider returned stale benchmark data (over seven days old)")
    return prices, covariates


def serialize_forecast(forecast, request, checkpoint):
    ranking = forecast.ranking(request.outperformance)
    returns = 100 * (forecast.point.iloc[-1] / forecast.last_prices - 1)
    series = {}
    for i, ticker in enumerate(forecast.point.columns):
        is_benchmark = ticker == BENCHMARK
        is_selected = bool(ranking.loc[ticker, "selected"]) if not is_benchmark else False
        is_sell = bool((returns[ticker] < 0) and not is_selected) if not is_benchmark else False
        recommendation = "BUY" if is_selected else ("SELL" if is_sell else "HOLD")
        series[ticker] = SeriesPrediction(
            ticker=ticker,
            last_price=float(forecast.last_prices[ticker]),
            forecast_return_pct=float(returns[ticker]),
            forecast_excess_return_pp=float(returns[ticker] - returns[BENCHMARK]),
            selected=is_selected,
            buy=is_selected,
            sell=is_sell,
            buy_signal=int(is_selected),
            sell_signal=int(is_sell),
            recommendation=recommendation,
            forecast=forecast.point[ticker].tolist(),
            quantiles={f"p{q}": forecast.quantiles[i, :, j].tolist()
                       for j, q in enumerate(range(10, 100, 10))},
        )
    return PredictionResponse(
        model=checkpoint, source=request.source, as_of=forecast.as_of.date(),
        context_len=request.context_len, horizon=request.horizon,
        session_steps=forecast.point.index.tolist(),
        predictions=[series[ticker] for ticker in request.tickers],
        benchmark=series[BENCHMARK], ranking=ranking.index.tolist(),
        selected_tickers=ranking.index[ranking["selected"]].tolist(),
    )


def create_app(model_factory=None, data_loader=None):
    """Inject model/data providers for deterministic offline endpoint tests."""
    application = FastAPI(title="TimesFM Stock Forecast API", version="1.0.0")
    application.state.model = None
    application.state.inference_lock = Lock()
    checkpoint = os.getenv("TIMESFM_CHECKPOINT", CHECKPOINT)
    factory = model_factory or (lambda: create_forecaster(
        device=os.getenv("TIMESFM_DEVICE", "auto"), batch_size=1, checkpoint=checkpoint,
    ))
    loader = data_loader or load_data

    @application.get("/health")
    def health():
        return {"status": "ok", "model_loaded": application.state.model is not None}

    @application.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest):
        # One model per process. Avoid concurrent model/download mutation and OOM.
        if not application.state.inference_lock.acquire(blocking=False):
            raise HTTPException(503, "Another forecast is running; retry shortly", headers={"Retry-After": "5"})
        try:
            try:
                prices, covariates = loader(request)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            except Exception as exc:
                logger.exception("Price data loading failed")
                raise HTTPException(502, "Could not load price data; check the provider or local CSV configuration") from exc

            def get_model():
                if application.state.model is None:
                    try:
                        application.state.model = factory()
                    except Exception as exc:
                        logger.exception("TimesFM loading failed")
                        raise HTTPException(503, "Could not load TimesFM; check server logs, dependencies and checkpoint access") from exc
                return application.state.model

            # Lazy proxy lets forecast_prices validate inputs before loading weights.
            class LazyForecaster:
                def predict_batch(self, **kwargs):
                    try:
                        return list(get_model().predict_batch(**kwargs))
                    except HTTPException:
                        raise
                    except Exception as exc:
                        logger.exception("TimesFM inference failed")
                        raise HTTPException(503, "TimesFM inference failed; check server logs") from exc

            try:
                forecast = forecast_prices(
                    prices, covariates, forecaster=LazyForecaster(),
                    context_len=request.context_len, horizon=request.horizon,
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            return serialize_forecast(forecast, request, checkpoint)
        finally:
            application.state.inference_lock.release()

    return application


app = create_app()
