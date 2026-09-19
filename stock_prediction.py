"""Joint stock/benchmark forecasting with pretrained TimesFM 3.0."""

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

OUTPERFORMANCE = 10
BENCHMARK = "MARKET"
DEFAULT_TICKERS = ("AAPL", "MSFT", "GOOG", "AMZN")
CHECKPOINT = "google/timesfm-3.0-pytorch"


def read_timeseries(path):
    """Parse daily timestamps, sort before any filling, reject ambiguous rows."""
    frame = pd.read_csv(path)
    if "Date" not in frame or frame.empty:
        raise ValueError(f"{path}: expected a nonempty CSV with a Date column")
    dates = pd.to_datetime(frame.pop("Date"), errors="raise")
    if dates.isna().any():
        raise ValueError(f"{path}: missing dates")
    frame.index = pd.DatetimeIndex(dates).normalize()
    frame.index.name = "Date"
    frame = frame.apply(pd.to_numeric, errors="raise").sort_index(kind="stable")
    # Exact repeated records are harmless; conflicting records have no safe winner.
    frame = frame.reset_index().drop_duplicates().set_index("Date")
    if frame.index.has_duplicates:
        raise ValueError(f"{path}: conflicting duplicate dates")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError(f"{path}: infinite values")
    return frame


def build_data_set(prices_path="stock_prices.csv", benchmark_path="sp500_index.csv",
                   tickers=DEFAULT_TICKERS):
    """Return aligned, sorted prices and past-only covariates, without filling."""
    tickers = list(tickers)
    if not tickers or len(set(tickers)) != len(tickers) or BENCHMARK in tickers:
        raise ValueError("Provide unique stock tickers; MARKET is reserved")
    stocks = read_timeseries(prices_path)
    market = read_timeseries(benchmark_path)
    missing = set(tickers) - set(stocks.columns)
    if missing:
        raise ValueError(f"Missing tickers: {', '.join(sorted(missing))}")
    price_column = "Adj Close" if "Adj Close" in market else "Close"
    if price_column not in market:
        raise ValueError("Benchmark needs an Adj Close or Close column")
    # Benchmark rows define observed sessions. Never silently shift missing dates.
    prices = stocks[tickers].reindex(market.index)
    prices[BENCHMARK] = market[price_column]
    if (prices <= 0).any().any():
        raise ValueError("Observed adjusted prices must be positive")
    covariates = pd.DataFrame(index=prices.index)
    if "Volume" in market:
        if (market["Volume"] < 0).any():
            raise ValueError("Benchmark volume must be nonnegative")
        covariates["market_log_volume"] = np.log1p(market["Volume"])
    return prices, covariates


def create_forecaster(device="auto", batch_size=16, checkpoint=CHECKPOINT):
    """Load weights only when inference is requested, so data tests work offline."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        import torch
        from timesfm3 import ModelConfig, TimesFM3Evaluator
    except ImportError as exc:
        raise RuntimeError(
            "Install TimesFM 3 with: pip install -r requirements.txt"
        ) from exc
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return TimesFM3Evaluator(ModelConfig(
        checkpoint_path=checkpoint, per_core_batch_size=batch_size, device=device,
    ))


@dataclass
class Forecast:
    as_of: pd.Timestamp
    last_prices: pd.Series
    point: pd.DataFrame
    quantiles: np.ndarray  # (targets, horizon, 9), target order matches point.columns

    def ranking(self, outperformance=OUTPERFORMANCE):
        if not np.isfinite(outperformance) or outperformance < 0:
            raise ValueError("outperformance must be finite and nonnegative")
        returns = 100 * (self.point.iloc[-1] / self.last_prices - 1)
        result = pd.DataFrame({
            "last_price": self.last_prices,
            "forecast_price": self.point.iloc[-1],
            "forecast_return_pct": returns,
            "forecast_excess_return_pp": returns - returns[BENCHMARK],
            "price_p10": self.quantiles[:, -1, 0],
            "price_p90": self.quantiles[:, -1, 8],
        }).drop(index=BENCHMARK)
        result["selected"] = result["forecast_excess_return_pp"] >= outperformance
        return result.sort_values("forecast_excess_return_pp", ascending=False)


def forecast_prices(prices, covariates=None, *, forecaster=None, context_len=128,
                    horizon=24, max_fill_gap=3, future_covariates=None):
    """Forecast from history ONLY; future covariates must be known at the origin.

    future_covariates, if provided, is a finite float array of shape
    (channels, context_len + horizon), aligned to the selected context and
    subsequent sessions. Never put future prices/volumes in this array.
    Outputs use session steps, since weekdays alone are not an exchange calendar.
    """
    if not 2 <= context_len <= 15360 or horizon < 1 or max_fill_gap < 0:
        raise ValueError("Require context_len in [2, 15360], horizon >= 1, fill gap >= 0")
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("Prices need a valid DatetimeIndex")
    if prices.index.has_duplicates or prices.columns.has_duplicates:
        raise ValueError("Prices must have unique dates and target names")
    history = prices.sort_index().tail(context_len).copy()
    if len(history) < context_len or len(history.columns) < 2:
        raise ValueError("Insufficient history or fewer than two joint targets")
    if history.iloc[-1].isna().any():
        raise ValueError("Missing price at forecast origin; refusing stale prices")
    if max_fill_gap:
        history = history.ffill(limit=max_fill_gap)
    target = history.to_numpy(dtype=np.float32).T
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Context has missing/invalid prices; choose another window or tickers")
    past_only = None
    if covariates is not None and len(covariates.columns):
        if covariates.index.has_duplicates:
            raise ValueError("Covariates must have unique dates")
        past = covariates.reindex(history.index)
        if max_fill_gap:
            past = past.ffill(limit=max_fill_gap)
        past_only = past.to_numpy(dtype=np.float32).T
        if not np.isfinite(past_only).all():
            raise ValueError("Past covariates have missing/invalid values")
    future = None
    if future_covariates is not None:
        future = np.asarray(future_covariates, dtype=np.float32)
        if (future.ndim != 2 or future.shape[0] < 1 or
                future.shape[1] != context_len + horizon or not np.isfinite(future).all()):
            raise ValueError("Future covariates need shape (channels, context_len + horizon)")
    # Evaluator otherwise chunks targets and may subsample covariates at >32 channels.
    channels = len(target) + (0 if past_only is None else len(past_only))
    channels += 0 if future is None else len(future)
    if channels > 32:
        raise ValueError("Use at most 32 total target/covariate channels for joint attention")
    if forecaster is None:
        forecaster = create_forecaster()
    outputs = list(forecaster.predict_batch(
        contexts=[target], horizon=horizon,
        past_only_covariates=[past_only] if past_only is not None else None,
        past_future_covariates=[future] if future is not None else None,
        return_quantiles=True, use_symmetric_averaging=False,
        make_positive=True, sort_quantiles=True,
    ))
    if len(outputs) != 1:
        raise ValueError("TimesFM returned an unexpected number of forecasts")
    point = np.asarray(outputs[0].forecast)
    quantiles = np.asarray(outputs[0].quantiles)
    if point.shape != (len(target), horizon) or quantiles.shape != (len(target), horizon, 9):
        raise ValueError("TimesFM returned unexpected forecast/quantile shapes")
    if not np.isfinite(point).all() or not np.isfinite(quantiles).all():
        raise ValueError("TimesFM returned nonfinite forecasts")
    if (point < 0).any() or (quantiles < 0).any() or (np.diff(quantiles, axis=-1) < 0).any():
        raise ValueError("TimesFM returned negative prices or crossed quantiles")
    steps = pd.RangeIndex(1, horizon + 1, name="session_step")
    return Forecast(history.index[-1], history.iloc[-1],
                    pd.DataFrame(point.T, index=steps, columns=history.columns), quantiles)


def argument_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--prices", default="stock_prices.csv")
    parser.add_argument("--benchmark", default="sp500_index.csv")
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--outperformance", type=float, default=OUTPERFORMANCE)
    return parser


def predict_stocks(prices_path="stock_prices.csv", benchmark_path="sp500_index.csv",
                   tickers=DEFAULT_TICKERS, *, forecaster=None, context_len=128,
                   horizon=24, outperformance=OUTPERFORMANCE):
    prices, covariates = build_data_set(prices_path, benchmark_path, tickers)
    forecast = forecast_prices(prices, covariates, forecaster=forecaster,
                               context_len=context_len, horizon=horizon)
    return forecast, forecast.ranking(outperformance)


def main():
    parser = argument_parser("TimesFM 3 multivariate adjusted-price forecasts")
    parser.add_argument("--output", default="forecast.csv")
    parser.add_argument("--ranking-output", default="stock_ranking.csv")
    args = parser.parse_args()
    prices, covariates = build_data_set(args.prices, args.benchmark, args.tickers)
    print(f"Forecast origin: {prices.index[-1].date()}; horizon: {args.horizon} sessions")
    model = create_forecaster(args.device, args.batch_size, args.checkpoint)
    forecast = forecast_prices(prices, covariates, forecaster=model,
                               context_len=args.context_len, horizon=args.horizon)
    ranking = forecast.ranking(args.outperformance)
    rows = []
    for i, ticker in enumerate(forecast.point.columns):
        frame = pd.DataFrame(forecast.quantiles[i], index=forecast.point.index,
                             columns=[f"p{q}" for q in range(10, 100, 10)])
        frame["forecast"] = forecast.point[ticker]
        frame["ticker"] = ticker
        frame["as_of"] = forecast.as_of
        rows.append(frame)
    pd.concat(rows).to_csv(args.output)
    ranking.to_csv(args.ranking_output, index_label="ticker")
    print(ranking.to_string())
    print(f"Saved {args.output} and {args.ranking_output}")


if __name__ == "__main__":
    main()
