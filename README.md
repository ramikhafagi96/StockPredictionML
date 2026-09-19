# Stock forecasting with TimesFM 3

This project uses Google's pretrained **TimesFM 3.0** to jointly forecast adjusted
stock prices and a SPY market benchmark. It replaces the Random Forest training,
fundamental-feature classifier, and random train/test split. No task-specific
training is needed. The legacy fundamental-data scripts and CSVs are retained
for reference but are not used by prediction or evaluation.

## Install and run

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python stock_prediction.py --tickers AAPL MSFT GOOG AMZN --context-len 128 --horizon 24
python backtesting.py --tickers AAPL MSFT GOOG AMZN --context-len 128 --horizon 24 --folds 5
python -m pytest -q
```

The installed package is `timesfm[torch]==3.0.2`; its Python import is `timesfm3`.
The first inference downloads `google/timesfm-3.0-pytorch` from Hugging Face and
needs internet access and enough RAM for the 330M-parameter model. Subsequent
runs can use the cached weights. `--checkpoint` accepts another checkpoint path.
`--device auto` uses CUDA when available, otherwise CPU. You can explicitly select
`cpu`, `cuda`, or `mps` (requires a compatible PyTorch installation).

The bundled data ends on **2014-12-31**. Forecasts from it are historical examples,
not current market forecasts. Download recent adjusted prices into separate files:

```bash
python download_historical_prices.py --start 2024-01-01 --end 2026-09-19 \
  --tickers AAPL MSFT GOOG AMZN \
  --prices-output recent_prices.csv --benchmark-output recent_market.csv
python stock_prediction.py --prices recent_prices.csv --benchmark recent_market.csv
python backtesting.py --prices recent_prices.csv --benchmark recent_market.csv
```

The download end date is exclusive. SPY is an ETF proxy for the S&P 500; its
adjusted returns include distributions. All target price series must use a
consistent adjusted-price convention. The downloader explicitly enables price
adjustment and does not fill missing quotes.

## Multivariate inputs and sorting

- `stock_prices.csv`: `Date` plus one adjusted-close column per ticker.
- `sp500_index.csv`: `Date`, `Adj Close` (or adjusted `Close`), optionally `Volume`.
- All dates are parsed and sorted **before** selecting context or filling gaps.
  Exact duplicate records are removed; conflicting rows for the same date fail.
- Benchmark dates define the observed session grid. Stock and volume channels
  are aligned by date, never by CSV row position.
- Selected stocks plus `MARKET` form one target array `(stocks + 1, context_len)`.
  When available, log-transformed benchmark volume is a past-only covariate
  `(1, context_len)`. Future volumes and future stock prices are never inputs.
- Only small interior gaps are forward-filled (maximum three observations),
  within the selected context. Leading gaps, longer gaps, missing origin prices,
  nonpositive prices, and nonfinite inputs fail with an error. No backward fill.
- The wrapper limits the combined targets/covariates to 32 channels so the
  evaluator does not silently split the joint target group or subsample features.
  Choose a related group of stocks for each run. At least two targets are required;
  one stock plus the benchmark is sufficient.

For known future signals, the Python API accepts `future_covariates` as a finite
array of shape `(channels, context_len + horizon)`. Supply context and future
values aligned to the actual session schedule and known at the forecast origin.
For example, scheduled events or exchange-calendar features can be used. These
are optional; the default pipeline does not fabricate future financial features.

```python
from stock_prediction import build_data_set, forecast_prices

prices, past_covariates = build_data_set(tickers=["AAPL", "MSFT", "GOOG"])
forecast = forecast_prices(prices, past_covariates, context_len=128, horizon=24)
print(forecast.point)      # (24, 4): three stocks plus MARKET
print(forecast.quantiles.shape)  # (4, 24, 9): p10 through p90
print(forecast.ranking(outperformance=10))
```

## Outputs and evaluation

`stock_prediction.py` writes `forecast.csv` with origin date, session step, ticker,
point price and all nine quantiles. `stock_ranking.csv` includes endpoint expected
returns, expected excess return against the joint benchmark forecast, p10/p90
price bounds, and a `selected` flag. `--outperformance 10` means **10 percentage
points** of forecast excess return over the chosen horizon; it is not annualized.
Quantiles describe model uncertainty and are not guaranteed calibrated intervals.
Future steps are trading-session counts, not invented weekday dates.

`backtesting.py` writes `backtest.csv`. It evaluates the last five non-overlapping
24-session windows by default. Each model call receives only prices and volume
strictly before that window. Earlier test windows become available history for
later origins, as they would in a walk-forward evaluation. Held-out price labels
are never imputed. Per-ticker/fold results include MAE, RMSE, MAPE, last-price
baseline MAE/MAPE, p10–p90 interval coverage, endpoint direction and returns.

Selection uses a boolean mask. The CLI also reports mean equal-weight stock and
benchmark returns across active folds; empty selections produce no trades.
These are diagnostic close-to-close returns before costs, not an execution
simulation or compounded portfolio. No accuracy or profitability improvement is
assumed: compare TimesFM against the last-price baseline on genuinely unseen data.
The model's pretraining may include the old bundled history, and the fixed ticker
universe has survivorship/selection bias, so that sample cannot establish live
performance. Keep a later untouched period for evaluating parameter choices.

Tests use a deterministic injected forecaster to verify sorting, alignment,
chronological isolation, missing-data handling, output validation and metrics
without downloading model weights. They do not measure TimesFM accuracy.

## Upstream model and license

See the [official TimesFM repository](https://github.com/google-research/timesfm)
for the multivariate API. The source is Apache-2.0, but the **TimesFM 3 default
weights are restricted to non-commercial, non-production use** under the separate
[timesfm-non-commercial-license-v1.0 notice](https://github.com/google-research/timesfm#license-notice-for-pretrained-weights).

## Verified migration run

The regression suite passed **19 tests**. A CPU run with the official 3.0.2 package
and default checkpoint produced and validated a 24-session forecast for four
stocks plus MARKET, including all nine quantiles. The default five-fold backtest
also completed on the bundled history (context 128, horizon 24):

| Target | TimesFM MAE | Last-price MAE |
| ------ | ----------: | -------------: |
| AAPL   |      3.9117 |         3.6487 |
| AMZN   |     14.6080 |        14.6954 |
| GOOG   |     13.2219 |        14.9228 |
| MSFT   |      1.3362 |         1.6376 |
| MARKET |      4.5966 |         4.6963 |

MAE improved for three of the four stocks and the benchmark; AAPL was worse.
There were no selections at the default 10-percentage-point excess-return
threshold. This historical integration check does not demonstrate performance
on current or unseen markets. Generated `forecast.csv`, `stock_ranking.csv`, and
`backtest.csv` are ignored by Git and can be regenerated with the commands above.

## POST API

Install the requirements above, then start the local API from the repository root:

```bash
source .venv/bin/activate
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open **http://127.0.0.1:8000/docs** to enter a request and inspect the complete
request/response schema. `GET /health` reports whether the server is running and
whether the model has loaded. The first valid forecast loads the model; later
requests reuse it. Model loading/download and CPU inference can make the first
request take several minutes. Run one worker to avoid duplicating model memory.

In a second terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"tickers": ["AAPL", "MSFT", "GOOG"], "horizon": 24, "context_len": 128}'
```

Only `tickers` is required. The API defaults to `"source": "live"`: it downloads
recent adjusted prices for the requested tickers and SPY into private temporary
files, sorts/aligns the data, forecasts, and deletes those temporary files.
It does not overwrite your project CSVs. Live here means a fresh historical-data
request, not streaming/intraday data: today's US session is excluded to avoid
using an unfinished daily candle. `as_of` gives the actual last observation.
The benchmark is always SPY; the current alignment is intended for stocks sharing
US trading sessions. Other exchanges can have different calendars and currencies.

To test with the bundled historical data without a price download:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"tickers": ["AAPL", "MSFT"], "source": "local", "horizon": 24}'
```

Local mode still needs the TimesFM weights cached or downloaded on first use.
It defaults to the bundled data ending in 2014. Set `STOCK_PRICES_PATH` and
`STOCK_BENCHMARK_PATH` on the server to use other local CSV files. Configure model
hardware with `TIMESFM_DEVICE=cpu` (or `auto`, `cuda`, `mps`) and optionally use
`TIMESFM_CHECKPOINT` for a different checkpoint.

| Request field      | Default  | Meaning                                                  |
| ------------------ | -------- | -------------------------------------------------------- |
| `tickers`        | Required | 1–30 unique symbols; whitespace removed and uppercased  |
| `horizon`        | 24       | 1–128 future trading-session steps                      |
| `context_len`    | 128      | 32–512 historical session observations                  |
| `source`         | `live` | Fresh provider download or configured`local` CSVs      |
| `outperformance` | 10       | Nonnegative excess-return threshold in percentage points |

The JSON response includes:

- `model`, `source`, `as_of`, `context_len`, `horizon`, and `session_steps`.
- `predictions`: one object per requested ticker, in request order, containing
  `ticker`, `last_price`, `forecast_return_pct`, `forecast_excess_return_pp`,
  `selected`, `forecast` (one price per step), and `quantiles` (p10 through p90,
  each an array of one price per step).
- `benchmark`: the same forecast fields for MARKET (SPY).
- `ranking`: requested ticker names ordered by predicted excess return.
- `selected_tickers`: names exceeding the chosen threshold.

Each prediction also includes simple rule-based action fields:

- `buy` / `sell`: JSON booleans.
- `buy_signal` / `sell_signal`: the same decisions as `1` or `0`.
- `recommendation`: `BUY`, `SELL`, or `HOLD`.

`BUY` means the predicted excess return is at least `outperformance` percentage
points above MARKET. `SELL` means the predicted return is negative and the stock
does not meet the buy threshold. A positive forecast below the buy threshold is
`HOLD`. These are model rules, not financial advice or a guarantee of future
returns. The benchmark itself always returns `buy=false` and `sell=false`.

Python client example:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"tickers": ["AAPL", "MSFT"], "horizon": 24},
    timeout=600,
)
response.raise_for_status()
result = response.json()
for stock in result["predictions"]:
    print(stock["ticker"], stock["forecast"], stock["quantiles"]["p10"])
```

Bad input, unavailable symbols, or insufficient/invalid history return **422**
with a JSON `detail`. Provider/file access failures return **502**; model load or
inference failures return **503**. The API processes one forecast at a time;
concurrent requests receive **503** with `Retry-After: 5`. Download failures do
not silently fall back to old local data. Server logs contain technical errors.
This is a local development API, with no authentication added.
The default model's non-commercial/non-production license still applies.

API validation and testing follow the [FastAPI request-body](https://fastapi.tiangolo.com/tutorial/body/)
and [testing](https://fastapi.tiangolo.com/tutorial/testing/) interfaces.

Verified API run: **42 tests passed** across the full suite. A real POST request
for AAPL and MSFT with `source=live` and `horizon=3` returned HTTP 200 using prices
through 2026-09-18 and the official cached TimesFM checkpoint. Live provider
availability and subsequent market data will vary.
