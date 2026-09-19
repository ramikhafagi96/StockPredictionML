"""Chronological walk-forward evaluation; no random splits or model fitting."""

import numpy as np
import pandas as pd

from stock_prediction import (
    BENCHMARK, DEFAULT_TICKERS, OUTPERFORMANCE, argument_parser,
    build_data_set, create_forecaster, forecast_prices,
)


def backtest(prices_path="stock_prices.csv", benchmark_path="sp500_index.csv",
             tickers=DEFAULT_TICKERS, *, forecaster=None, context_len=128,
             horizon=24, folds=5, outperformance=OUTPERFORMANCE):
    """Evaluate the last `folds` non-overlapping horizons, using history only.

    Error metrics cover every horizon step. Strategy results use endpoint returns
    and are averages across equal-weight selected stocks, before costs. They are
    diagnostic fold returns, not an executable trading strategy or equity curve.
    """
    if folds < 1 or horizon < 1 or context_len < 2:
        raise ValueError("folds/horizon must be positive and context_len >= 2")
    if not np.isfinite(outperformance) or outperformance < 0:
        raise ValueError("outperformance must be finite and nonnegative")
    prices, covariates = build_data_set(prices_path, benchmark_path, tickers)
    first_origin = len(prices) - folds * horizon
    if first_origin < context_len:
        raise ValueError("Not enough rows for context_len + folds * horizon")
    if forecaster is None:
        forecaster = create_forecaster()
    rows = []
    for origin in range(first_origin, len(prices) - horizon + 1, horizon):
        forecast = forecast_prices(
            prices.iloc[:origin], covariates.iloc[:origin], forecaster=forecaster,
            context_len=context_len, horizon=horizon,
        )
        actual = prices.iloc[origin:origin + horizon]
        if not np.isfinite(actual.to_numpy()).all():
            raise ValueError("Missing held-out prices; refusing to impute evaluation labels")
        predicted = forecast.point.to_numpy()
        truth = actual.to_numpy()
        last = forecast.last_prices.to_numpy()
        error = predicted - truth
        naive_error = last[None, :] - truth
        predicted_returns = 100 * (predicted[-1] / last - 1)
        actual_returns = 100 * (truth[-1] / last - 1)
        market_idx = prices.columns.get_loc(BENCHMARK)
        selected = predicted_returns - predicted_returns[market_idx] >= outperformance
        selected[market_idx] = False
        q = forecast.quantiles.transpose(1, 0, 2)
        for i, ticker in enumerate(prices.columns):
            rows.append({
                "as_of": forecast.as_of, "end_date": actual.index[-1], "ticker": ticker,
                "mae": np.abs(error[:, i]).mean(),
                "rmse": np.sqrt(np.square(error[:, i]).mean()),
                "mape_pct": (100 * np.abs(error[:, i] / truth[:, i])).mean(),
                "naive_mae": np.abs(naive_error[:, i]).mean(),
                "naive_mape_pct": (100 * np.abs(naive_error[:, i] / truth[:, i])).mean(),
                "interval_80_coverage": ((truth[:, i] >= q[:, i, 0]) &
                                         (truth[:, i] <= q[:, i, 8])).mean(),
                "forecast_return_pct": predicted_returns[i],
                "actual_return_pct": actual_returns[i],
                "market_return_pct": actual_returns[market_idx],
                "direction_correct": bool(np.sign(predicted_returns[i]) == np.sign(actual_returns[i])),
                "selected": bool(selected[i]),
            })
    return pd.DataFrame(rows)


def main():
    parser = argument_parser("Walk-forward TimesFM 3 versus last-price forecasts")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="backtest.csv")
    args = parser.parse_args()
    model = create_forecaster(args.device, args.batch_size, args.checkpoint)
    results = backtest(args.prices, args.benchmark, args.tickers, forecaster=model,
                       context_len=args.context_len, horizon=args.horizon,
                       folds=args.folds, outperformance=args.outperformance)
    results.to_csv(args.output, index=False)
    metrics = ["mae", "naive_mae", "mape_pct", "naive_mape_pct",
               "interval_80_coverage", "direction_correct"]
    print(results.groupby("ticker")[metrics].mean().to_string())
    selected = results[results["selected"]]
    print(f"Selected stock/fold observations: {len(selected)}")
    if not selected.empty:
        fold_returns = selected.groupby("as_of")[["actual_return_pct", "market_return_pct"]].mean()
        print("Mean equal-weight returns across active folds (%), before costs:")
        print(fold_returns.mean().to_string())
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
