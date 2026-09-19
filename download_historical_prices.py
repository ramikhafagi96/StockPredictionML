"""Download sorted adjusted closes without hiding missing or stale observations."""

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from stock_prediction import DEFAULT_TICKERS

START_DATE = "2003-08-01"
END_DATE = "2015-01-01"


def _tickers(tickers):
    if tickers is not None:
        values = list(tickers)
    else:
        path = Path("intraQuarter/_KeyStats")
        values = [p.name for p in path.iterdir() if not p.name.startswith(".")] if path.exists() else list(DEFAULT_TICKERS)
    values = sorted(set(t.upper() for t in values))
    if not values:
        raise ValueError("No tickers supplied")
    return values


def _close_frame(data, tickers):
    if data.empty:
        raise ValueError("Download returned no data")
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close = close.sort_index()
    if close.index.has_duplicates:
        raise ValueError("Download contains duplicate dates")
    return close


def build_stock_dataset(start=START_DATE, end=END_DATE, tickers=None,
                        output="stock_prices.csv"):
    tickers = _tickers(tickers)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    close = _close_frame(data, tickers).dropna(how="all", axis=1)
    missing = sorted(set(tickers) - set(close.columns))
    if missing:
        raise ValueError(f"No prices returned for: {', '.join(missing)}")
    # No global forward/back fill: the forecasting window owns missing-data policy.
    close.to_csv(output, index_label="Date")
    return close


def build_sp500_dataset(start=START_DATE, end=END_DATE, output="sp500_index.csv"):
    data = yf.download("SPY", start=start, end=end, auto_adjust=True)
    close = _close_frame(data, ["SPY"])
    result = close.iloc[:, 0].rename("Adj Close").to_frame()
    volume = data["Volume"]
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]
    result["Volume"] = volume.reindex(result.index)
    result.to_csv(output, index_label="Date")
    return result


def build_dataset_iteratively(idx_start, idx_end, date_start=START_DATE,
                              date_end=END_DATE, tickers=None, output="stock_prices.csv"):
    selected = _tickers(tickers)[idx_start:idx_end]
    if not selected:
        raise ValueError("Ticker slice is empty")
    frames = []
    for ticker in selected:
        data = yf.download(ticker, start=date_start, end=date_end, auto_adjust=True)
        frames.append(_close_frame(data, [ticker]))
    result = pd.concat(frames, axis=1).sort_index()
    result.to_csv(output, index_label="Date")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download adjusted stock prices and SPY benchmark")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE, help="Exclusive end date")
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--prices-output", default="stock_prices.csv")
    parser.add_argument("--benchmark-output", default="sp500_index.csv")
    args = parser.parse_args()
    build_stock_dataset(args.start, args.end, args.tickers, args.prices_output)
    build_sp500_dataset(args.start, args.end, args.benchmark_output)
