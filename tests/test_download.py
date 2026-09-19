import numpy as np
import pandas as pd

import download_historical_prices as download


def test_downloader_sorts_without_filling_and_uses_adjustment(monkeypatch, tmp_path):
    calls = []
    def fake_download(tickers, **kwargs):
        calls.append((tickers, kwargs))
        dates = pd.to_datetime(["2024-01-04", "2024-01-02", "2024-01-03"])
        return pd.DataFrame({("Close", "AAA"): [14., 12., np.nan]}, index=dates)
    monkeypatch.setattr(download.yf, "download", fake_download)
    result = download.build_stock_dataset("2024-01-01", "2024-02-01", ["AAA"], tmp_path / "stocks.csv")
    assert result.index.is_monotonic_increasing
    assert pd.isna(result.iloc[1, 0])
    assert calls[0][1] == {"start": "2024-01-01", "end": "2024-02-01", "auto_adjust": True}


def test_benchmark_respects_dates_and_retains_volume(monkeypatch, tmp_path):
    def fake_download(ticker, **kwargs):
        assert ticker == "SPY"
        assert kwargs == {"start": "2024-01-01", "end": "2024-02-01", "auto_adjust": True}
        return pd.DataFrame({("Close", "SPY"): [102., 101.], ("Volume", "SPY"): [22, 11]},
                            index=pd.to_datetime(["2024-01-03", "2024-01-02"]))
    monkeypatch.setattr(download.yf, "download", fake_download)
    result = download.build_sp500_dataset("2024-01-01", "2024-02-01", tmp_path / "market.csv")
    assert result["Adj Close"].tolist() == [101., 102.]
    assert result.Volume.tolist() == [11, 22]
