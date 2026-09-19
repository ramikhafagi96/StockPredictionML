"""Tests for the price datasets used by the TimesFM pipeline."""

import numpy as np

from stock_prediction import BENCHMARK, DEFAULT_TICKERS, build_data_set


def test_repository_price_dataset():
    prices, covariates = build_data_set()
    assert prices.index.is_monotonic_increasing
    assert prices.index.is_unique
    assert list(prices.columns) == [*DEFAULT_TICKERS, BENCHMARK]
    assert len(prices) >= 128 + 5 * 24
    assert covariates.index.equals(prices.index)
    assert np.isfinite(prices.tail(248).to_numpy()).all()
    assert (prices.tail(248) > 0).all().all()
