from stock_prediction import DEFAULT_TICKERS, OUTPERFORMANCE


def test_defaults():
    assert OUTPERFORMANCE >= 0
    assert len(DEFAULT_TICKERS) >= 2
    assert len(set(DEFAULT_TICKERS)) == len(DEFAULT_TICKERS)
