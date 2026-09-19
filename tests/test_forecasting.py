from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtesting import backtest
from stock_prediction import BENCHMARK, build_data_set, forecast_prices, read_timeseries


class RecordingForecaster:
    """A deterministic persistence test double; never presented as TimesFM output."""
    def __init__(self):
        self.calls = []

    def predict_batch(self, **kwargs):
        self.calls.append(kwargs)
        target = kwargs["contexts"][0]
        point = np.repeat(target[:, -1:], kwargs["horizon"], axis=1)
        quantiles = point[:, :, None] * np.linspace(.9, 1.1, 9)
        yield SimpleNamespace(forecast=point, quantiles=quantiles)


@pytest.fixture
def history():
    dates = pd.bdate_range("2024-01-02", periods=12)
    return pd.DataFrame({"AAA": np.arange(10., 22.), "BBB": np.arange(30., 42.),
                         BENCHMARK: np.arange(100., 112.)}, index=dates)


def write_inputs(tmp_path, history):
    stocks = tmp_path / "prices.csv"
    market = tmp_path / "market.csv"
    history.drop(columns=BENCHMARK).sample(frac=1, random_state=5).to_csv(stocks, index_label="Date")
    benchmark = history[[BENCHMARK]].rename(columns={BENCHMARK: "Adj Close"})
    benchmark["Volume"] = np.arange(len(history)) + 1000
    benchmark.iloc[::-1].to_csv(market, index_label="Date")
    return stocks, market


def test_sort_before_fill_and_channel_alignment(history):
    shuffled = history.iloc[::-1].copy()
    shuffled.loc[history.index[3], "AAA"] = np.nan
    covariates = pd.DataFrame({"volume": np.arange(12)}, index=history.index).iloc[::-1]
    future = np.ones((2, 14), dtype=np.float32)
    model = RecordingForecaster()
    result = forecast_prices(shuffled, covariates, forecaster=model, context_len=12,
                             horizon=2, future_covariates=future)
    call = model.calls[0]
    target = call["contexts"][0]
    assert target.shape == (3, 12)
    assert target.dtype == np.float32
    assert target[0, 3] == history.iloc[2]["AAA"]  # earlier date, never later date
    np.testing.assert_array_equal(target[1], history["BBB"])
    np.testing.assert_array_equal(call["past_only_covariates"][0], [np.arange(12)])
    assert call["past_future_covariates"][0].shape == (2, 14)
    assert result.as_of == history.index[-1]
    assert result.point.shape == (2, 3)
    assert result.quantiles.shape == (3, 2, 9)
    assert not result.ranking()["selected"].any()


def test_csv_sort_duplicates_and_alignment(tmp_path, history):
    stocks, market = write_inputs(tmp_path, history)
    prices, covariates = build_data_set(stocks, market, ["AAA", "BBB"])
    pd.testing.assert_frame_equal(prices, history.rename_axis("Date"), check_freq=False)
    np.testing.assert_allclose(covariates.iloc[:, 0], np.log1p(np.arange(12) + 1000))
    frame = pd.read_csv(stocks)
    pd.concat([frame, frame.iloc[:1]]).to_csv(stocks, index=False)
    assert len(read_timeseries(stocks)) == 12
    frame.loc[len(frame)] = frame.iloc[0]
    frame.loc[len(frame) - 1, "AAA"] = 999
    frame.to_csv(stocks, index=False)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        read_timeseries(stocks)


@pytest.mark.parametrize("row", [0, -1])
def test_never_backfill_or_use_stale_last_price(history, row):
    history.iloc[row, 0] = np.nan
    model = RecordingForecaster()
    with pytest.raises(ValueError, match="missing|Missing"):
        forecast_prices(history, forecaster=model, context_len=12)
    assert not model.calls


def test_long_gap_and_future_covariate_validation(history):
    model = RecordingForecaster()
    with pytest.raises(ValueError, match="Future covariates"):
        forecast_prices(history, forecaster=model, context_len=12, horizon=2,
                        future_covariates=np.ones((1, 12)))
    history.iloc[3:7, 0] = np.nan
    with pytest.raises(ValueError, match="missing/invalid"):
        forecast_prices(history, forecaster=model, context_len=12)
    assert not model.calls


def test_walk_forward_no_future_prices_or_volumes(tmp_path, history):
    stocks, market = write_inputs(tmp_path, history)
    model = RecordingForecaster()
    result = backtest(stocks, market, ["AAA", "BBB"], forecaster=model,
                      context_len=4, horizon=2, folds=3)
    assert len(model.calls) == 3
    for origin, call in zip([6, 8, 10], model.calls):
        np.testing.assert_array_equal(call["contexts"][0], history.iloc[origin-4:origin].to_numpy().T)
        np.testing.assert_allclose(call["past_only_covariates"][0],
                                   [np.log1p(np.arange(origin-4, origin) + 1000)])
        assert call["past_future_covariates"] is None
    np.testing.assert_allclose(result.mae, 1.5)
    np.testing.assert_allclose(result.mae, result.naive_mae)
    assert not result.selected.any()
    assert (result.as_of < result.end_date).all()


def test_future_changes_do_not_change_earlier_context(tmp_path, history):
    stocks, market = write_inputs(tmp_path, history)
    first = RecordingForecaster()
    backtest(stocks, market, ["AAA", "BBB"], forecaster=first,
             context_len=4, horizon=2, folds=3)
    history.iloc[6:] *= 100
    stocks, market = write_inputs(tmp_path, history)
    second = RecordingForecaster()
    backtest(stocks, market, ["AAA", "BBB"], forecaster=second,
             context_len=4, horizon=2, folds=3)
    np.testing.assert_array_equal(first.calls[0]["contexts"][0], second.calls[0]["contexts"][0])


def test_missing_evaluation_labels_are_not_filled(tmp_path, history):
    history.iloc[-2, 0] = np.nan
    stocks, market = write_inputs(tmp_path, history)
    with pytest.raises(ValueError, match="held-out"):
        backtest(stocks, market, ["AAA", "BBB"], forecaster=RecordingForecaster(),
                 context_len=4, horizon=2, folds=1)


@pytest.mark.parametrize("failure", ["shape", "nan", "crossed"])
def test_bad_model_outputs_rejected(history, failure):
    class BadModel(RecordingForecaster):
        def predict_batch(self, **kwargs):
            output = next(super().predict_batch(**kwargs))
            if failure == "shape":
                output.forecast = output.forecast.T
            elif failure == "nan":
                output.forecast[0, 0] = np.nan
            else:
                output.quantiles = output.quantiles[:, :, ::-1]
            yield output
    with pytest.raises(ValueError, match="TimesFM returned"):
        forecast_prices(history, forecaster=BadModel(), context_len=12, horizon=2)


def test_ranking_uses_percentage_points_and_boolean_selection(history):
    model = RecordingForecaster()
    result = forecast_prices(history, forecaster=model, context_len=12, horizon=2)
    result.point.iloc[-1] = result.last_prices * [1.2, .95, 1.05]
    ranking = result.ranking(outperformance=10)
    assert ranking[ranking.selected].index.tolist() == ["AAA"]
    assert ranking.loc["AAA", "forecast_excess_return_pp"] == pytest.approx(15)


def test_too_many_channels_rejected(history):
    covariates = pd.DataFrame(np.ones((12, 30)), index=history.index)
    with pytest.raises(ValueError, match="32 total"):
        forecast_prices(history, covariates, forecaster=RecordingForecaster(), context_len=12)
