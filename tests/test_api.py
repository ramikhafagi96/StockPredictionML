from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
from stock_prediction import BENCHMARK


class FakeModel:
    def predict_batch(self, **kwargs):
        point = np.repeat(kwargs['contexts'][0][:, -1:], kwargs['horizon'], axis=1)
        yield SimpleNamespace(forecast=point, quantiles=point[:, :, None] * np.linspace(.9, 1.1, 9))


def data_loader(request):
    dates = pd.bdate_range('2025-01-01', periods=128)
    prices = pd.DataFrame({ticker: np.arange(128) + 10.0 * (i + 1)
                           for i, ticker in enumerate([*request.tickers, BENCHMARK])}, index=dates)
    return prices, pd.DataFrame({'volume': np.arange(128)}, index=dates)


def test_prediction_contract_and_model_reuse():
    factories = []
    def factory():
        factories.append(1)
        return FakeModel()
    app = api.create_app(factory, data_loader)
    with TestClient(app) as client:
        assert client.get('/health').json()['model_loaded'] is False
        for _ in range(2):
            response = client.post('/predict', json={'tickers': [' msft ', 'aapl'], 'horizon': 2})
            assert response.status_code == 200, response.text
            body = response.json()
            assert [p['ticker'] for p in body['predictions']] == ['MSFT', 'AAPL']
            assert body['source'] == 'live'
            assert body['benchmark']['ticker'] == BENCHMARK
            assert body['session_steps'] == [1, 2]
            assert body['selected_tickers'] == []
            for prediction in body['predictions']:
                assert len(prediction['forecast']) == 2
                assert list(prediction['quantiles']) == [f'p{q}' for q in range(10, 100, 10)]
                assert all(len(q) == 2 for q in prediction['quantiles'].values())
                assert prediction['forecast_return_pct'] == pytest.approx(0)
                assert prediction['buy'] is False
                assert prediction['sell'] is False
                assert prediction['buy_signal'] == 0
                assert prediction['sell_signal'] == 0
                assert prediction['recommendation'] == 'HOLD'
        assert len(factories) == 1
        assert client.get('/health').json()['model_loaded'] is True
        assert client.get('/openapi.json').status_code == 200


@pytest.mark.parametrize('payload', [
    {}, {'tickers': []}, {'tickers': ['AAPL', 'aapl']}, {'tickers': ['']},
    {'tickers': ['MARKET']}, {'tickers': ['../../file']}, {'tickers': [12]},
    {'tickers': ['AAPL'], 'horizon': 0}, {'tickers': ['AAPL'], 'horizon': True},
    {'tickers': ['AAPL'], 'horizon': 129}, {'tickers': ['AAPL'], 'context_len': 513},
    {'tickers': ['AAPL'], 'source': 'unknown'}, {'tickers': ['AAPL'], 'surprise': 1},
    {'tickers': ['AAPL'], 'outperformance': -1}, {'tickers': [f'T{i}' for i in range(31)]},
])
def test_invalid_request_rejected_before_loading(payload):
    def forbidden(*args):
        pytest.fail('Invalid request must not load data or model')
    with TestClient(api.create_app(forbidden, forbidden)) as client:
        assert client.post('/predict', json=payload).status_code == 422


def test_invalid_history_does_not_load_weights():
    def missing(request):
        prices, covariates = data_loader(request)
        prices.iloc[-1, 0] = np.nan
        return prices, covariates
    def forbidden():
        pytest.fail('Must validate history before loading model')
    with TestClient(api.create_app(forbidden, missing)) as client:
        response = client.post('/predict', json={'tickers': ['AAPL']})
        assert response.status_code == 422
        assert 'Missing price' in response.json()['detail']


@pytest.mark.parametrize('kind,status', [('data', 502), ('model', 503), ('inference', 503)])
def test_failures_release_lock_and_do_not_expose_exception(kind, status):
    def broken(*args, **kwargs):
        raise RuntimeError('private-server-error')
    model = FakeModel()
    if kind == 'inference':
        model.predict_batch = broken
    app = api.create_app(broken if kind == 'model' else lambda: model,
                         broken if kind == 'data' else data_loader)
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post('/predict', json={'tickers': ['AAPL']})
            assert response.status_code == status
            assert 'private-server-error' not in response.text
            assert not app.state.inference_lock.locked()


def test_busy_response():
    app = api.create_app(FakeModel, data_loader)
    app.state.inference_lock.acquire()
    try:
        with TestClient(app) as client:
            response = client.post('/predict', json={'tickers': ['AAPL']})
            assert response.status_code == 503
            assert response.headers['Retry-After'] == '5'
    finally:
        app.state.inference_lock.release()


def test_local_mode_uses_repository_data(monkeypatch):
    monkeypatch.delenv('STOCK_PRICES_PATH', raising=False)
    monkeypatch.delenv('STOCK_BENCHMARK_PATH', raising=False)
    with TestClient(api.create_app(FakeModel)) as client:
        response = client.post('/predict', json={'tickers': ['AAPL', 'MSFT'], 'source': 'local'})
        assert response.status_code == 200, response.text
        assert response.json()['as_of'] == '2014-12-31'
        assert response.json()['source'] == 'local'
        assert client.post('/predict', json={'tickers': ['UNKNOWN'], 'source': 'local'}).status_code == 422


def test_live_download_uses_private_files_and_excludes_today(monkeypatch):
    paths = []
    today = datetime.now(ZoneInfo('America/New_York')).date()
    dates = pd.bdate_range(end=pd.Timestamp(today) - pd.Timedelta(days=1), periods=128)
    def stocks(start, end, tickers, output):
        assert end == str(today)
        assert tickers == ['AAPL']
        paths.append(output)
        pd.DataFrame({'AAPL': np.arange(128) + 100}, index=dates).iloc[::-1].to_csv(output, index_label='Date')
    def market(start, end, output):
        paths.append(output)
        pd.DataFrame({'Adj Close': np.arange(128) + 200, 'Volume': 500}, index=dates).to_csv(output, index_label='Date')
    monkeypatch.setattr(api, 'build_stock_dataset', stocks)
    monkeypatch.setattr(api, 'build_sp500_dataset', market)
    with TestClient(api.create_app(FakeModel)) as client:
        response = client.post('/predict', json={'tickers': ['AAPL']})
        assert response.status_code == 200, response.text
        assert response.json()['as_of'] == str(dates[-1].date())
    assert all(not path.exists() for path in paths)
