import numpy as np
import pytest
from fastapi.testclient import TestClient

from forecast_api import main, utils
from forecast_api.config import DATE
from forecast_api.data import LiveDataUnavailable
from forecast_api.models import ModelUnavailable


@pytest.fixture
def client():
    return TestClient(main.app)


def test_dashboard_and_health(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize("query,code", [("horizon=0", 422), ("horizon=169", 422), ("horizon=abc", 422), ("model=invalid", 422), ("city=London", 400)])
def test_validation(client, query, code):
    assert client.get("/forecast?" + query).status_code == code


@pytest.mark.parametrize("error,code", [(ModelUnavailable("Train the model"), 503), (LiveDataUnavailable("Try historical mode"), 502), (RuntimeError("private path"), 500)])
def test_errors_are_consistent(client, monkeypatch, error, code):
    def fail(*args):
        raise error
    monkeypatch.setattr(main, "forecast", fail)
    response = client.get("/forecast")
    assert response.status_code == code
    assert "private path" not in response.text


@pytest.mark.parametrize("horizon", [1, 24, 168])
def test_forecast_contract_and_cache(client, monkeypatch, history, horizon):
    utils._cache.clear()
    monkeypatch.setattr(utils, "historical_data", lambda: (history.copy(), "fingerprint"))
    monkeypatch.setattr(utils, "load_sarima", lambda: {"metadata": {"data_fingerprint": "fingerprint", "version": "v1", "trained_through": "2009-12-31"}})
    calls = []

    def predict(model, context, steps):
        calls.append(steps)
        return np.ones(steps), np.zeros(steps), np.full(steps, 2)

    monkeypatch.setattr(utils, "predict_sarima", predict)
    result = client.get(f"/forecast?horizon={horizon}")
    assert result.status_code == 200
    body = result.json()
    assert len(body["forecast"]) == len(body["forecast_timestamps"]) == horizon
    assert body["data_cutoff"] == history[DATE].iloc[-1].isoformat()
    assert body["forecast_timestamps"][0] > body["data_cutoff"]
    assert body["interval_level"] == .95
    assert len(body["history"]) == 48
    assert client.get(f"/forecast?horizon={horizon}").status_code == 200
    assert calls == [horizon]


def test_changed_data_rejected(client, monkeypatch, history):
    monkeypatch.setattr(utils, "historical_data", lambda: (history, "new"))
    monkeypatch.setattr(utils, "load_sarima", lambda: {"metadata": {"data_fingerprint": "old"}})
    assert client.get("/forecast").status_code == 503


def test_readiness_verifies_artifacts(client, monkeypatch, history):
    monkeypatch.setattr(main, "historical_data", lambda: (history, "new"))
    monkeypatch.setattr(main, "load_sarima", lambda: {"metadata": {"data_fingerprint": "old"}})
    assert client.get("/ready").status_code == 503
