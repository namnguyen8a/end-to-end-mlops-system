from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# Patch the storage client during import so no real GCP call is made.
with patch("google.cloud.storage.Client") as mock_storage_client:
    mock_storage_client.return_value = MagicMock()
    from src.api import api as api_module


client = TestClient(api_module.app)


def test_health_check_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "tracking_uri" in body
    assert "stage" in body


def test_predict_next_week_success(monkeypatch):
    ticker = "BIC"
    fake_features = [1.0, 2.0, 3.0, 4.0, 5.0]
    fake_time = "2025-11-01T00:00:00Z"

    fake_model = MagicMock()
    fake_model.predict.return_value = [210.5]

    monkeypatch.setattr(
        api_module, "build_latest_features", lambda requested: (fake_features, fake_time)
    )
    monkeypatch.setattr(
        api_module, "load_model_for_ticker", lambda requested: fake_model
    )
    api_module.loaded_models.pop(ticker, None)
    api_module.loaded_uris[ticker] = "models:/BIC_weekly_linear/Staging"

    response = client.get("/predict_next_week", params={"ticker": ticker})
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == ticker
    assert body["predicted_close"] == pytest.approx(210.5)
    assert body["latest_time"] == fake_time
    assert body["model_uri"] == "models:/BIC_weekly_linear/Staging"


def test_predict_next_week_invalid_ticker():
    response = client.get("/predict_next_week", params={"ticker": "XYZ"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Ticker not supported"


