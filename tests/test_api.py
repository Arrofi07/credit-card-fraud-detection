import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import Transaction

SAMPLE_TRANSACTION = Transaction.model_config["json_schema_extra"]["example"]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_expected_shape(client):
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"fraud_probability", "is_fraud", "threshold"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["is_fraud"] == (body["fraud_probability"] >= body["threshold"])


def test_predict_rejects_missing_field(client):
    incomplete = {k: v for k, v in SAMPLE_TRANSACTION.items() if k != "V14"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
