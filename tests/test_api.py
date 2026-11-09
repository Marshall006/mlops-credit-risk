"""Tests de l'API : on entraîne un mini-modèle et on interroge l'API comme un
vrai client, grâce au TestClient de FastAPI (aucun serveur à lancer)."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from credit_risk import api, config
from credit_risk.features import build_pipeline
from credit_risk.schema import ALLOWED_VALUES

EXAMPLE = api.CreditApplication.model_config["json_schema_extra"]["example"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    # 1. Mini-modèle entraîné sur des données synthétiques
    rng = np.random.default_rng(0)
    n = 100
    data = {col: rng.integers(1, 4, n) for col in config.NUMERIC_FEATURES}
    data["amount"] = rng.integers(500, 10000, n)
    data["age"] = rng.integers(19, 70, n)
    for col in config.CATEGORICAL_FEATURES:
        data[col] = rng.choice(ALLOWED_VALUES[col], n)
    X = pd.DataFrame(data)
    y = (X["amount"] > 5000).astype(int)
    pipeline = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)

    import joblib

    joblib.dump(pipeline, tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"model_name": "test", "model_version": "0", "threshold": 0.5})
    )
    # 2. L'API pointe vers ce modèle et journalise ses prédictions
    monkeypatch.setattr(api, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(api, "PREDICTION_LOG", str(tmp_path / "preds.jsonl"))
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_probability_and_decision(client):
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability_default"] <= 1
    assert body["decision"] in {"approve", "reject"}
    assert body["model_version"] == "0"


def test_high_amount_is_riskier(client):
    low = client.post("/predict", json={**EXAMPLE, "amount": 600}).json()
    high = client.post("/predict", json={**EXAMPLE, "amount": 9500}).json()
    assert high["probability_default"] > low["probability_default"]


def test_invalid_category_is_rejected(client):
    response = client.post("/predict", json={**EXAMPLE, "housing": "castle"})
    assert response.status_code == 422


def test_invalid_age_is_rejected(client):
    response = client.post("/predict", json={**EXAMPLE, "age": 7})
    assert response.status_code == 422


def test_batch_and_prediction_log(client, tmp_path):
    response = client.post("/predict/batch", json=[EXAMPLE, EXAMPLE, EXAMPLE])
    assert response.status_code == 200
    assert len(response.json()) == 3
    lines = (tmp_path / "preds.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    assert "probability_default" in json.loads(lines[0])


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json=[]).status_code == 400


def test_metrics_endpoint_exposes_prometheus_format(client):
    client.get("/health")
    assert "http_requests_total" in client.get("/metrics").text
