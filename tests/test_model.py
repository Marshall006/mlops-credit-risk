"""Tests de l'évaluation et du pipeline de modélisation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from credit_risk import config
from credit_risk.evaluate import best_threshold, business_cost, fairness_report
from credit_risk.features import build_pipeline
from credit_risk.schema import ALLOWED_VALUES


def test_business_cost_weights_missed_defaults_five_times_more():
    y_true = np.array([1, 0])
    # Un défaut raté (faux négatif) coûte 5, un bon client refusé coûte 1
    assert business_cost(y_true, np.array([0, 0])) == 5 / 2
    assert business_cost(y_true, np.array([1, 1])) == 1 / 2
    assert business_cost(y_true, np.array([1, 0])) == 0


def test_best_threshold_is_cautious_when_defaults_are_costly():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.3 + rng.normal(0.35, 0.15, 500), 0, 1)
    # Rater un défaut coûte cher -> le seuil optimal doit être sous 0,5
    assert best_threshold(y, proba) < 0.5


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n = 200
    data = {col: rng.integers(1, 5, n) for col in config.NUMERIC_FEATURES}
    data["amount"] = rng.integers(500, 10000, n)
    data["age"] = rng.integers(19, 70, n)
    for col in config.CATEGORICAL_FEATURES + config.SENSITIVE_FEATURES:
        data[col] = rng.choice(ALLOWED_VALUES[col], n)
    X = pd.DataFrame(data)
    y = pd.Series((X["amount"] > 6000).astype(int))
    return X, y


def test_pipeline_ignores_sensitive_features(synthetic_data):
    X, y = synthetic_data
    pipeline = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    used = pipeline.named_steps["preprocess"].get_feature_names_out()
    for sensitive in config.SENSITIVE_FEATURES:
        assert not any(sensitive in name for name in used)


def test_pipeline_handles_unknown_category(synthetic_data):
    X, y = synthetic_data
    pipeline = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    new = X.head(1).copy()
    new["purpose"] = "spaceship"  # catégorie jamais vue
    proba = pipeline.predict_proba(new)[:, 1]
    assert 0 <= proba[0] <= 1


def test_fairness_report_covers_all_groups(synthetic_data):
    X, y = synthetic_data
    report = fairness_report(X, y, np.ones(len(X), dtype=int))
    assert set(report["attribute"]) == {"sex", "foreign_worker"}
    assert report["n"].sum() == 2 * len(X)
