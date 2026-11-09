"""Tests du monitoring de dérive (données synthétiques, sans Internet)."""

import numpy as np
import pandas as pd
import pytest

from credit_risk import config
from credit_risk.monitor import drift_summary, run, simulate_crisis
from credit_risk.schema import ALLOWED_VALUES


@pytest.fixture
def reference() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 400
    data = {col: rng.integers(1, 5, n) for col in config.NUMERIC_FEATURES}
    data["amount"] = rng.integers(500, 8000, n)
    data["duration"] = rng.integers(6, 48, n)
    data["age"] = rng.integers(19, 70, n)
    for col in config.CATEGORICAL_FEATURES:
        data[col] = rng.choice(ALLOWED_VALUES[col], n)
    return pd.DataFrame(data)[config.FEATURES]


def test_no_drift_on_same_distribution(reference, tmp_path):
    summary = run(reference.sample(frac=1, random_state=0), reference, tmp_path)
    assert summary["alert"] is False
    assert (tmp_path / "drift_report.html").exists()


def test_crisis_is_detected(reference, tmp_path):
    baseline = run(reference.sample(frac=1, random_state=0), reference, tmp_path)
    summary = run(simulate_crisis(reference), reference, tmp_path)
    assert {"amount", "duration", "status", "savings"} <= set(
        summary["drifted_columns"]
    )
    assert summary["drift_share"] > baseline["drift_share"]


def test_drift_summary_handles_both_method_types():
    report = {
        "metrics": [
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "a",
                    "method": "K-S p_value",
                    "threshold": 0.05,
                },
                "value": 0.01,  # p-valeur faible -> dérive
            },
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "b",
                    "method": "Wasserstein distance (normed)",
                    "threshold": 0.1,
                },
                "value": 0.02,  # distance faible -> pas de dérive
            },
        ]
    }
    summary = drift_summary(report)
    assert summary["drifted_columns"] == ["a"]
    assert summary["drift_share"] == 0.5
