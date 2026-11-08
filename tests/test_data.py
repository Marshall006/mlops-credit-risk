"""Tests de la couche données.

Ces tests n'ont pas besoin d'Internet : ils fabriquent de petites données
de test (des « fixtures ») pour vérifier le comportement du code.
"""

import pandas as pd
import pandera.errors
import pytest

from credit_risk import config
from credit_risk.data import add_target, split
from credit_risk.schema import raw_schema


@pytest.fixture
def valid_row() -> dict:
    return {
        "status": "no checking account",
        "duration": 12,
        "credit_history": "existing credits paid back duly till now",
        "purpose": "car (new)",
        "amount": 2500,
        "savings": "... < 100 DM",
        "employment_duration": "1 <= ... < 4 years",
        "installment_rate": 2,
        "personal_status_sex": "male : single",
        "other_debtors": "none",
        "present_residence": 2,
        "property": "real estate",
        "age": 35,
        "other_installment_plans": "none",
        "housing": "own",
        "number_credits": 1,
        "job": "skilled employee/official",
        "people_liable": 1,
        "telephone": "no",
        "foreign_worker": "no",
        "credit_risk": 1,
    }


@pytest.fixture
def raw_df(valid_row) -> pd.DataFrame:
    rows = []
    for i in range(20):
        row = dict(valid_row)
        row["credit_risk"] = 0 if i % 4 == 0 else 1  # 25 % de défauts
        row["age"] = 20 + i
        rows.append(row)
    return pd.DataFrame(rows)


def test_valid_data_passes_schema(raw_df):
    validated = raw_schema.validate(raw_df, lazy=True)
    assert len(validated) == len(raw_df)


def test_unknown_category_is_rejected(raw_df):
    raw_df.loc[0, "housing"] = "castle"
    with pytest.raises(pandera.errors.SchemaErrors):
        raw_schema.validate(raw_df, lazy=True)


def test_unrealistic_age_is_rejected(raw_df):
    raw_df.loc[0, "age"] = 7
    with pytest.raises(pandera.errors.SchemaErrors):
        raw_schema.validate(raw_df, lazy=True)


def test_unexpected_column_is_rejected(raw_df):
    raw_df["salary"] = 1000
    with pytest.raises(pandera.errors.SchemaErrors):
        raw_schema.validate(raw_df, lazy=True)


def test_add_target_inverts_label(raw_df):
    out = add_target(raw_df)
    assert config.RAW_TARGET not in out.columns
    # credit_risk = 0 (mauvais payeur)  ->  is_default = 1
    assert (out[config.TARGET] == 1 - raw_df[config.RAW_TARGET]).all()


def test_split_keeps_default_rate(raw_df):
    X_train, X_test, y_train, y_test = split(add_target(raw_df))
    assert len(X_train) + len(X_test) == len(raw_df)
    assert config.TARGET not in X_train.columns
    assert abs(y_train.mean() - y_test.mean()) < 0.1
