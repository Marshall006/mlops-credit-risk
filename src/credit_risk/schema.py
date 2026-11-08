"""Contrat de données : ce à quoi les données DOIVENT ressembler.

On utilise pandera pour décrire chaque colonne (type, valeurs autorisées,
bornes). Si un jour la source change (nouvelle catégorie, âge négatif,
colonne manquante...), le pipeline s'arrête avec une erreur claire au lieu
d'entraîner silencieusement un mauvais modèle.
"""

import pandera.pandas as pa

ALLOWED_VALUES = {
    "status": [
        "... < 100 DM",
        "... >= 200 DM / salary for at least 1 year",
        "0 <= ... < 200 DM",
        "no checking account",
    ],
    "credit_history": [
        "all credits at this bank paid back duly",
        "critical account/other credits existing",
        "delay in paying off in the past",
        "existing credits paid back duly till now",
        "no credits taken/all credits paid back duly",
    ],
    "purpose": [
        "business",
        "car (new)",
        "car (used)",
        "domestic appliances",
        "education",
        "furniture/equipment",
        "others",
        "radio/television",
        "repairs",
        "retraining",
    ],
    "savings": [
        "... < 100 DM",
        "... >= 1000 DM",
        "100 <= ... < 500 DM",
        "500 <= ... < 1000 DM",
        "unknown/no savings account",
    ],
    "employment_duration": [
        "... < 1 year",
        "... >= 7 years",
        "1 <= ... < 4 years",
        "4 <= ... < 7 years",
        "unemployed",
    ],
    "personal_status_sex": [
        "female : divorced/separated/married",
        "male : divorced/separated",
        "male : married/widowed",
        "male : single",
    ],
    "other_debtors": ["co-applicant", "guarantor", "none"],
    "property": [
        "building society savings agreement/life insurance",
        "car or other",
        "real estate",
        "unknown/no property",
    ],
    "other_installment_plans": ["bank", "none", "stores"],
    "housing": ["for free", "own", "rent"],
    "job": [
        "management/self-employed/highly qualified employee/officer",
        "skilled employee/official",
        "unemployed/unskilled - non-resident",
        "unskilled - resident",
    ],
    "telephone": ["no", "yes"],
    "foreign_worker": ["no", "yes"],
}


def _categorical(name: str) -> pa.Column:
    return pa.Column(str, pa.Check.isin(ALLOWED_VALUES[name]), nullable=False)


raw_schema = pa.DataFrameSchema(
    {
        # Variables numériques avec des bornes réalistes
        "duration": pa.Column(int, pa.Check.in_range(1, 120)),  # en mois
        "amount": pa.Column(int, pa.Check.gt(0)),  # montant du crédit
        "installment_rate": pa.Column(int, pa.Check.in_range(1, 4)),
        "present_residence": pa.Column(int, pa.Check.in_range(1, 4)),
        "age": pa.Column(int, pa.Check.in_range(18, 100)),
        "number_credits": pa.Column(int, pa.Check.in_range(1, 10)),
        "people_liable": pa.Column(int, pa.Check.in_range(1, 5)),
        # Variables catégorielles : uniquement les valeurs connues
        **{name: _categorical(name) for name in ALLOWED_VALUES},
        # Cible brute : 1 = bon payeur, 0 = mauvais payeur
        "credit_risk": pa.Column(int, pa.Check.isin([0, 1])),
    },
    strict=True,  # refuse toute colonne inattendue
    coerce=True,  # convertit les types quand c'est possible (ex. "5" -> 5)
)
