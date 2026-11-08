"""Préparation des variables (« feature engineering »).

Un modèle ne comprend que des nombres. Ce module transforme les colonnes
brutes en matrice numérique :
- variables numériques  -> mises à la même échelle (StandardScaler) ;
- variables catégorielles -> encodées en colonnes 0/1 (OneHotEncoder).

Tout est emballé dans un `Pipeline` scikit-learn : la MÊME transformation
est appliquée à l'entraînement et en production, ce qui évite le piège
classique du « training/serving skew » (le modèle voit des données préparées
différemment en production qu'à l'entraînement).
"""

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk import config


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "cat",
                # handle_unknown="ignore" : une catégorie jamais vue ne fait pas
                # planter l'API, elle est simplement encodée en zéros.
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        # remainder="drop" : toute autre colonne (dont les variables
        # sensibles) est ignorée par le modèle.
        remainder="drop",
    )


def build_pipeline(model: ClassifierMixin) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])
