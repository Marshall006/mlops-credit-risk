"""Évaluation : métriques techniques, coût métier et audit d'équité."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Matrice de coût classique de ce jeu de données (fournie avec les données UCI) :
# accorder un crédit à quelqu'un qui ne rembourse pas coûte 5 fois plus cher
# que refuser un crédit à un bon payeur.
COST_FALSE_NEGATIVE = 5  # défaut non détecté -> perte d'argent
COST_FALSE_POSITIVE = 1  # bon client refusé -> manque à gagner


def business_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coût moyen par demande de crédit selon la matrice de coût."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float((COST_FALSE_NEGATIVE * fn + COST_FALSE_POSITIVE * fp) / len(y_true))


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Cherche le seuil de décision qui minimise le coût métier.

    Par défaut un modèle dit « défaut » si la probabilité dépasse 0,5. Mais
    comme rater un défaut coûte 5 fois plus cher, le meilleur seuil est
    souvent bien plus bas : on préfère être prudent.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    costs = [business_cost(y_true, (proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmin(costs))])


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, proba),
        "business_cost": business_cost(y_true, y_pred),
    }


def fairness_report(
    X: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray
) -> pd.DataFrame:
    """Compare le comportement du modèle entre groupes sensibles.

    Le modèle n'a jamais vu ces variables, mais il pourrait les « deviner » via
    d'autres colonnes (âge, type d'emploi...). On mesure donc, par groupe :
    - le taux de refus (part des demandes classées « défaut ») ;
    - le rappel (part des vrais défauts correctement détectés).
    Des écarts importants entre groupes signalent un biais à investiguer.
    """
    groups = pd.DataFrame(
        {
            "sex": X["personal_status_sex"].str.split(" : ").str[0],
            "foreign_worker": X["foreign_worker"],
        },
        index=X.index,
    )
    rows = []
    for attribute in groups.columns:
        for value, idx in groups.groupby(attribute).groups.items():
            yt = y_true.loc[idx].to_numpy()
            yp = y_pred[X.index.get_indexer(idx)]
            rows.append(
                {
                    "attribute": attribute,
                    "group": value,
                    "n": len(idx),
                    "rejection_rate": float(yp.mean()),
                    "recall": recall_score(yt, yp, zero_division=0),
                }
            )
    return pd.DataFrame(rows)
