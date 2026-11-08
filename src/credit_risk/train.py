"""Entraînement, comparaison de modèles et suivi des expériences avec MLflow.

Lancement :  python -m credit_risk.train

Ce que fait ce script :
1. charge et valide les données ;
2. compare plusieurs modèles par validation croisée (5 plis) ;
3. choisit le meilleur, puis calcule le seuil de décision qui minimise le coût ;
4. évalue sur le jeu de test (jamais vu pendant l'entraînement) ;
5. enregistre TOUT dans MLflow (paramètres, métriques, audit d'équité, modèle)
   et inscrit le modèle dans le registre avec l'alias « champion » ;
6. sauvegarde aussi le modèle localement pour l'API.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

import joblib
import mlflow
import numpy as np
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from credit_risk import config
from credit_risk.data import add_target, download_data, load_raw, split
from credit_risk.evaluate import (
    best_threshold,
    classification_metrics,
    fairness_report,
)
from credit_risk.features import build_pipeline

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "credit-risk"
REGISTERED_MODEL_NAME = "credit-risk"


def candidate_models() -> dict:
    """Les modèles en compétition, du plus simple au plus puissant."""
    return {
        # Simple, rapide, interprétable : toujours commencer par une baseline.
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=0.5
        ),
        # Forêt d'arbres de décision : capte les interactions entre variables.
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        # Gradient boosting : souvent le plus performant sur données tabulaires.
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            verbose=-1,
        ),
    }


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = add_target(load_raw(download_data()))
    X_train, X_test, y_train, y_test = split(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    # --- 1. Comparaison des candidats par validation croisée -------------------
    results = {}
    for name, model in candidate_models().items():
        with mlflow.start_run(run_name=name):
            pipeline = build_pipeline(model)
            # Probabilités « hors pli » : chaque ligne est prédite par un modèle
            # qui ne l'a pas vue -> estimation honnête de la performance.
            oof_proba = cross_val_predict(
                pipeline, X_train, y_train, cv=cv, method="predict_proba"
            )[:, 1]
            threshold = best_threshold(y_train.to_numpy(), oof_proba)
            metrics = classification_metrics(y_train.to_numpy(), oof_proba, threshold)

            mlflow.log_params({"model": name, **model.get_params()})
            mlflow.log_param("threshold", threshold)
            mlflow.log_metrics({f"cv_{k}": v for k, v in metrics.items()})
            results[name] = {"threshold": threshold, **metrics}
            logger.info(
                "%-20s CV ROC-AUC=%.3f  coût=%.3f  seuil=%.2f",
                name,
                metrics["roc_auc"],
                metrics["business_cost"],
                threshold,
            )

    # --- 2. Sélection : le modèle au coût métier le plus faible ----------------
    best_name = min(results, key=lambda n: results[n]["business_cost"])
    threshold = results[best_name]["threshold"]
    logger.info("Meilleur modèle : %s", best_name)

    # --- 3. Ré-entraînement sur tout le jeu d'entraînement + évaluation test ---
    with mlflow.start_run(run_name=f"champion-{best_name}") as run:
        pipeline = build_pipeline(candidate_models()[best_name])
        pipeline.fit(X_train, y_train)
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = classification_metrics(y_test.to_numpy(), test_proba, threshold)
        fairness = fairness_report(
            X_test, y_test, (test_proba >= threshold).astype(int)
        )

        mlflow.log_params({"model": best_name, "threshold": threshold})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_dict(results, "cv_comparison.json")
        mlflow.log_table(fairness, "fairness_report.json")

        # Le modèle est enregistré avec sa « signature » (format d'entrée/sortie)
        # puis inscrit dans le registre de modèles.
        features = X_train[config.FEATURES]
        signature = infer_signature(features, pipeline.predict_proba(features)[:, 1])
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=features.head(3),
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, "champion", info.registered_model_version
        )

        # --- 4. Copie locale pour l'API (indépendante du serveur MLflow) ------
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, config.MODELS_DIR / "model.joblib")
        metadata = {
            "model_name": best_name,
            "model_version": str(info.registered_model_version),
            "mlflow_run_id": run.info.run_id,
            "threshold": threshold,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "features": config.FEATURES,
            "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
        }
        (config.MODELS_DIR / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )

    logger.info(
        "Test : ROC-AUC=%.3f  rappel=%.3f  coût=%.3f",
        test_metrics["roc_auc"],
        test_metrics["recall"],
        test_metrics["business_cost"],
    )
    logger.info("Audit d'équité :\n%s", fairness.round(3).to_string(index=False))
    return metadata


if __name__ == "__main__":
    np.random.seed(config.RANDOM_STATE)
    main()
