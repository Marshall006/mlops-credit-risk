"""Monitoring de la dérive des données (data drift) avec Evidently.

Le « bilan de santé » du modèle : les clients d'aujourd'hui ressemblent-ils
encore à ceux sur lesquels le modèle a appris ? Si la population change
(crise économique, nouvelle offre commerciale, nouvelle région…), le modèle
peut se tromper sans que personne ne s'en rende compte : c'est la dérive.

Usage :
  # Données réelles : les prédictions journalisées par l'API
  python -m credit_risk.monitor --current predictions.jsonl
  # Démonstration : simulation d'une crise économique
  python -m credit_risk.monitor --simulate
  # Pour un pipeline CI / une tâche planifiée : code retour 1 si dérive
  python -m credit_risk.monitor --simulate --fail-on-drift

Sorties : reports/drift_report.html (rapport visuel interactif)
          reports/drift_summary.json (résumé lisible par une machine)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from credit_risk import config
from credit_risk.data import add_target, download_data, load_raw, split

logger = logging.getLogger(__name__)

REPORTS_DIR = config.ROOT_DIR / "reports"
# Alerte si plus de 30 % des variables ont dérivé
DRIFT_SHARE_ALERT = 0.3


def reference_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Référence = données d'entraînement ; renvoie aussi le jeu de test."""
    X_train, X_test, _, _ = split(add_target(load_raw(download_data())))
    return X_train[config.FEATURES], X_test[config.FEATURES]


def simulate_crisis(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Simule une crise économique : montants et durées en hausse, plus de
    clients sans compte courant ni épargne."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["amount"] = (out["amount"] * rng.uniform(1.4, 1.9, len(out))).astype(int)
    out["duration"] = np.minimum(out["duration"] + 12, 72)
    poor = rng.random(len(out)) < 0.5
    out.loc[poor, "status"] = "... < 100 DM"
    out.loc[poor, "savings"] = "... < 100 DM"
    return out


def load_prediction_log(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    return df[config.FEATURES]


def drift_summary(report_dict: dict) -> dict:
    """Extrait du rapport Evidently la liste des variables qui ont dérivé."""
    columns = {}
    for metric in report_dict["metrics"]:
        cfg = metric["config"]
        if not cfg.get("type", "").endswith("ValueDrift"):
            continue
        method = cfg["method"]
        value = float(metric["value"])
        # Méthodes à p-valeur (K-S, chi²) : dérive si p < seuil.
        # Méthodes à distance (Wasserstein, Jensen-Shannon…) : dérive si d >= seuil.
        if re.search(r"p_value", method):
            drifted = value < cfg["threshold"]
        else:
            drifted = value >= cfg["threshold"]
        columns[cfg["column"]] = {
            "method": method,
            "score": round(value, 6),
            "drifted": bool(drifted),
        }
    drifted = sorted(c for c, v in columns.items() if v["drifted"])
    share = len(drifted) / len(columns) if columns else 0.0
    return {
        "n_columns": len(columns),
        "n_drifted": len(drifted),
        "drift_share": round(share, 3),
        "alert": share > DRIFT_SHARE_ALERT,
        "drifted_columns": drifted,
        "columns": columns,
    }


def run(current: pd.DataFrame, reference: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = Report([DataDriftPreset()]).run(
        current_data=current, reference_data=reference
    )
    snapshot.save_html(str(out_dir / "drift_report.html"))
    summary = drift_summary(snapshot.dict())
    (out_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--current", type=Path, help="Journal JSONL de l'API")
    source.add_argument("--simulate", action="store_true", help="Simule une crise")
    parser.add_argument("--fail-on-drift", action="store_true")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)

    reference, test = reference_data()
    if args.current:
        current, label = load_prediction_log(args.current), str(args.current)
    elif args.simulate:
        current, label = simulate_crisis(test), "simulation de crise"
    else:
        current, label = test, "jeu de test (aucune dérive attendue)"

    summary = run(current, reference, args.out)
    logger.info(
        "Données actuelles : %s | %d/%d variables en dérive (%.0f %%) -> %s",
        label,
        summary["n_drifted"],
        summary["n_columns"],
        100 * summary["drift_share"],
        "ALERTE" if summary["alert"] else "OK",
    )
    if summary["drifted_columns"]:
        logger.info("Variables en dérive : %s", ", ".join(summary["drifted_columns"]))
    logger.info("Rapport : %s", args.out / "drift_report.html")
    return 1 if (args.fail_on_drift and summary["alert"]) else 0


if __name__ == "__main__":
    sys.exit(main())
