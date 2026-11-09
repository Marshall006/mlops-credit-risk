"""API de prédiction (FastAPI) : le « guichet » du modèle.

Lancement local :  uvicorn credit_risk.api:app --reload
Documentation interactive auto-générée : http://localhost:8000/docs

Endpoints :
- GET  /health         -> l'API et le modèle sont-ils prêts ?
- GET  /model/info     -> version, seuil, métriques du modèle servi
- POST /predict        -> score une demande de crédit
- POST /predict/batch  -> score plusieurs demandes d'un coup
- GET  /metrics        -> métriques Prometheus (latence, nb de requêtes…)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

from credit_risk import __version__, config
from credit_risk.schema import ALLOWED_VALUES

logger = logging.getLogger("credit_risk.api")

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(config.MODELS_DIR)))
# Si défini, chaque prédiction est journalisée (JSON Lines) : ces données
# alimentent ensuite le monitoring de dérive (credit_risk.monitor).
PREDICTION_LOG = os.getenv("PREDICTION_LOG_PATH")


def _choices(column: str):
    """Transforme une liste de valeurs autorisées en type `Literal` :
    pydantic refusera automatiquement toute autre valeur (erreur 422)."""
    return Literal[tuple(ALLOWED_VALUES[column])]


class CreditApplication(BaseModel):
    """Une demande de crédit, telle que reçue au guichet."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "no checking account",
                "duration": 24,
                "credit_history": "existing credits paid back duly till now",
                "purpose": "car (used)",
                "amount": 4500,
                "savings": "... < 100 DM",
                "employment_duration": "1 <= ... < 4 years",
                "installment_rate": 3,
                "present_residence": 2,
                "property": "car or other",
                "age": 32,
                "other_installment_plans": "none",
                "housing": "rent",
                "number_credits": 1,
                "job": "skilled employee/official",
                "people_liable": 1,
                "telephone": "yes",
                "other_debtors": "none",
            }
        }
    )

    status: _choices("status")
    duration: int = Field(ge=1, le=120, description="Durée du crédit en mois")
    credit_history: _choices("credit_history")
    purpose: _choices("purpose")
    amount: int = Field(gt=0, description="Montant demandé")
    savings: _choices("savings")
    employment_duration: _choices("employment_duration")
    installment_rate: int = Field(ge=1, le=4)
    other_debtors: _choices("other_debtors")
    present_residence: int = Field(ge=1, le=4)
    property: _choices("property")
    age: int = Field(ge=18, le=100)
    other_installment_plans: _choices("other_installment_plans")
    housing: _choices("housing")
    number_credits: int = Field(ge=1, le=10)
    job: _choices("job")
    people_liable: int = Field(ge=1, le=5)
    telephone: _choices("telephone")
    # Note : aucune variable sensible (sexe, nationalité) n'est demandée.


class Prediction(BaseModel):
    probability_default: float = Field(description="Probabilité de défaut (0 à 1)")
    decision: Literal["approve", "reject"]
    threshold: float
    model_version: str


class ModelStore:
    """Garde le modèle en mémoire : chargé une seule fois au démarrage."""

    def __init__(self) -> None:
        self.pipeline = None
        self.metadata: dict = {}

    def load(self, model_dir: Path) -> None:
        self.pipeline = joblib.load(model_dir / "model.joblib")
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        logger.info(
            "Modèle chargé : %s v%s (seuil %.2f)",
            self.metadata["model_name"],
            self.metadata["model_version"],
            self.metadata["threshold"],
        )

    @property
    def ready(self) -> bool:
        return self.pipeline is not None


store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Exécuté une fois au démarrage du serveur
    try:
        store.load(MODEL_DIR)
    except FileNotFoundError:
        logger.error("Aucun modèle dans %s : lancez `make train`.", MODEL_DIR)
    yield


app = FastAPI(
    title="Credit Risk API",
    description="Scoring du risque de défaut d'une demande de crédit.",
    version=__version__,
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def _score(applications: list[CreditApplication]) -> list[Prediction]:
    if not store.ready:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    df = pd.DataFrame([a.model_dump() for a in applications])
    probas = store.pipeline.predict_proba(df[config.FEATURES])[:, 1]
    threshold = store.metadata["threshold"]
    predictions = [
        Prediction(
            probability_default=round(float(p), 4),
            decision="reject" if p >= threshold else "approve",
            threshold=threshold,
            model_version=store.metadata["model_version"],
        )
        for p in probas
    ]
    if PREDICTION_LOG:
        _log_predictions(df, predictions)
    return predictions


def _log_predictions(df: pd.DataFrame, predictions: list[Prediction]) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with open(PREDICTION_LOG, "a", encoding="utf-8") as f:
        for row, pred in zip(df.to_dict(orient="records"), predictions, strict=True):
            record = {"timestamp": now, **row, **pred.model_dump()}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if store.ready else "degraded", "model_loaded": store.ready}


@app.get("/model/info")
def model_info() -> dict:
    if not store.ready:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return store.metadata


@app.post("/predict", response_model=Prediction)
def predict(application: CreditApplication) -> Prediction:
    return _score([application])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(applications: list[CreditApplication]) -> list[Prediction]:
    if not applications:
        raise HTTPException(status_code=400, detail="Liste vide")
    if len(applications) > 1000:
        raise HTTPException(status_code=413, detail="1000 demandes maximum")
    return _score(applications)
