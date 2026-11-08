"""Récupération, validation et découpage des données."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk import config
from credit_risk.schema import raw_schema

logger = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier (sa « carte d'identité »)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_data(dest: Path = config.RAW_DATA_PATH, force: bool = False) -> Path:
    """Télécharge le jeu de données s'il n'est pas déjà présent, puis vérifie
    son intégrité grâce à l'empreinte SHA-256."""
    if dest.exists() and not force:
        logger.info("Données déjà présentes : %s", dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Téléchargement depuis %s", config.DATA_URL)
        urllib.request.urlretrieve(config.DATA_URL, dest)  # noqa: S310

    actual = sha256_of(dest)
    if actual != config.DATA_SHA256:
        raise ValueError(
            f"Empreinte inattendue pour {dest} : {actual}. "
            "Le fichier a peut-être été modifié ou corrompu."
        )
    return dest


def load_raw(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Charge le CSV brut et le valide contre le contrat de données."""
    df = pd.read_csv(path)
    # lazy=True : on collecte TOUTES les erreurs avant d'échouer, pour avoir
    # un rapport complet plutôt que de corriger les problèmes un par un.
    return raw_schema.validate(df, lazy=True)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Crée la cible du modèle : 1 = défaut de paiement, 0 = remboursé."""
    out = df.copy()
    out[config.TARGET] = 1 - out[config.RAW_TARGET]
    return out.drop(columns=[config.RAW_TARGET])


def split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Sépare les données en un jeu d'entraînement et un jeu de test.

    `stratify` garde la même proportion de défauts (30 %) dans les deux jeux,
    sinon l'évaluation serait biaisée par le hasard du découpage.
    Les variables sensibles sont conservées dans X pour l'audit d'équité ;
    c'est le pipeline du modèle qui choisit de ne pas les utiliser.
    """
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = download_data()
    data = add_target(load_raw(path))
    logger.info(
        "%d lignes valides, taux de défaut = %.1f %%",
        len(data),
        100 * data[config.TARGET].mean(),
    )
