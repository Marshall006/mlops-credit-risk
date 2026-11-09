"""Configuration centrale du projet.

Tout ce qui pourrait changer (chemins, URL des données, liste des colonnes)
est regroupé ici, pour ne jamais avoir de « valeurs magiques » éparpillées
dans le code.
"""

import os
from pathlib import Path

# --- Chemins -----------------------------------------------------------------
# ROOT_DIR = dossier de travail du projet. Par défaut, la racine du dépôt (deux
# niveaux au-dessus de src/credit_risk/) ; dans un conteneur Docker, le paquet
# est installé ailleurs, donc on le fixe avec la variable CREDIT_RISK_HOME.
_repo_root = Path(__file__).resolve().parents[2]
_default_root = _repo_root if (_repo_root / "pyproject.toml").exists() else Path.cwd()
ROOT_DIR = Path(os.getenv("CREDIT_RISK_HOME", str(_default_root)))
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "german_credit.csv"
MODELS_DIR = ROOT_DIR / "models"

# --- Source des données --------------------------------------------------------
# German Credit Dataset (UCI) : 1 000 demandes de crédit, 20 variables,
# et une cible indiquant si le crédit a été bien remboursé.
DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
# Empreinte SHA-256 du fichier attendu : si le fichier téléchargé est
# différent (corrompu, modifié), on refuse de l'utiliser.
DATA_SHA256 = "ac003d1158e44018e3a1416381cebde213f96df542b2191910baec3b2afa7616"

# --- Colonnes ------------------------------------------------------------------
# Colonne brute : 1 = bon payeur, 0 = mauvais payeur.
RAW_TARGET = "credit_risk"
# Cible du modèle : 1 = défaut de paiement (c'est l'événement qu'on veut détecter).
TARGET = "is_default"

NUMERIC_FEATURES = [
    "duration",
    "amount",
    "installment_rate",
    "present_residence",
    "age",
    "number_credits",
    "people_liable",
]

CATEGORICAL_FEATURES = [
    "status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
]

# IA responsable : ces variables (sexe/situation familiale, nationalité) ne
# doivent pas servir à décider d'un crédit. On les garde dans les données pour
# pouvoir auditer l'équité du modèle, mais on ne les donne JAMAIS au modèle.
SENSITIVE_FEATURES = ["personal_status_sex", "foreign_worker"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2
