# Raccourcis : `make install`, `make data`, `make test`, `make lint`
PYTHON ?= python3
VENV = .venv
BIN = $(VENV)/bin

.PHONY: install data train mlflow-ui test lint format

install:  ## Crée l'environnement virtuel et installe le projet
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

data:  ## Télécharge et valide les données
	$(BIN)/python -m credit_risk.data

train:  ## Entraîne, compare les modèles et enregistre tout dans MLflow
	$(BIN)/python -m credit_risk.train

mlflow-ui:  ## Ouvre l'interface MLflow sur http://localhost:5000
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

test:  ## Lance les tests
	$(BIN)/pytest -v

lint:  ## Vérifie la qualité du code
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## Reformate automatiquement le code
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .
