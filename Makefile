# Raccourcis : `make install`, `make data`, `make test`, `make lint`
PYTHON ?= python3
VENV = .venv
BIN = $(VENV)/bin

.PHONY: install data train mlflow-ui api monitor monitor-demo docker-build up down test lint format

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

api:  ## Lance l'API en local sur http://localhost:8000/docs
	PREDICTION_LOG_PATH=logs/predictions.jsonl $(BIN)/uvicorn credit_risk.api:app --reload --port 8000

monitor:  ## Rapport de dérive sur les prédictions réelles de l'API
	$(BIN)/python -m credit_risk.monitor --current logs/predictions.jsonl

monitor-demo:  ## Rapport de dérive sur une crise économique simulée
	$(BIN)/python -m credit_risk.monitor --simulate

docker-build:  ## Construit l'image Docker de l'API
	docker build --target runtime -t credit-risk-api:local .

up:  ## Démarre MLflow + API avec Docker Compose
	docker compose up -d --build

down:  ## Arrête les conteneurs
	docker compose down

test:  ## Lance les tests
	$(BIN)/pytest -v

lint:  ## Vérifie la qualité du code
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## Reformate automatiquement le code
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .
