# Raccourcis : `make install`, `make data`, `make test`, `make lint`
PYTHON ?= python3
VENV = .venv
BIN = $(VENV)/bin

.PHONY: install data test lint format

install:  ## Crée l'environnement virtuel et installe le projet
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

data:  ## Télécharge et valide les données
	$(BIN)/python -m credit_risk.data

test:  ## Lance les tests
	$(BIN)/pytest -v

lint:  ## Vérifie la qualité du code
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## Reformate automatiquement le code
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .
