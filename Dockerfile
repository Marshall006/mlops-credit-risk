# syntax=docker/dockerfile:1
# =============================================================================
# Image de l'API de scoring — construction « multi-stage »
#
#   Étape 1 « trainer » : installe les outils d'entraînement (MLflow…),
#                         télécharge les données et entraîne le modèle.
#   Étape 2 « runtime » : image finale LÉGÈRE, qui ne contient que l'API et le
#                         modèle entraîné copié depuis l'étape 1.
#
# Analogie : on cuisine dans une grande cuisine (étape 1), puis on ne livre
# que le plat dans une boîte (étape 2), pas toute la cuisine.
# =============================================================================

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CREDIT_RISK_HOME=/app
WORKDIR /app

# --- Étape 1 : entraînement ---------------------------------------------------
FROM base AS trainer
# On copie d'abord uniquement la description du projet : tant qu'elle ne
# change pas, Docker réutilise la couche « pip install » en cache.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[train]"
RUN python -m credit_risk.train

# --- Étape 2 : image finale de l'API -------------------------------------------
FROM base AS runtime
LABEL org.opencontainers.image.title="credit-risk-api" \
      org.opencontainers.image.description="API de scoring du risque de crédit" \
      org.opencontainers.image.source="https://github.com/Marshall006/mlops-credit-risk" \
      org.opencontainers.image.licenses="MIT"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]" \
 && useradd --create-home --uid 1000 appuser \
 && mkdir -p /app/logs && chown appuser /app/logs

COPY --from=trainer /app/models ./models

# Sécurité : on n'exécute jamais l'application en tant que root
USER appuser
EXPOSE 8000

# Docker vérifie toutes les 30 s que l'API répond
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "credit_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
