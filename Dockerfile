# Image de l'API de triage CHSA (FastAPI).
# Par défaut : image légère qui sert l'API en mode REPLI (sans torch/poids) — idéale pour
# une démo d'endpoint qui ne tombe jamais. Pour servir le vrai modèle, voir le commentaire
# plus bas (ajout de torch/transformers + montage du dossier models/).
FROM python:3.12-slim

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python (couche cache : requirements avant le code)
COPY pyproject.toml README.md ./
COPY requirements/ requirements/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements/serve.txt

# Code + config
COPY src/ src/
COPY config/ config/
RUN pip install --no-cache-dir -e .

# Pour servir le VRAI modèle (GPU) : décommenter et fournir le dossier models/ au run.
# RUN pip install --no-cache-dir -r requirements/train.txt

ENV CHSA_HOST=0.0.0.0 CHSA_PORT=8000 CHSA_DEVICE=cpu
EXPOSE 8000

# Healthcheck : l'endpoint /health doit répondre
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fs http://localhost:8000/health || exit 1

# Lancement direct d'uvicorn ; utilise $PORT si fourni par la plateforme (Render, Cloud Run…)
CMD ["sh", "-c", "uvicorn chsa_triage.api.app:app --host ${CHSA_HOST} --port ${PORT:-${CHSA_PORT}}"]
