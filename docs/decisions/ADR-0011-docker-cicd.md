# ADR-0011 — Conteneurisation (Docker) et CI/CD (GitHub Actions)

Statut : accepté (étapes 12-13)
Date : 2026

## Décisions
1. **Docker** : image `python:3.12-slim` servant l'API FastAPI. Par défaut **légère** (sans
   torch) → l'API démarre en **mode repli** : un endpoint de démo qui ne tombe jamais.
   Le service du vrai modèle (GPU) est documenté (ajout de `requirements/train.txt` + montage
   de `models/`). Healthcheck sur `/health`. `docker-compose.yml` pour un lancement simple.
2. **CI/CD (GitHub Actions)** : à chaque push/PR,
   - **job `tests`** : installe base+dev+serve (léger, pas de torch — aucun test n'importe de
     dépendance lourde) et lance les 53 tests ;
   - **job `docker-build`** : construit l'image et fait un **smoke test** du conteneur
     (`/health`), garantissant que l'endpoint démarre.
3. **Sécurité** : aucun secret dans l'image ni le workflow ; les tokens passent par les
   secrets GitHub / variables d'environnement au déploiement.

## Conséquences
Tests et build automatisés à chaque modification (maintenabilité, non-régression). L'image
est déployable telle quelle en repli ; le service GPU/vLLM est branché à l'étape cloud.
