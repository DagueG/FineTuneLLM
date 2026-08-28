---
title: CHSA Triage API
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# CHSA Triage API

Endpoint de démonstration de l'agent d'aide au triage médical (POC).
Outil d'**aide à la décision** — ne pose pas de diagnostic, ne remplace pas un soignant.

- `GET /health` — état du service
- `POST /triage` — `{ "text": "...", "language": "fr|en" }`
- `/docs` — interface Swagger

> Sur HF Spaces gratuit (CPU), l'API tourne en **mode repli** (rule-based). Pour servir le
> vrai modèle avec **vLLM**, voir `docs/DEPLOYMENT.md` (instance GPU).
