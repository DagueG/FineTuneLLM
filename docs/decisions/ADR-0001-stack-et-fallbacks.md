# ADR-0001 — Stack technique et stratégie de repli (fallbacks)

Statut : accepté (étape 1)
Date : 2026

## Contexte

POC d'agent IA de triage médical pour le CHSA, en 4 semaines : préparation des données
bilingues, SFT (LoRA) puis alignement DPO d'un `Qwen3-1.7B-Base`, déploiement d'un endpoint
vLLM + FastAPI avec CI/CD GitHub Actions. Exigences transverses fortes : conformité RGPD,
traçabilité auditable, et robustesse de la démo.

## Décisions

1. **Langage & structure** : Python 3.10+, layout `src/`, packaging `pyproject.toml`,
   installation editable pour des imports propres et testables.
2. **Modèle** : `Qwen/Qwen3-1.7B-Base` (Apache-2.0), conforme à la mission. Vérifié présent
   sur le Hub. Des modèles plus récents existent (famille Qwen3.5) mais on respecte la
   spécification pour la comparabilité des résultats.
3. **Post-training** : Hugging Face **TRL** (`SFTTrainer`, `DPOTrainer`) + **PEFT/LoRA**.
   TRL est passé en v1.x (API stabilisée pour SFT/DPO/GRPO) — les versions exactes seront
   épinglées et **testées** au moment des étapes d'entraînement, pas devinées à l'avance.
4. **Configuration vs secrets** : config métier non sensible dans `config/config.yaml`
   (versionnée) ; secrets via variables d'environnement `CHSA_*` / `.env` non versionné.
   Séparation nette pour l'hygiène RGPD/sécurité.
5. **Traçabilité** : journal d'audit **append-only** JSONL avec **chaîne de hachage
   SHA-256** (inviolabilité vérifiable). Répond à « traçabilité de chaque interaction » et
   « auditabilité de chaque transformation de données ».

## Stratégie de repli (fallbacks) — pour qu'une démo ne tombe jamais en panne

- **Datasets indisponibles** (réseau/Hub) : chaque source aura un mini-jeu synthétique de
  secours pour que la pipeline s'exécute partout (à brancher à l'étape données).
- **Suivi d'entraînement** : W&B optionnel avec bascule `WANDB_MODE=offline` ; à défaut,
  logs locaux. Aucune dépendance bloquante à un compte externe.
- **Inférence** : endpoint vLLM en cible ; repli `transformers` si vLLM/GPU indisponible,
  pour que la démo réponde toujours (à brancher à l'étape déploiement).
- **Config absente** : retour aux valeurs par défaut plutôt qu'un crash.

## Conséquences

Ossature testable immédiatement sans GPU ni réseau. Les briques lourdes (téléchargements,
entraînement, serving) sont isolées dans des étapes ultérieures avec leurs propres fichiers
de dépendances (`requirements/*.txt`) et seront validées sur votre environnement GPU.
