# ADR-0009 — Alignement par préférences (DPO)

Statut : accepté (étape 9)
Date : 2026

## Décisions
1. **DPO au-dessus du SFT** : on charge base + adaptateur SFT, on **fusionne**
   (`merge_and_unload`) pour obtenir le modèle de politique, puis on entraîne un **nouvel
   adaptateur LoRA** par DPO. Le modèle de référence = le même modèle adaptateur désactivé
   (pas de second modèle en mémoire).
2. **Données** : nos paires DPO conversationnelles (splits `dpo_train`/`dpo_val`), mêmes
   prompts système de triage que le SFT (cohérence).
3. **Hyperparamètres DPO** : `beta=0.1`, `learning_rate=5e-6` (plus faible que le SFT),
   `epochs=1`. Profils VRAM réutilisés.
4. **Mêmes garde-fous que le SFT** : construction de config résiliente aux versions
   (`_construct`), `--dry-run`/`--smoke`, reprise, pont token HF.

## Conséquences
Adaptateur DPO dans `models/dpo-lora/`. Comparaison SFT vs DPO à l'étape 10 (même harnais
d'évaluation), pour quantifier l'apport de l'alignement.

## Limites de test
Comme le SFT : le run réel se valide côté GPU (torch/Hub indisponibles ici). Sont testés :
logique de profils, dry-run, filtrage des arguments de config.

## Étape 10 — Comparaison SFT vs DPO
`scripts/compare_models.py` évalue plusieurs modèles sur le MÊME harnais clinique et sort un
tableau comparatif (exactitude globale + par niveau, sécurité, non parsées, rappel signes
d'alerte) + `data/processed/comparison_report.json`. Permet de quantifier l'apport du DPO.
