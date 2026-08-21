# ADR-0004 — Conception du dataset DPO

Statut : accepté (étape 4)
Date : 2026

## Décisions

1. **Format DPO conversationnel** (`prompt` = [system, user], `chosen` / `rejected` = [assistant]),
   consommable directement par TRL `DPOTrainer`. On réutilise le **même prompt système de
   triage** que le SFT : l'alignement DPO affine donc un comportement déjà installé au SFT,
   dans le même cadre conversationnel (cohérence prompt SFT ↔ DPO).
2. **Source** : `TsinghuaC3I/UltraMedical-Preference` (MIT), qui fournit déjà l'ordre
   chosen/rejected annoté par des LLM juges + revue humaine partielle. On fait confiance à cet
   étiquetage amont et on ajoute des contrôles de cohérence.
3. **Contrôles de cohérence** (« paires validées / non validées ») :
   - rejet si `chosen == rejected` (normalisé) — paire non informative ;
   - rejet si prompt / chosen / rejected vide ou hors bornes de longueur ;
   - déduplication sur le triplet (question, chosen, rejected).
4. **Cible** : `dpo_target_pairs` (défaut 3 000), plafonnement après mélange déterministe
   (`seed`). Modifiable via `--target` (0 = toutes les paires disponibles).

## Périmètre / limites

- Les **contrôles de sécurité** avancés (hallucinations, recommandations dangereuses) relèvent
  de l'étape d'évaluation (étape 8), pas de la simple construction du jeu.
- Le split train/val/test (DPO comme SFT) sera fait à l'étape 6, avec séparation stricte
  train ↔ évaluation.

## Traçabilité

Évènements `dpo.build.start` / `dpo.build.done` (volumétries, rejets par motif, langues,
cible atteinte). Statistiques détaillées dans `data/processed/dpo_stats.json`.

## Conséquences

`data/processed/dpo.jsonl` prêt pour le `DPOTrainer`. Sur le mini-jeu de repli (3 paires) la
cible n'est pas atteinte : normal. Sur données réelles, UltraMedical-Preference (~100k paires)
permet d'atteindre largement la cible ; augmenter l'ingestion si besoin.
