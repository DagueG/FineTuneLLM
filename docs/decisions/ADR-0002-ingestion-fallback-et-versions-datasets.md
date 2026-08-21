# ADR-0002 — Ingestion des données : normalisation, fallback et version de `datasets`

Statut : accepté (étape 2)
Date : 2026

## Contexte

On agrège 4 corpus hétérogènes (2 langues, 3 formats). Les schémas amont varient et
la lib `datasets` a durci ses règles : **depuis `datasets ≥ 4.0`, les scripts de
chargement ne sont plus supportés** (`trust_remote_code` ne fonctionne plus pour ça).
Or `qanastek/frenchmedmcqa` est justement un dataset à script.

## Décisions

1. **Schéma canonique unique** (`data/schema.py`) pour toutes les sources : `qa`,
   `mcqa`, `preference`. On normalise **avant** d'écrire en JSONL. Bénéfice : une fois
   ingéré, notre dataset ne dépend plus des versions ni des scripts distants — les
   étapes SFT/DPO consomment notre JSONL, pas le Hub.
2. **`datasets>=2.19,<4.0`** pour l'ingestion (`requirements/data.txt`), afin de
   conserver le support des scripts (FrenchMedMCQA). Version testée : 3.6.0.
3. **Loaders tolérants** (`data/loaders.py`, `data/sources.py`) : recherche de colonnes
   insensible à la casse et à plusieurs noms possibles ; extraction de préférences
   tolérante à plusieurs schémas (colonnes plates ou listes de messages).
4. **Sources retenues** : `lavita/MedQuAD` (EN, QA), `qanastek/frenchmedmcqa` (FR, QCM,
   Apache-2.0), `TsinghuaC3I/UltraMedical-Preference` (EN, préférences, MIT), et le
   créneau « MediQA » (voir point ouvert) défauté sur `openlifescienceai/medmcqa`.

## Stratégie de repli (fallback)

Chaque source embarque un **mini-jeu synthétique** (`data/fallback/*.json`) imitant le
schéma réel. Si le Hub échoue (réseau, gated, script non supporté…), l'ingestion bascule
automatiquement dessus et **continue**. L'erreur réelle du Hub est journalisée et
remontée dans l'inventaire (`mode: fallback`, `hub_error: ...`). Testé pour de vrai :
sans réseau, `load_dataset` lève une erreur, elle est attrapée, et le repli prend le
relais (pipeline en succès, code 0).

## Traçabilité

Chaque source ingérée émet un évènement `data.ingested` dans le journal d'audit
(chaîne de hachage), avec langue, licence, mode et volumétrie. Auditabilité complète.

## Point ouvert (à confirmer par le métier)

« MediQA » est **ambigu** : plusieurs datasets MEDIQA existent (MEDIQA-QA, -RQE, -AnS,
-Chat…), souvent à script. Par défaut on utilise **MedMCQA** (EN, QCM, parquet propre)
comme source anglophone d'examen, facilement remplaçable (un seul point à changer dans
`REGISTRY`). À trancher avec le client selon l'intention réelle du brief.

## Conséquences

Ingestion reproductible, hors-ligne-safe, auditable. Les volumétries réelles seront
constatées côté GPU/poste de l'utilisateur (accès Hub) ; la structure de sortie est
identique en mode hub ou fallback.
