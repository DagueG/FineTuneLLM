# CHSA Triage — POC agent IA de triage médical

Proof of Concept d'un agent IA d'aide au **triage médical** pour le Centre Hospitalier
Saint-Aurélien (CHSA). Le projet couvrira : préparation d'un dataset médical bilingue
(FR/EN) anonymisé RGPD, fine-tuning supervisé (SFT + LoRA) et alignement par préférences
(DPO) de `Qwen3-1.7B-Base`, puis déploiement d'un endpoint (vLLM + FastAPI) avec CI/CD.

> ⚠️ **Usage** : outil d'**aide à la décision** pour du personnel soignant. Il ne pose pas
> de diagnostic et ne remplace pas un professionnel de santé.

## Étape 1 — Ossature du projet

Cette première étape pose les fondations transverses, **testables sans GPU ni réseau** :
- chargement de configuration (`config/config.yaml`) avec séparation stricte des secrets ;
- **journal d'audit inviolable** (chaîne de hachage) pour la traçabilité exigée par la mission ;
- tests automatisés et smoke test de bout en bout.

### Prérequis
- Python 3.10+ (testé sur 3.12) et Git.

### Installation (commandes à copier-coller)

```bash
cd chsa-triage
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### Lancer les tests

```bash
pytest
```

**Ce que vous devez voir si ça marche :** une ligne verte du type `6 passed in 0.xx s`.

### Lancer le smoke test

```bash
python scripts/smoke_test.py
```

**Ce que vous devez voir si ça marche :** plusieurs lignes `[config]`, `[secrets]`, `[audit]`
puis, à la fin :

```
✅ Smoke test réussi : config OK, audit OK, intégrité OK.
```

(le code de sortie est `0`).

### En cas d'échec
- `ModuleNotFoundError: chsa_triage` → l'environnement virtuel n'est pas activé, ou
  `pip install -e ".[dev]"` n'a pas été relancé. Réactivez `.venv` et réinstallez.
- `command not found: pytest` → activez `.venv` puis `pip install -e ".[dev]"`.
- Toute autre erreur → **copiez-collez-la telle quelle**, on diagnostique ensemble.

## Étape 2 — Ingestion des sources

Charge les 4 corpus, les **normalise** vers un schéma commun (`data/raw/<source>.jsonl`)
et produit un **inventaire** (`data/raw/inventory.json`). Chaque source a un **repli
synthétique** : si un téléchargement échoue, la pipeline continue au lieu de planter.

### Installer les dépendances data

```bash
pip install -r requirements/data.txt
```

### Lancer l'ingestion

Test hors-ligne immédiat (mini-jeux embarqués, aucune connexion requise) :

```bash
python scripts/ingest.py --force-fallback
```

Vrai téléchargement depuis le Hub (plafond 3000 lignes/source par défaut) :

```bash
python scripts/ingest.py
```

Options utiles : `--max-rows 500` (plus rapide), `--sources medquad frenchmedmcqa`
(sous-ensemble), `--max-rows 0` (illimité).

**Ce que tu dois voir si ça marche :** un tableau d'inventaire, puis soit
`✅ Toutes les sources chargées depuis le Hub.`, soit la liste des sources tombées en
repli avec la raison. Dans les deux cas, des fichiers `.jsonl` apparaissent dans `data/raw/`.

### Action qui dépend de toi

- Rien d'obligatoire pour tester en `--force-fallback`.
- Pour le **vrai** téléchargement : aucune clé n'est requise pour les sources publiques.
  Si une source s'avère *gated*, il faudra un **token Hugging Face** — mets-le dans `.env`
  (`CHSA_HF_TOKEN=...`) **au moment où l'erreur l'indiquera**, pas avant.
- ⚠️ **À trancher** : le créneau « MediQA » du brief est ambigu (défaut = MedMCQA). Dis-moi
  quelle source tu veux exactement, je la câble.

### En cas d'échec
- Une source en repli n'est **pas** un échec : c'est le comportement voulu. Colle-moi la
  ligne `hub_error` correspondante, je diagnostique (réseau, ID, version de `datasets`…).

## Étape 3 — Construction du dataset SFT

Transforme les JSONL normalisés en un dataset SFT **conversationnel** prêt à l'entraînement
(`data/processed/sft.jsonl`) : mise en forme avec prompt de triage, filtres qualité,
déduplication, équilibrage entre sources.

### Pré-requis
Avoir lancé l'étape 2 au moins une fois (des `.jsonl` dans `data/raw/`).

### Lancer la construction

```bash
python scripts\build_sft.py --target 5000
```

Options : `--target N`, `--no-preference-chosen` (ne pas réutiliser les réponses `chosen`
des préférences), `--sources ...`.

**Attendu :** un récapitulatif (total, rejets qualité, doublons, répartition par langue /
source / type) et un fichier `data\processed\sft.jsonl`. Si la cible n'est pas atteinte,
ingère plus de lignes : `python scripts\ingest.py --max-rows 0` puis relance.

Format d'une ligne : `{"messages": [{"role":"system",...},{"role":"user",...},
{"role":"assistant",...}], "meta": {...}}` — directement chargeable par
`datasets.load_dataset("json", data_files="data/processed/sft.jsonl")`.

## Étape 4 — Construction du dataset DPO

Construit les paires préférentielles (`data/processed/dpo.jsonl`) au format attendu par
TRL `DPOTrainer`, avec contrôles de cohérence (chosen ≠ rejected), déduplication et
plafonnement à la cible.

### Pré-requis
Avoir ingéré la source de préférences (`ultramedical_pref`) à l'étape 2.

### Lancer la construction

```bash
python scripts\build_dpo.py --target 3000
```

**Attendu :** un récap (total, rejets par motif, doublons, langues) + `data\processed\dpo.jsonl`.
Format d'une ligne : `{"prompt":[system,user], "chosen":[assistant], "rejected":[assistant], "meta":{...}}`.

> Pour viser la cible : `python scripts\ingest.py --sources ultramedical_pref --max-rows 0`
> puis relancer le build.

## Étape 5 — Anonymisation (Presidio) & RGPD

Anonymise les datasets SFT/DPO (PII), produit un rapport et un **contrôle qualité**.
Documentation de conformité : [`docs/RGPD.md`](docs/RGPD.md).

### Installer les dépendances + modèles spaCy

```bash
pip install -r requirements/anonymize.txt
python -m spacy download fr_core_news_md
python -m spacy download en_core_web_lg
```

(Repli accepté si tu préfères plus léger : `fr_core_news_sm` / `en_core_web_sm`.)

### Lancer l'anonymisation

```bash
python scripts\anonymize_dataset.py --strategy replace --mode training
```

Modes : `--mode training` (PII **structurées seulement** — préserve le contenu médical, à
utiliser pour les données d'entraînement issues de corpus publics) ; `--mode full` (masquage
NER complet PERSON/LOCATION/DATE — pour de vraies données patients). Stratégies : `replace`
(`<PERSON>`), `mask` (`****`), `redact`. Voir `docs/RGPD.md` §8 pour l'arbitrage.

**Attendu :** pour chaque fichier, le nombre d'entités masquées par type, et un contrôle
qualité : **PII structurées résiduelles = 0 → OK**. Les « entités NER résiduelles » sont une
métrique (faux positifs possibles des petits modèles sur du vocabulaire médical — voir RGPD.md).
Sorties : `data\processed\*_anonymized.jsonl` + `anonymization_report.json`.

> Si tu vois un `Traceback` mentionnant `tldextract`/`publicsuffix` **hors-ligne**, c'est un
> avertissement réseau d'une dépendance de Presidio, sans effet sur le résultat.

## Étape 6 — Splits, éval clinique, métadonnées & versionnement

Dernière brique « données ». Découpe train/val/test (anti-fuite), pose le jeu d'évaluation
clinique séparé, le schéma de métadonnées et une version reproductible du dataset.

### Pré-requis
Avoir lancé l'anonymisation (étape 5) : `*_anonymized.jsonl` présents.

### Découper puis versionner

```bash
python scripts\split_dataset.py
python scripts\version_dataset.py --version 1.0.0
```

**Attendu (split)** : par dataset, les effectifs `{train, val, test}` et `anti-fuite : OK`.
**Attendu (version)** : la liste des fichiers avec leur empreinte SHA-256, un manifeste
`data\processed\manifest.json` et une carte `docs\DATASET_CARD.md`.

- Schéma de métadonnées : `docs\METADATA.md` (+ `src\chsa_triage\data\metadata.py`).
- Jeu d'éval clinique séparé : `data\eval\clinical_eval.jsonl` (18 scénarios, 3 niveaux).

> Sur de très petits volumes (mode fallback), le split `val` peut être vide (strates trop
> petites) : c'est normal. Sur les 5 000/3 000 réels, le 90/5/5 donne des val/test corrects.

## Étape 7 — Fine-tuning supervisé (SFT) LoRA

Entraîne un adaptateur LoRA sur `Qwen3-1.7B-Base` à partir des splits SFT. **À lancer sur ta
machine GPU (ou Colab)** — pas dans mon environnement.

### Installer les dépendances (sur la machine GPU)

```bash
pip install -r requirements/train.txt
```

### Valider la pipeline AVANT le vrai run

```bash
python scripts\train_sft.py --dry-run     # prépare tout (données, profil) sans entraîner
python scripts\train_sft.py --smoke       # run minuscule (4 pas) — valide le chemin TRL réel
```

**Attendu (dry-run)** : un JSON avec `status: dry_run_ok`, le profil auto-détecté, le nombre
d'exemples. **Attendu (smoke)** : télécharge le modèle base (~3,4 Go, public, sans token),
tourne 4 pas, écrit dans `models\sft-lora\`.

### Vrai entraînement

```bash
python scripts\train_sft.py                       # profil auto-détecté depuis la VRAM
python scripts\train_sft.py --profile mid --epochs 3 --report-to wandb
python scripts\train_sft.py --resume              # reprise depuis le dernier checkpoint
```

Profils : `low` (≤8 Go, QLoRA 4-bit), `mid` (12–16 Go), `high` (≥24 Go). Sortie : adaptateur
LoRA + tokenizer dans `models\sft-lora\`.

> **Actions qui dépendent de toi** : installer la stack GPU ; si `--smoke` renvoie une erreur
> d'API (les versions `transformers/trl/peft` bougent vite), **colle-la-moi** — je l'ajuste.
> Le modèle base étant public, aucun token n'est requis pour entraîner.

## Étape 8 — Évaluation clinique + contrôles de sécurité

Évalue le modèle SFT sur le jeu d'éval clinique séparé : exactitude de la priorité, contrôles
de sécurité (recommandations dangereuses), rappel des signes d'alerte. Inclut un module
d'inférence propre (`infer/generate.py`) réutilisé par l'API.

### Tester la pipeline d'éval sans GPU

```bash
python scripts\evaluate.py --mock
```

### Évaluer le vrai modèle (sur GPU)

```bash
python scripts\evaluate.py --model-dir models\sft-lora
```

**Attendu** : un résumé (exactitude globale + par niveau, réponses non parsées, drapeaux de
sécurité) et un rapport complet `data\processed\eval_report.json` (avec les réponses, pour
audit clinique). Les cas avec drapeau de sécurité sont signalés.

### Générer une réponse en un appel propre (démo)

```python
from chsa_triage.infer.generate import TriageModel
tm = TriageModel.load("models/sft-lora")
print(tm.generate([
    {"role": "system", "content": "You are a medical triage decision-support assistant for CHSA."},
    {"role": "user", "content": "A 55-year-old man has chest pain radiating to the left arm with sweating."},
]))
```

## Structure

```
chsa-triage/
├── config/config.yaml         # config métier versionnée (pas de secrets)
├── src/chsa_triage/
│   ├── config.py              # config + secrets (séparés)
│   └── audit.py               # journal d'audit inviolable (hash chain)
├── scripts/smoke_test.py      # démo bout-en-bout sans GPU
├── tests/                     # tests automatisés
├── requirements/              # dépendances modulaires par étape
├── docs/decisions/            # décisions d'architecture (ADR)
├── data/ , logs/audit/        # données & journaux (non versionnés)
└── .env.example               # gabarit des secrets
```

## Roadmap (indicatif)
1. **[en cours]** Ossature (config, audit, tests).
2. Ingestion des sources (MediQA, FrenchMedMCQA, MedQuAD, UltraMedical-Preference) + fallbacks.
3. Construction du dataset SFT (~5 000 paires) au format JSONL/HF.
4. Construction des paires de préférences DPO.
5. Anonymisation Presidio + documentation RGPD.
6. Splits train/val/test + jeu d'éval clinique + schéma de métadonnées + versionnement.
7–10. SFT LoRA, évaluation & sécurité, DPO, comparaison.
11–15. FastAPI + vLLM (avec fallback), Docker, CI/CD GitHub Actions, déploiement, rapport.

## Étape 9 — Alignement par préférences (DPO)

Entraîne un adaptateur DPO **au-dessus du SFT** (fusion de l'adaptateur SFT + nouveau LoRA).

### Pré-requis
Le SFT est entraîné (`models\sft-lora\`), et les splits DPO existent (étape 6).

### Valider puis lancer

```bash
python scripts\train_dpo.py --dry-run     # vérifie données + présence de l'adaptateur SFT
python scripts\train_dpo.py --smoke       # run minuscule
python scripts\train_dpo.py               # vrai DPO (profil auto)
```

**Attendu** : JSON avec `status: trained` et un dossier `models\dpo-lora\`. En cas de
`CUDA out of memory`, utilise `--profile low` (QLoRA, `pip install bitsandbytes`).

> Le DPO part du modèle SFT : si l'adaptateur SFT est absent, le script le signale.

## Étape 10 — Comparaison SFT vs DPO

```bash
python scripts\compare_models.py --models models\sft-lora models\dpo-lora
```

**Attendu** : un tableau comparatif (SFT vs DPO) sur le jeu clinique + `comparison_report.json`.
Teste la pipeline sans GPU avec `--mock`.

## Fusion des adaptateurs (serving + éval correcte)

**Le DPO est entraîné au-dessus du SFT** : pour l'évaluer et le déployer correctement, on fusionne
les adaptateurs en modèles complets.

```bash
python scripts\merge_model.py --which both       # -> models\sft-merged, models\dpo-merged
```

Puis compare les modèles COMPLETS (empilement SFT+DPO correct) :

```bash
python scripts\compare_models.py --models models\sft-merged models\dpo-merged
```

`TriageModel.load(dir)` accepte aussi bien un adaptateur qu'un modèle complet fusionné.
