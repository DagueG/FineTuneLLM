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
