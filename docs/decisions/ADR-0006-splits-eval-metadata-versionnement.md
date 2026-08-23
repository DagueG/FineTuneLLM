# ADR-0006 — Splits, éval clinique, métadonnées et versionnement

Statut : accepté (étape 6)
Date : 2026

## Décisions
1. **Splits train/val/test stratifiés** par (source, langue), seed déterministe, ratios
   90/5/5 par défaut. Garantit un équilibre représentatif dans chaque split.
2. **Contrôle anti-fuite** systématique : aucune même question (hash normalisé) présente à
   la fois dans train et val/test. C'est le point de vigilance n°1 du brief.
3. **Jeu d'évaluation clinique séparé** (`data/eval/clinical_eval.jsonl`) : 18 scénarios
   bilingues **authorés indépendamment** (donc sans fuite depuis l'entraînement), couvrant
   les 3 niveaux de priorité (6/6/6). Sert à évaluer spécifiquement le comportement de triage
   (priorité + signes d'alerte), que les corpus QA/QCM ne fournissent pas directement.
   Marqué comme POC : à faire valider/enrichir par des cliniciens.
4. **Schéma de métadonnées** (`src/chsa_triage/data/metadata.py`) : `TriageInput` (motif,
   symptômes, antécédents, constantes, âge, langue) et `TriageAssessment` (priorité,
   confiance 0..1, signes d'alerte, raisonnement, recommandation). Validation pydantic.
5. **Versionnement sans dépendance externe** : manifeste `manifest.json` (SHA-256 + nombre
   de lignes par fichier + empreinte globale + commit git) et carte `docs/DATASET_CARD.md`.
   Poussable ensuite vers HF datasets (nécessite le token, optionnel).

## Conséquences
Dernière brique « données » : on dispose de splits propres, d'un jeu d'éval clinique
indépendant, d'un schéma métier documenté et d'une version reproductible et auditable.
Sur très petit volume (fallback), val peut être vide (strates minuscules) : normal.

## Correctif (split conscient des groupes)

Constat en conditions réelles : l'anonymisation peut **fusionner** des questions distinctes
en textes identiques (âges/nombres masqués, noms → `<PERSON>`), recréant des doublons APRÈS
la déduplication (faite à l'étape SFT, avant anonymisation). Un split naïf répartissait alors
ces doublons entre train/val/test → fuite.

Correctif : `stratified_split` est **conscient des groupes** — toutes les lignes de même
question (hash normalisé) sont assignées au même split. Résultat : 0 fuite garantie, sans
perte de données. Vérifié par un test dédié et sur simulation à grande échelle.
