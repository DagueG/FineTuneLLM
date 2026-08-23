# Conformité RGPD — POC agent IA de triage médical (CHSA)

Ce document justifie le processus de protection des données personnelles suivi pour le
POC. Il accompagne le livrable « dataset médical bilingue anonymisé ».

## 1. Contexte et responsabilités

- **Responsable de traitement** : Centre Hospitalier Saint-Aurélien (CHSA).
- **Rôle de l'équipe IA** : conception technique du POC (sous-traitant au sens RGPD).
- **Finalité** : entraîner un assistant d'**aide à la décision** au triage, destiné au
  personnel soignant. L'outil n'établit pas de diagnostic et ne remplace pas un soignant.

## 2. Nature des données utilisées

Le POC s'appuie **exclusivement sur des corpus médicaux publics** (MedQuAD, FrenchMedMCQA,
MedMCQA, UltraMedical-Preference), constitués de questions d'examen et de Q/R médicales
générales, **déjà largement dé-identifiées à la source**. Aucune donnée de patient réel du
CHSA n'est utilisée à ce stade. Les licences de chaque source sont documentées dans
l'inventaire d'ingestion (`data/raw/inventory.json`) et les ADR.

## 3. Principes RGPD appliqués

- **Minimisation** : on ne conserve que les champs utiles à l'entraînement (question,
  réponse, préférence, métadonnées non identifiantes).
- **Limitation des finalités** : usage strictement limité au POC de triage.
- **Anonymisation** : passe systématique de masquage des PII (défense en profondeur), même
  sur des données déjà publiques (voir §4).
- **Sécurité** : les secrets (tokens) ne sont jamais versionnés (`.env` ignoré par Git,
  séparé de la config). 
- **Auditabilité** : chaque transformation est tracée dans un journal d'audit **inviolable**
  (chaîne de hachage SHA-256), y compris l'évènement `data.anonymized`.

## 4. Processus d'anonymisation (Presidio)

Outil : **Microsoft Presidio** (`AnalyzerEngine` pour la détection, `AnonymizerEngine` pour
le masquage), NER multilingue via **spaCy** (`fr_core_news_md` / `en_core_web_lg`
recommandés ; repli automatique sur les variantes `sm`).

- **Entités ciblées** : `PERSON` (nom/prénom — exigence minimale), `EMAIL_ADDRESS`,
  `PHONE_NUMBER`, `DATE_TIME`, `LOCATION`, `IBAN_CODE`, `CREDIT_CARD`. Filet de sécurité
  regex (email, téléphone, date, NIR, IBAN) si Presidio/spaCy indisponible.
- **Stratégies** disponibles et testées : `replace` (`<PERSON>`), `mask` (`****`),
  `redact` (suppression). Défaut : `replace` (préserve la lisibilité et signale le masquage).
- **Périmètre** : seuls les contenus **utilisateur** et **assistant** (issus des corpus)
  sont anonymisés ; le prompt système (rédigé par nous, sans PII patient) est préservé.

## 5. Contrôle qualité du masquage

Après anonymisation, un **re-scan** d'un échantillon vérifie l'absence de PII résiduelle :

- **PII structurées** (email, téléphone, IBAN, carte, NIR) : exigence **stricte de 0
  résiduel** (détection déterministe). Statut `qc_passed`.
- **Entités NER** (PERSON, LOCATION, DATE_TIME) : suivies comme **métrique**. Les petits
  modèles spaCy produisent des faux positifs sur du vocabulaire médical (ex. « Fréquence
  cardiaque » pris pour une entité) ; ce n'est pas une fuite de PII. Recommandation :
  modèles `md`/`lg` en production, et éventuellement une *allow-list* de termes médicaux.

Le rapport complet est écrit dans `data/processed/anonymization_report.json`.

## 6. Limites et recommandations (avant tout traitement de données réelles)

- Ce POC ne traite pas de données patients réelles. **Avant la production** (données du
  CHSA) : réaliser une **AIPD (analyse d'impact)** complète, définir la base légale
  (mission d'intérêt public en santé), les durées de conservation, les droits des personnes,
  et mettre en place une supervision clinique et une revue humaine.
- Renforcer la détection avec des modèles NER médicaux dédiés et une allow-list, et étendre
  les entités (numéros de dossier, identifiants internes) au contexte hospitalier réel.

## 7. Traçabilité

Évènements d'audit émis : `anonymize.start`, `data.anonymized` (par fichier, avec compteurs
d'entités et résultat du QC), `anonymize.done`. Journal vérifiable via `verify_chain`.
