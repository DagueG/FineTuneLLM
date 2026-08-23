# ADR-0005 — Anonymisation RGPD (Presidio + repli regex)

Statut : accepté (étape 5)
Date : 2026

## Décisions
1. **Presidio** (Analyzer + Anonymizer) avec **spaCy** multilingue FR/EN comme moteur
   principal. Modèles `md`/`lg` demandés (brief), repli automatique sur `sm`.
2. **Filet regex** (email, téléphone, date, NIR, IBAN) si Presidio/spaCy indisponible :
   la pipeline s'exécute toujours, même en CI léger.
3. **3 stratégies** de masquage : `replace` (défaut), `mask`, `redact`.
4. On n'anonymise **pas le prompt système** (notre texte, sans PII patient).
5. **Contrôle qualité à deux niveaux** : exigence stricte de 0 résiduel sur les PII
   structurées (déterministes) ; suivi métrique des entités NER (PERSON/LOCATION), bruitées
   par les petits modèles sur du vocabulaire médical (faux positifs, pas des fuites).

## Justification
- Le brief impose Presidio et la détection nom/prénom : couvert par l'entité PERSON.
- Le repli garantit la robustesse (principe fil-rouge du projet).
- La distinction structuré/NER évite un faux sentiment d'échec dû aux limites des petits
  modèles, tout en gardant une garantie forte sur les PII déterministes.

## Limites
- Petits modèles = faux positifs sur termes médicaux -> recommander `md`/`lg` + allow-list.
- Données réelles patients : AIPD complète requise avant production (voir docs/RGPD.md).

## Traçabilité
Évènements `anonymize.start` / `data.anonymized` / `anonymize.done` dans le journal d'audit.
