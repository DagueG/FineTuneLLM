# Schéma des métadonnées de triage

Défini et validé dans `src/chsa_triage/data/metadata.py` (pydantic).

## Entrée collectée — `TriageInput`
| Champ | Type | Description |
|---|---|---|
| chief_complaint | str | Motif principal de consultation |
| symptoms | list[str] | Symptômes rapportés |
| medical_history | list[str] | Antécédents |
| vitals | Vitals | Constantes vitales (voir ci-dessous) |
| age | int? | Âge |
| language | "fr"/"en" | Langue de l'échange |

### Constantes — `Vitals`
fréquence cardiaque (bpm), tension systolique/diastolique (mmHg), SpO2 (%),
température (°C), fréquence respiratoire. Toutes optionnelles.

## Évaluation produite — `TriageAssessment`
| Champ | Type | Description |
|---|---|---|
| priority | urgence_vitale / urgent / non_urgent | Niveau de priorité |
| confidence | float [0,1] | Niveau de confiance |
| red_flags | list[str] | Signes d'alerte détectés |
| rationale | str | Explication du raisonnement |
| recommendation | str | Recommandation d'orientation |
| source | str | Provenance (modèle/version/protocole) |

Ce schéma structure l'inférence (étape déploiement) et la traçabilité des interactions.
