"""Schéma des métadonnées cliniques de triage.

Répond à l'exigence du brief : « Définir le schéma des métadonnées (symptômes, antécédents,
constantes, source, niveau de confiance) ». Ce schéma décrit :

- ce que l'agent COLLECTE d'un patient (symptômes, antécédents, constantes vitales) ;
- ce que l'agent PRODUIT (niveau de priorité, signes d'alerte, confiance, recommandation).

Il servira à l'inférence (étape déploiement) et à la traçabilité des interactions, et
structure aussi le jeu d'évaluation clinique.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Niveaux de priorité de triage (3 niveaux, alignés sur le brief).
PriorityLevel = Literal["urgence_vitale", "urgent", "non_urgent"]
PRIORITY_LEVELS: tuple[str, ...] = ("urgence_vitale", "urgent", "non_urgent")


class Vitals(BaseModel):
    """Constantes vitales (toutes optionnelles : rarement toutes disponibles au triage)."""

    heart_rate_bpm: Optional[int] = None          # fréquence cardiaque
    systolic_bp_mmhg: Optional[int] = None         # tension systolique
    diastolic_bp_mmhg: Optional[int] = None        # tension diastolique
    spo2_percent: Optional[float] = None           # saturation en oxygène
    temperature_c: Optional[float] = None          # température
    respiratory_rate: Optional[int] = None         # fréquence respiratoire


class TriageInput(BaseModel):
    """Informations collectées auprès du patient."""

    chief_complaint: str = ""                       # motif principal
    symptoms: list[str] = Field(default_factory=list)          # symptômes
    medical_history: list[str] = Field(default_factory=list)   # antécédents
    vitals: Vitals = Field(default_factory=Vitals)             # constantes
    age: Optional[int] = None
    language: Literal["fr", "en"] = "fr"


class TriageAssessment(BaseModel):
    """Évaluation produite par l'agent."""

    priority: PriorityLevel                          # niveau de priorité
    confidence: float = Field(ge=0.0, le=1.0)        # niveau de confiance (0..1)
    red_flags: list[str] = Field(default_factory=list)          # signes d'alerte détectés
    rationale: str = ""                              # explication du raisonnement
    recommendation: str = ""                         # recommandation d'orientation
    source: str = ""                                 # provenance (modèle, version, protocole)
