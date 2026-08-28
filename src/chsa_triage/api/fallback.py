"""Repli de triage sans modèle (rule-based).

Garantit que l'API répond TOUJOURS, même si le modèle est indisponible (pas de GPU, poids
absents, erreur de chargement). C'est un filet de sécurité de démo, volontairement prudent :
il escalade dès qu'un signe d'alerte est détecté et renvoie systématiquement au clinicien.
"""

from __future__ import annotations

_VITAL = [
    "chest pain", "douleur thoracique", "difficulty breathing", "détresse respiratoire",
    "shortness of breath", "unconscious", "inconscient", "unrespons", "severe bleeding",
    "hémorragie", "stroke", "avc", "face droop", "anaphyl", "not breathing", "ne respire",
    "saturation", "cyanos",
]
_URGENT = [
    "fever", "fièvre", "stiff neck", "raideur de la nuque", "fracture", "severe pain",
    "douleur intense", "vomiting blood", "dehydrat", "déshydrat", "asthma", "asthme",
]

_MESSAGES = {
    ("fr", "urgence_vitale"): "Signes potentiellement graves détectés. Prise en charge en "
        "urgence immédiate recommandée (appeler le 15 ou le 112). Cet avis est une aide à la "
        "décision et ne remplace pas un professionnel de santé.",
    ("fr", "urgent"): "Situation nécessitant une évaluation médicale rapide. Orientez le "
        "patient vers un soignant sans tarder. Aide à la décision, ne remplace pas un soignant.",
    ("fr", "non_urgent"): "Pas de signe d'alerte évident. Surveillance et conseils, avec "
        "réévaluation si aggravation. Aide à la décision, ne remplace pas un soignant.",
    ("en", "urgence_vitale"): "Potentially serious signs detected. Immediate emergency care "
        "recommended (call emergency services). This is decision support and does not replace "
        "a healthcare professional.",
    ("en", "urgent"): "Situation requires prompt medical evaluation. Refer to a clinician "
        "without delay. Decision support, not a replacement for a clinician.",
    ("en", "non_urgent"): "No obvious red flags. Monitoring and advice, with re-assessment if "
        "worsening. Decision support, not a replacement for a clinician.",
}


def _priority_for(text: str) -> str:
    low = text.lower()
    if any(kw in low for kw in _VITAL):
        return "urgence_vitale"
    if any(kw in low for kw in _URGENT):
        return "urgent"
    return "non_urgent"


def fallback_triage(text: str, language: str = "fr") -> str:
    """Retourne une réponse de triage prudente + un tag PRIORITY parseable."""
    lang = language if language in ("fr", "en") else "en"
    priority = _priority_for(text)
    msg = _MESSAGES[(lang, priority)]
    return f"{msg}\nPRIORITY: {priority}"
