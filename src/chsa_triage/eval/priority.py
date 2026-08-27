"""Extraction du niveau de priorité à partir d'une réponse en langage naturel.

Stratégie à deux niveaux :
1. on demande au modèle de terminer par une ligne `PRIORITY: <niveau>` (parsing fiable) ;
2. à défaut, heuristique par mots-clés (FR/EN), du plus grave au moins grave.
"""

from __future__ import annotations

import re
from typing import Optional

from ..data.metadata import PRIORITY_LEVELS  # ("urgence_vitale", "urgent", "non_urgent")

# Mots-clés par niveau (minuscules). Ordre de sévérité décroissant appliqué au parsing.
_KEYWORDS = {
    "urgence_vitale": [
        "urgence vitale", "life-threatening", "life threatening", "emergency",
        "immediately", "immediate", "samu", "appeler les secours", "call an ambulance",
        "ambulance", "resuscitat", "réanimation", "911", "112", " 15 ",
    ],
    "urgent": [
        "urgent", "within hours", "dans les heures", "rapidement", "sans délai",
        "prompt evaluation", "priority",
    ],
    "non_urgent": [
        "non urgent", "non-urgent", "differé", "différé", "routine", "self-care",
        "self care", "mild", "bénin", "benign", "surveillance à domicile", "monitor at home",
        "conseils", "safety-net", "safety net",
    ],
}

_TAG = re.compile(
    r"priorit[ye]\s*[:=]\s*(urgence[_ ]vitale|urgent|non[_ ]urgent)", re.IGNORECASE
)


def parse_priority(text: str) -> Optional[str]:
    """Retourne le niveau de priorité détecté, ou None."""
    if not text:
        return None
    low = text.lower()

    # 1) tag explicite
    m = _TAG.search(low)
    if m:
        tag = m.group(1).replace(" ", "_")
        if tag in PRIORITY_LEVELS:
            return tag

    # 2) heuristique par mots-clés, du plus grave au moins grave
    for level in ("urgence_vitale", "urgent", "non_urgent"):
        if any(kw in low for kw in _KEYWORDS[level]):
            return level
    return None
