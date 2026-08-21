"""Schéma canonique des exemples, commun à toutes les sources.

Chaque source (QA, QCM, préférences) est normalisée vers ce format unique. C'est ce
schéma qui alimentera le SFT (étape 3) et le DPO (étape 4). Normaliser tôt nous
découple des formats amont et de la fragilité de la lib `datasets` : une fois écrit en
JSONL, notre dataset ne dépend plus des versions ni des scripts distants.

Types (`kind`) :
- "qa"         : question ouverte -> réponse de référence (`output`).
- "mcqa"       : question à choix multiples -> options (`input`) + bonne réponse (`output`).
- "preference" : prompt + réponse préférée (`chosen`) et rejetée (`rejected`), pour le DPO.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Kind = Literal["qa", "mcqa", "preference"]


class Example(BaseModel):
    """Exemple normalisé, indépendant de la source d'origine."""

    id: str
    source: str
    language: Literal["fr", "en"]
    kind: Kind

    # Champs pour qa / mcqa
    instruction: str = ""
    input: str = ""
    output: str = ""

    # Champs pour preference (DPO)
    chosen: str = ""
    rejected: str = ""

    # Métadonnées libres (licence, champs d'origine, scores, golden_answer, etc.)
    meta: dict[str, Any] = Field(default_factory=dict)

    def is_usable(self) -> bool:
        """Un exemple est exploitable s'il porte le contenu minimal attendu par son type."""
        if self.kind == "preference":
            return bool(self.instruction and self.chosen and self.rejected)
        return bool(self.instruction and self.output)
