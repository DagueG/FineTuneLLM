"""Chargement d'une source avec repli gracieux (fallback).

Stratégie : on tente d'abord le téléchargement réel depuis le Hub Hugging Face. En cas
d'échec (réseau coupé, dataset déplacé, gated, script non supporté par datasets>=4…),
on bascule automatiquement sur un mini-jeu synthétique embarqué dans le dépôt, afin que
la pipeline s'exécute TOUJOURS — y compris hors-ligne et en démo.

`datasets` est importé paresseusement : le module reste utilisable même si la lib n'est
pas installée (on part alors directement en fallback).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from ..config import PROJECT_ROOT
from .sources import Source

FALLBACK_DIR = PROJECT_ROOT / "data" / "fallback"


def _iter_hub_rows(src: Source, max_rows: Optional[int]) -> list[dict[str, Any]]:
    """Récupère jusqu'à `max_rows` lignes brutes depuis le Hub. Peut lever une exception."""
    import datasets  # import paresseux, volontaire

    load_kwargs: dict[str, Any] = {"split": src.split}
    if src.trust_remote_code:
        # Utile pour les datasets à script (ex. FrenchMedMCQA) avec datasets<4.
        load_kwargs["trust_remote_code"] = True

    # On tente le mode streaming (rapide, pas de téléchargement complet).
    try:
        ds = datasets.load_dataset(src.hf_id, streaming=True, **load_kwargs)
        it: Iterable[dict] = ds if max_rows is None else itertools.islice(ds, max_rows)
        return [dict(r) for r in it]
    except Exception:
        # Repli non-streaming (certaines sources ne streament pas).
        ds = datasets.load_dataset(src.hf_id, **load_kwargs)
        rows = [dict(r) for r in ds]
        return rows if max_rows is None else rows[:max_rows]


def _load_fallback_rows(src: Source) -> list[dict[str, Any]]:
    path = FALLBACK_DIR / src.fallback_file
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_source(
    src: Source,
    max_rows: Optional[int] = None,
    force_fallback: bool = False,
) -> tuple[list[dict[str, Any]], str, Optional[str]]:
    """Charge les lignes brutes d'une source.

    Retourne (rows, mode, error) où :
    - mode  = "hub" (téléchargement réel) ou "fallback" (mini-jeu synthétique).
    - error = message d'erreur du Hub si on a dû basculer en fallback, sinon None.
    """
    if not force_fallback:
        try:
            rows = _iter_hub_rows(src, max_rows)
            if rows:
                return rows, "hub", None
            return _load_fallback_rows(src), "fallback", "hub a renvoyé 0 ligne"
        except Exception as e:  # noqa: BLE001 — on veut vraiment tout attraper ici
            return _load_fallback_rows(src), "fallback", f"{type(e).__name__}: {e}"
    return _load_fallback_rows(src), "fallback", "force_fallback=True"
