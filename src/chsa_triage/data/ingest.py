"""Ingestion : charge chaque source, normalise, écrit en JSONL et produit un inventaire.

Sorties (dans data/raw/) :
- `<source>.jsonl`  : exemples normalisés (schéma canonique).
- `inventory.json`  : récapitulatif par source (langue, licence, mode hub/fallback,
                      nombre de lignes brutes / exemples exploitables, éventuelle erreur).

Chaque étape est tracée dans le journal d'audit (chaîne de hachage) : on garde une
preuve auditable de « quelle source, combien de lignes, en mode réel ou repli ».
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..audit import AuditLogger
from ..config import PROJECT_ROOT, load_config
from .loaders import load_source
from .sources import REGISTRY, Source


def _rel(path: Path) -> str:
    """Chemin relatif au projet si possible, sinon chemin absolu (robuste hors projet/tests)."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def ingest_source(
    src: Source,
    out_dir: Path,
    audit: AuditLogger,
    max_rows: Optional[int],
    force_fallback: bool,
) -> dict:
    """Ingest une source et retourne son entrée d'inventaire."""
    rows, mode, error = load_source(src, max_rows=max_rows, force_fallback=force_fallback)

    examples: list[dict] = []
    for idx, row in enumerate(rows):
        ex = src.normalize(row, idx, src)
        if ex is not None and ex.is_usable():
            examples.append(ex.model_dump())

    out_path = out_dir / f"{src.key}.jsonl"
    _write_jsonl(out_path, examples)

    entry = {
        "source": src.key,
        "hf_id": src.hf_id,
        "language": src.language,
        "kind": src.kind,
        "license": src.license,
        "mode": mode,               # "hub" ou "fallback"
        "hub_error": error,         # None si tout va bien
        "raw_rows": len(rows),
        "usable_examples": len(examples),
        "output_file": _rel(out_path),
    }
    audit.log("data.ingested", entry)
    return entry


def run(
    sources: Optional[list[str]] = None,
    max_rows: Optional[int] = 3000,
    force_fallback: bool = False,
    config_path: Optional[str] = None,
) -> dict:
    """Lance l'ingestion de toutes les sources demandées.

    - `sources` : liste de clés (défaut : toutes). 
    - `max_rows` : plafond de lignes par source (None = tout).
    - `force_fallback` : force le mode repli (utile hors-ligne / pour les tests).
    """
    cfg = load_config(config_path)
    out_dir = PROJECT_ROOT / cfg.data.raw_dir
    audit = AuditLogger(PROJECT_ROOT / cfg.audit.log_path)

    keys = sources or list(REGISTRY.keys())
    audit.log("ingest.start", {"sources": keys, "max_rows": max_rows, "force_fallback": force_fallback})

    inventory = []
    for key in keys:
        src = REGISTRY[key]
        entry = ingest_source(src, out_dir, audit, max_rows, force_fallback)
        inventory.append(entry)

    inv_path = out_dir / "inventory.json"
    with inv_path.open("w", encoding="utf-8") as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2)
    audit.log("ingest.done", {"inventory_file": _rel(inv_path)})

    return {"inventory": inventory, "inventory_path": str(inv_path)}


def format_inventory_table(inventory: list[dict]) -> str:
    """Rend l'inventaire sous forme de tableau texte lisible."""
    header = f"{'source':<18}{'lang':<6}{'type':<12}{'mode':<10}{'brut':>7}{'exploit.':>10}"
    lines = [header, "-" * len(header)]
    for e in inventory:
        lines.append(
            f"{e['source']:<18}{e['language']:<6}{e['kind']:<12}{e['mode']:<10}"
            f"{e['raw_rows']:>7}{e['usable_examples']:>10}"
        )
    return "\n".join(lines)
