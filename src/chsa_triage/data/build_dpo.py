"""Construction du dataset DPO (paires préférentielles chosen / rejected).

Pipeline :
1. lit les JSONL normalisés des sources de type "preference" (ex. UltraMedical) ;
2. met en forme au format DPO conversationnel (mêmes prompts de triage que le SFT) ;
3. contrôles de cohérence : chosen ≠ rejected, non-vides, longueurs ;
4. déduplication (triplet prompt/chosen/rejected) ;
5. plafonne à la cible ;
6. écrit data/processed/dpo.jsonl + dpo_stats.json, et trace dans l'audit.

Sortie directement consommable par TRL `DPOTrainer`.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Optional

from ..audit import AuditLogger
from ..config import PROJECT_ROOT, load_config
from .sources import REGISTRY
from .templates import to_dpo_record

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _assistant_text(messages: list[dict]) -> str:
    return messages[-1]["content"] if messages else ""


def _user_text(messages: list[dict]) -> str:
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


def build(
    sources: Optional[list[str]] = None,
    target: Optional[int] = None,
    min_len: int = 3,
    max_len: int = 8000,
    config_path: Optional[str] = None,
) -> dict:
    """Construit le dataset DPO et retourne les statistiques."""
    cfg = load_config(config_path)
    raw_dir = PROJECT_ROOT / cfg.data.raw_dir
    out_dir = PROJECT_ROOT / cfg.data.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditLogger(PROJECT_ROOT / cfg.audit.log_path)
    seed = cfg.model.seed
    target = target if target is not None else cfg.data.dpo_target_pairs

    # Sources de préférences uniquement.
    keys = sources or [k for k, s in REGISTRY.items() if s.kind == "preference"]
    audit.log("dpo.build.start", {"sources": keys, "target": target, "seed": seed})

    records: list[dict] = []
    raw_total = 0
    dropped_invalid = 0        # prompt/chosen/rejected manquant
    dropped_identical = 0      # chosen == rejected
    dropped_length = 0
    for key in keys:
        rows = _read_jsonl(raw_dir / f"{key}.jsonl")
        raw_total += len(rows)
        for row in rows:
            rec = to_dpo_record(row)
            if rec is None:
                dropped_invalid += 1
                continue
            chosen = _assistant_text(rec["chosen"])
            rejected = _assistant_text(rec["rejected"])
            user = _user_text(rec["prompt"])
            if _norm(chosen) == _norm(rejected):
                dropped_identical += 1
                continue
            if not (min_len <= len(chosen) <= max_len) or not (min_len <= len(rejected) <= max_len):
                dropped_length += 1
                continue
            if len(user) < min_len:
                dropped_invalid += 1
                continue
            records.append(rec)

    # 4. Déduplication sur le triplet (question, chosen, rejected).
    seen: set[str] = set()
    dropped_dup = 0
    unique: list[dict] = []
    for rec in records:
        key = hashlib.sha256(
            (_norm(_user_text(rec["prompt"])) + "||" +
             _norm(_assistant_text(rec["chosen"])) + "||" +
             _norm(_assistant_text(rec["rejected"]))).encode()
        ).hexdigest()
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)
        unique.append(rec)

    # 5. Plafonnement à la cible (mélange déterministe).
    rng = random.Random(seed)
    rng.shuffle(unique)
    result = unique[:target] if target and target > 0 else unique

    # 6. Écriture + stats + audit.
    out_path = out_dir / "dpo.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in result:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_lang: dict[str, int] = {}
    for rec in result:
        lg = rec["meta"]["language"]
        by_lang[lg] = by_lang.get(lg, 0) + 1

    stats = {
        "target": target,
        "total": len(result),
        "raw_rows": raw_total,
        "dropped_invalid": dropped_invalid,
        "dropped_identical": dropped_identical,
        "dropped_length": dropped_length,
        "dropped_duplicates": dropped_dup,
        "final_by_language": by_lang,
        "output_file": str(out_path),
        "reached_target": (target is not None and target > 0 and len(result) >= target),
    }
    with (out_dir / "dpo_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    audit.log("dpo.build.done", {k: stats[k] for k in
                                 ("total", "dropped_identical", "dropped_duplicates",
                                  "final_by_language", "reached_target")})
    return stats


def format_stats(stats: dict) -> str:
    lines = [
        f"Total DPO         : {stats['total']} (cible {stats['target']}, "
        f"atteinte : {'oui' if stats['reached_target'] else 'non'})",
        f"Lignes brutes     : {stats['raw_rows']}",
        f"Rejets invalides  : {stats['dropped_invalid']}",
        f"chosen==rejected  : {stats['dropped_identical']}",
        f"Rejets longueur   : {stats['dropped_length']}",
        f"Doublons retirés  : {stats['dropped_duplicates']}",
        f"Par langue        : {stats['final_by_language']}",
    ]
    return "\n".join(lines)
