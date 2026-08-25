"""Construction du dataset SFT (~5 000 paires instruction→réponse).

Pipeline :
1. lit les JSONL normalisés (data/raw/<source>.jsonl) produits à l'étape 2 ;
2. filtre la qualité (longueurs, exemples dégénérés) ;
3. met en forme conversationnelle (templates de triage) ;
4. déduplique (question normalisée) ;
5. équilibre entre sources (round-robin) et plafonne à la cible ;
6. écrit data/processed/sft.jsonl + sft_stats.json, et trace dans l'audit.

Le résultat est un JSONL au format conversationnel, directement chargeable par
`datasets.load_dataset("json", ...)` et consommable par TRL `SFTTrainer`.
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
from .templates import to_sft_messages

# Sources utilisables pour le SFT : QA + QCM. Les préférences sont réservées au DPO,
# mais leur réponse "chosen" (haute qualité) peut aussi servir de cible SFT (option).
SFT_KINDS = {"qa", "mcqa"}

_WS = re.compile(r"\s+")


def _norm_key(text: str) -> str:
    """Clé de déduplication : minuscules, espaces normalisés."""
    return _WS.sub(" ", text.strip().lower())


def _passes_quality(user: str, answer: str, min_q: int, min_a: int, max_len: int) -> bool:
    if len(user) < min_q or len(answer) < min_a:
        return False
    if len(user) > max_len or len(answer) > max_len:
        return False
    if _norm_key(user) == _norm_key(answer):
        return False
    return True


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


def build(
    sources: Optional[list[str]] = None,
    target: Optional[int] = None,
    include_preference_chosen: bool = True,
    max_share_mcqa: Optional[float] = 0.3,
    min_q: int = 10,
    min_a: int = 3,
    max_len: int = 6000,
    config_path: Optional[str] = None,
) -> dict:
    """Construit le dataset SFT et retourne les statistiques."""
    cfg = load_config(config_path)
    raw_dir = PROJECT_ROOT / cfg.data.raw_dir
    out_dir = PROJECT_ROOT / cfg.data.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditLogger(PROJECT_ROOT / cfg.audit.log_path)
    seed = cfg.model.seed
    target = target if target is not None else cfg.data.sft_target_pairs

    keys = sources or list(REGISTRY.keys())
    audit.log("sft.build.start", {"sources": keys, "target": target, "seed": seed,
                                  "include_preference_chosen": include_preference_chosen,
                                  "max_share_mcqa": max_share_mcqa})

    # 1-3. Lecture, filtre qualité, mise en forme — regroupés par source.
    by_source: dict[str, list[dict]] = {}
    stats_raw: dict[str, int] = {}
    dropped_quality = 0
    for key in keys:
        src = REGISTRY[key]
        use = src.kind in SFT_KINDS or (src.kind == "preference" and include_preference_chosen)
        if not use:
            continue
        rows = _read_jsonl(raw_dir / f"{key}.jsonl")
        stats_raw[key] = len(rows)
        kept = []
        for row in rows:
            rec = to_sft_messages(row)
            if rec is None:
                dropped_quality += 1
                continue
            user = rec["messages"][1]["content"]
            answer = rec["messages"][2]["content"]
            if not _passes_quality(user, answer, min_q, min_a, max_len):
                dropped_quality += 1
                continue
            kept.append(rec)
        by_source[key] = kept

    # 4. Déduplication globale (sur la question normalisée), première occurrence gardée.
    seen: set[str] = set()
    dropped_dup = 0
    for key, lst in by_source.items():
        deduped = []
        for rec in lst:
            k = hashlib.sha256(_norm_key(rec["messages"][1]["content"]).encode()).hexdigest()
            if k in seen:
                dropped_dup += 1
                continue
            seen.add(k)
            deduped.append(rec)
        by_source[key] = deduped

    # 5. Équilibrage : on plafonne la part de QCM (comportement cible = réponses rédigées,
    #    pas des questions à choix multiples), le reste vient des sources rédigées (QA + chosen).
    rng = random.Random(seed)
    for lst in by_source.values():
        rng.shuffle(lst)
    order = [k for k in keys if by_source.get(k)]
    mcqa_keys = [k for k in order if REGISTRY[k].kind == "mcqa"]
    other_keys = [k for k in order if REGISTRY[k].kind != "mcqa"]

    def _round_robin(src_keys: list[str], budget: int) -> list[dict]:
        picked: list[dict] = []
        pos = {k: 0 for k in src_keys}
        while len(picked) < budget and any(pos[k] < len(by_source[k]) for k in src_keys):
            for k in src_keys:
                if pos[k] < len(by_source[k]):
                    picked.append(by_source[k][pos[k]])
                    pos[k] += 1
                    if len(picked) >= budget:
                        break
        return picked

    mcqa_budget = int(max_share_mcqa * target) if max_share_mcqa is not None else target
    mcqa_part = _round_robin(mcqa_keys, mcqa_budget)
    other_part = _round_robin(other_keys, target - len(mcqa_part))
    result = mcqa_part + other_part
    rng.shuffle(result)
    result = result[:target]

    # 6. Écriture + stats + audit.
    out_path = out_dir / "sft.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in result:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_lang: dict[str, int] = {}
    by_src_final: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for rec in result:
        m = rec["meta"]
        by_lang[m["language"]] = by_lang.get(m["language"], 0) + 1
        by_src_final[m["source"]] = by_src_final.get(m["source"], 0) + 1
        by_kind[m["kind"]] = by_kind.get(m["kind"], 0) + 1

    stats = {
        "target": target,
        "total": len(result),
        "raw_per_source": stats_raw,
        "dropped_quality": dropped_quality,
        "dropped_duplicates": dropped_dup,
        "final_by_source": by_src_final,
        "final_by_language": by_lang,
        "final_by_kind": by_kind,
        "output_file": str(out_path),
        "reached_target": len(result) >= target,
    }
    with (out_dir / "sft_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    audit.log("sft.build.done", {k: stats[k] for k in
                                 ("total", "dropped_quality", "dropped_duplicates",
                                  "final_by_language", "reached_target")})
    return stats


def format_stats(stats: dict) -> str:
    lines = [
        f"Total SFT       : {stats['total']} (cible {stats['target']}, "
        f"atteinte : {'oui' if stats['reached_target'] else 'non'})",
        f"Rejetés qualité : {stats['dropped_quality']}",
        f"Doublons retirés: {stats['dropped_duplicates']}",
        f"Par langue      : {stats['final_by_language']}",
        f"Par source      : {stats['final_by_source']}",
        f"Par type        : {stats['final_by_kind']}",
    ]
    return "\n".join(lines)
