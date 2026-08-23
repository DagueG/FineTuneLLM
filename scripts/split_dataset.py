"""Découpe les datasets anonymisés SFT/DPO en train/val/test.

Entrées : data/processed/sft_anonymized.jsonl, dpo_anonymized.jsonl
Sorties : data/processed/splits/{sft,dpo}_{train,val,test}.jsonl + split_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chsa_triage.audit import AuditLogger
from chsa_triage.config import PROJECT_ROOT, load_config
from chsa_triage.data.split import leakage_report, stratified_split, write_splits


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _process(name: str, in_path: Path, out_dir: Path, ratios, seed, audit) -> dict:
    records = _read_jsonl(in_path)
    if not records:
        return {"dataset": name, "status": "absent", "input": in_path.name}
    splits = stratified_split(records, ratios=ratios, seed=seed)
    leak = leakage_report(splits)
    paths = write_splits(splits, out_dir, prefix=name)
    entry = {
        "dataset": name,
        "input": in_path.name,
        "counts": {k: len(v) for k, v in splits.items()},
        "leakage": leak,
        "leakage_ok": all(v == 0 for v in leak.values()),
        "files": {k: Path(p).name for k, p in paths.items()},
    }
    audit.log("data.split", entry)
    return entry


def main() -> int:
    p = argparse.ArgumentParser(description="Découpage train/val/test CHSA.")
    p.add_argument("--train", type=float, default=0.9)
    p.add_argument("--val", type=float, default=0.05)
    p.add_argument("--test", type=float, default=0.05)
    args = p.parse_args()
    ratios = (args.train, args.val, args.test)

    cfg = load_config()
    proc = PROJECT_ROOT / cfg.data.processed_dir
    out_dir = proc / "splits"
    audit = AuditLogger(PROJECT_ROOT / cfg.audit.log_path)
    seed = cfg.model.seed
    audit.log("split.start", {"ratios": ratios, "seed": seed})

    reports = []
    reports.append(_process("sft", proc / "sft_anonymized.jsonl", out_dir, ratios, seed, audit))
    reports.append(_process("dpo", proc / "dpo_anonymized.jsonl", out_dir, ratios, seed, audit))

    with (proc / "split_report.json").open("w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print("\n== Découpage train/val/test ==")
    for r in reports:
        if r.get("status") == "absent":
            print(f"  - {r['dataset']} : entrée absente ({r['input']}) — lance l'anonymisation d'abord")
            continue
        print(f"  - {r['dataset']:4} : {r['counts']} | anti-fuite : "
              f"{'OK' if r['leakage_ok'] else 'FUITE ! ' + str(r['leakage'])}")
    print(f"\nSorties dans : {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
