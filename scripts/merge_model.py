"""Fusionne les adaptateurs LoRA en modèles complets (pour serving + éval correcte).

Exemples :
  python scripts/merge_model.py --which both     # produit models/sft-merged et models/dpo-merged
  python scripts/merge_model.py --which sft
  python scripts/merge_model.py --which dpo
"""

from __future__ import annotations

import argparse
import sys

from chsa_triage.audit import AuditLogger
from chsa_triage.config import PROJECT_ROOT, load_config
from chsa_triage.train.merge import merge


def main() -> int:
    p = argparse.ArgumentParser(description="Fusion des adaptateurs LoRA (CHSA).")
    p.add_argument("--which", choices=["sft", "dpo", "both"], default="both")
    p.add_argument("--sft-dir", default="models/sft-lora")
    p.add_argument("--dpo-dir", default="models/dpo-lora")
    args = p.parse_args()

    cfg = load_config()
    base = cfg.model.base_model_id
    audit = AuditLogger(PROJECT_ROOT / cfg.audit.log_path)
    sft = str(PROJECT_ROOT / args.sft_dir)
    dpo = str(PROJECT_ROOT / args.dpo_dir)

    if args.which in ("sft", "both"):
        out = str(PROJECT_ROOT / "models" / "sft-merged")
        print(f"[merge] SFT -> {out}")
        merge(base, [sft], out)
        audit.log("model.merged", {"type": "sft", "out": out})
        print("  ok")

    if args.which in ("dpo", "both"):
        out = str(PROJECT_ROOT / "models" / "dpo-merged")
        print(f"[merge] SFT+DPO -> {out}")
        merge(base, [sft, dpo], out)
        audit.log("model.merged", {"type": "sft+dpo", "out": out})
        print("  ok")

    print("\nModèles complets prêts pour l'évaluation et le déploiement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
