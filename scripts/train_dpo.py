"""Point d'entrée CLI pour l'alignement DPO (au-dessus du SFT).

Exemples :
  python scripts/train_dpo.py --dry-run
  python scripts/train_dpo.py --smoke
  python scripts/train_dpo.py                      # vrai DPO, profil auto
  python scripts/train_dpo.py --beta 0.1 --epochs 1 --report-to wandb
"""

from __future__ import annotations

import argparse
import json
import sys

from chsa_triage.train.hf_utils import PROFILES
from chsa_triage.train.dpo import run_dpo


def main() -> int:
    p = argparse.ArgumentParser(description="DPO — alignement par préférences (CHSA).")
    p.add_argument("--profile", choices=[k for k in PROFILES if k != "smoke"], default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--report-to", choices=["none", "wandb", "tensorboard"], default="none")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sft-dir", default="models/sft-lora")
    args = p.parse_args()

    meta = run_dpo(
        profile_name=args.profile, smoke=args.smoke, dry_run=args.dry_run,
        epochs=args.epochs, learning_rate=args.learning_rate, beta=args.beta,
        report_to=args.report_to, resume=args.resume, sft_dir=args.sft_dir,
    )
    print("\n== DPO ==")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
