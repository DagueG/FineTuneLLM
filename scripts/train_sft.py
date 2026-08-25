"""Point d'entrée CLI pour le SFT LoRA de Qwen3-1.7B-Base.

Exemples :
  python scripts/train_sft.py --dry-run          # prépare tout, n'entraîne pas (validation pipeline)
  python scripts/train_sft.py --smoke            # run minuscule (4 pas), CPU-friendly
  python scripts/train_sft.py                    # vrai entraînement, profil auto-détecté
  python scripts/train_sft.py --profile mid --epochs 3 --report-to wandb
  python scripts/train_sft.py --resume           # reprend depuis le dernier checkpoint
"""

from __future__ import annotations

import argparse
import json
import sys

from chsa_triage.train.hf_utils import PROFILES
from chsa_triage.train.sft import run_sft


def main() -> int:
    p = argparse.ArgumentParser(description="SFT LoRA — Qwen3-1.7B-Base (CHSA).")
    p.add_argument("--profile", choices=[k for k in PROFILES if k != "smoke"], default=None,
                   help="Force un profil matériel (sinon auto-détecté depuis la VRAM).")
    p.add_argument("--smoke", action="store_true", help="Run minuscule pour valider la pipeline.")
    p.add_argument("--dry-run", action="store_true", help="Prépare tout sans entraîner.")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--report-to", choices=["none", "wandb", "tensorboard"], default="none")
    p.add_argument("--resume", action="store_true", help="Reprend depuis le dernier checkpoint.")
    args = p.parse_args()

    meta = run_sft(
        profile_name=args.profile, smoke=args.smoke, dry_run=args.dry_run,
        epochs=args.epochs, learning_rate=args.learning_rate,
        report_to=args.report_to, resume=args.resume,
    )
    print("\n== SFT LoRA ==")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
