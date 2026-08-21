"""Point d'entrée CLI pour construire le dataset DPO.

Exemples :
  python scripts/build_dpo.py                # cible = config (3000)
  python scripts/build_dpo.py --target 1000
"""

from __future__ import annotations

import argparse
import sys

from chsa_triage.data.build_dpo import build, format_stats


def main() -> int:
    p = argparse.ArgumentParser(description="Construction du dataset DPO CHSA.")
    p.add_argument("--target", type=int, default=None, help="Nombre cible de paires (0 = toutes).")
    args = p.parse_args()

    stats = build(target=args.target)
    print("\n== Dataset DPO construit ==")
    print(format_stats(stats))
    print(f"\nÉcrit : {stats['output_file']}")
    if not stats["reached_target"]:
        print("\nℹ️  Cible non atteinte : ingère davantage de préférences "
              "(`python scripts/ingest.py --sources ultramedical_pref --max-rows 0`) puis relance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
