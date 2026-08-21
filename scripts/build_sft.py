"""Point d'entrée CLI pour construire le dataset SFT.

Exemples :
  python scripts/build_sft.py                 # cible = config (5000), toutes sources
  python scripts/build_sft.py --target 2000
  python scripts/build_sft.py --no-preference-chosen
  python scripts/build_sft.py --sources medquad frenchmedmcqa mediqa
"""

from __future__ import annotations

import argparse
import sys

from chsa_triage.data.build_sft import build, format_stats
from chsa_triage.data.sources import REGISTRY


def main() -> int:
    p = argparse.ArgumentParser(description="Construction du dataset SFT CHSA.")
    p.add_argument("--sources", nargs="*", choices=list(REGISTRY.keys()), default=None)
    p.add_argument("--target", type=int, default=None, help="Nombre cible de paires (défaut : config).")
    p.add_argument("--no-preference-chosen", action="store_true",
                   help="Ne pas utiliser les réponses 'chosen' des préférences comme paires SFT.")
    args = p.parse_args()

    stats = build(
        sources=args.sources,
        target=args.target,
        include_preference_chosen=not args.no_preference_chosen,
    )
    print("\n== Dataset SFT construit ==")
    print(format_stats(stats))
    print(f"\nÉcrit : {stats['output_file']}")
    if not stats["reached_target"]:
        print("\nℹ️  Cible non atteinte : ingère davantage de lignes "
              "(`python scripts/ingest.py --max-rows 0`) puis relance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
