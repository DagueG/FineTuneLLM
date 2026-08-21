"""Point d'entrée CLI pour l'ingestion des données.

Exemples :
  python scripts/ingest.py                      # tente le Hub, plafond 3000 lignes/source
  python scripts/ingest.py --max-rows 500       # plus rapide
  python scripts/ingest.py --force-fallback     # hors-ligne : mini-jeux embarqués
  python scripts/ingest.py --sources medquad frenchmedmcqa
"""

from __future__ import annotations

import argparse
import sys

from chsa_triage.data.ingest import format_inventory_table, run
from chsa_triage.data.sources import REGISTRY


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingestion des corpus médicaux CHSA.")
    parser.add_argument("--sources", nargs="*", choices=list(REGISTRY.keys()), default=None,
                        help="Sous-ensemble de sources (défaut : toutes).")
    parser.add_argument("--max-rows", type=int, default=3000,
                        help="Plafond de lignes par source (0 = illimité).")
    parser.add_argument("--force-fallback", action="store_true",
                        help="Force le mode repli (mini-jeux embarqués, hors-ligne).")
    args = parser.parse_args()

    max_rows = None if args.max_rows == 0 else args.max_rows
    result = run(sources=args.sources, max_rows=max_rows, force_fallback=args.force_fallback)

    inv = result["inventory"]
    print("\n== Inventaire d'ingestion ==")
    print(format_inventory_table(inv))
    print(f"\nInventaire écrit : {result['inventory_path']}")

    # Signale clairement les sources tombées en repli.
    fallbacks = [e for e in inv if e["mode"] == "fallback"]
    if fallbacks:
        print("\n⚠️  Sources en mode repli (fallback) :")
        for e in fallbacks:
            print(f"   - {e['source']} : {e['hub_error']}")
        print("   (la pipeline continue ; ces sources utilisent le mini-jeu embarqué)")
    else:
        print("\n✅ Toutes les sources chargées depuis le Hub.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
