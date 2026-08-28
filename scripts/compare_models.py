"""Compare plusieurs modèles (ex. SFT vs DPO) sur le jeu d'éval clinique.

Exemples :
  python scripts/compare_models.py --mock
  python scripts/compare_models.py --models models/sft-lora models/dpo-lora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chsa_triage.audit import AuditLogger
from chsa_triage.config import PROJECT_ROOT, load_config
from chsa_triage.eval.compare import compare, format_comparison_table
from chsa_triage.eval.harness import evaluate, load_scenarios


def _mock_generator(messages, **kwargs):
    """Générateur factice pour tester la pipeline sans GPU."""
    user = messages[-1]["content"].lower()
    if any(w in user for w in ["chest pain", "douleur thoracique", "unrespons", "bleeding",
                                "anaphyl", "avc", "stroke", "saturation of 85", "guêpe"]):
        return "This looks like an emergency. Call an ambulance immediately.\nPRIORITY: urgence_vitale"
    if any(w in user for w in ["fever", "fièvre", "asthma", "asthme", "fracture", "abdominal", "migraine"]):
        return "This needs prompt medical evaluation.\nPRIORITY: urgent"
    return "Likely minor. Rest and monitor, with safety-net advice.\nPRIORITY: non_urgent"


def main() -> int:
    p = argparse.ArgumentParser(description="Comparaison de modèles CHSA (SFT vs DPO).")
    p.add_argument("--models", nargs="+", default=["models/sft-lora", "models/dpo-lora"])
    p.add_argument("--mock", action="store_true", help="Générateur factice (test sans GPU).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = load_config()
    audit = AuditLogger(PROJECT_ROOT / cfg.audit.log_path)
    scenarios = load_scenarios(PROJECT_ROOT / "data" / "eval" / "clinical_eval.jsonl", args.limit)

    named_reports = {}
    for model_dir in args.models:
        label = Path(model_dir).name
        if args.mock:
            generator = _mock_generator
        else:
            from chsa_triage.infer.generate import TriageModel
            tm = TriageModel.load(model_dir, device=args.device)
            generator = tm.generate
        print(f"[éval] {label}…")
        named_reports[label] = {k: v for k, v in evaluate(generator, scenarios).items()
                                if k != "details"}

    comparison = compare(named_reports)
    print("\n== Comparaison des modèles ==")
    print(format_comparison_table(comparison))

    out = PROJECT_ROOT / cfg.data.processed_dir / "comparison_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"reports": named_reports, "comparison": comparison}, f, ensure_ascii=False, indent=2)
    audit.log("compare.done", {"models": comparison["models"]})
    print(f"\nRapport : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
