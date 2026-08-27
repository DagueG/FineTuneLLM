"""Évaluation clinique + sécurité de l'agent de triage.

Exemples :
  python scripts/evaluate.py --mock                       # teste la pipeline sans GPU
  python scripts/evaluate.py --model-dir models/sft-lora  # vrai modèle (sur GPU)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chsa_triage.audit import AuditLogger
from chsa_triage.config import PROJECT_ROOT, load_config
from chsa_triage.eval.harness import evaluate, load_scenarios


def _mock_generator(messages: list[dict], **kwargs) -> str:
    """Générateur factice : répond en fonction de mots-clés du scénario (test pipeline)."""
    user = messages[-1]["content"].lower()
    if any(w in user for w in ["chest pain", "douleur thoracique", "unrespons", "bleeding",
                                "anaphyl", "avc", "stroke", "saturation of 85", "guêpe"]):
        return "This looks like an emergency. Call an ambulance immediately.\nPRIORITY: urgence_vitale"
    if any(w in user for w in ["fever", "fièvre", "asthma", "asthme", "fracture", "abdominal", "migraine"]):
        return "This needs prompt medical evaluation.\nPRIORITY: urgent"
    return "Likely minor. Rest and monitor, with safety-net advice.\nPRIORITY: non_urgent"


def main() -> int:
    p = argparse.ArgumentParser(description="Évaluation clinique + sécurité CHSA.")
    p.add_argument("--model-dir", default="models/sft-lora")
    p.add_argument("--mock", action="store_true", help="Utilise un générateur factice (test sans GPU).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = load_config()
    audit = AuditLogger(PROJECT_ROOT / cfg.audit.log_path)
    scenarios = load_scenarios(PROJECT_ROOT / "data" / "eval" / "clinical_eval.jsonl", args.limit)

    if args.mock:
        generator = _mock_generator
        model_id = "mock"
    else:
        from chsa_triage.infer.generate import TriageModel
        tm = TriageModel.load(args.model_dir, device=args.device)
        generator = tm.generate
        model_id = args.model_dir

    audit.log("eval.start", {"model": model_id, "n_scenarios": len(scenarios)})
    report = evaluate(generator, scenarios)

    # Résumé (sans les détails complets à l'écran).
    summary = {k: v for k, v in report.items() if k != "details"}
    print("\n== Évaluation clinique ==")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out = PROJECT_ROOT / cfg.data.processed_dir / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"model": model_id, **report}, f, ensure_ascii=False, indent=2)
    audit.log("eval.done", {**summary, "model": model_id})
    print(f"\nRapport complet (avec réponses) : {out}")

    if report["safety_flagged"]:
        print(f"\n⚠️  {report['safety_flagged']} réponse(s) avec drapeau de sécurité "
              f"— à inspecter dans le rapport.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
