"""Comparaison de plusieurs modèles sur le même harnais d'évaluation.

Fonctions pures (testables sans GPU) qui agrègent les rapports d'évaluation en un tableau
comparatif SFT vs DPO (ou plus).
"""

from __future__ import annotations

from ..data.metadata import PRIORITY_LEVELS

COMPARE_METRICS = [
    ("priority_accuracy", "Exactitude priorité"),
    ("safety_flagged", "Drapeaux sécurité"),
    ("unparsed", "Réponses non parsées"),
    ("avg_red_flag_recall", "Rappel signes d'alerte"),
]


def compare(named_reports: dict[str, dict]) -> dict:
    """Construit une structure comparative à partir de {nom_modèle: rapport_éval}."""
    models = list(named_reports)
    metrics = {}
    for key, _label in COMPARE_METRICS:
        metrics[key] = {name: rep.get(key) for name, rep in named_reports.items()}
    per_level = {}
    for lvl in PRIORITY_LEVELS:
        per_level[lvl] = {
            name: (rep.get("per_level_accuracy") or {}).get(lvl)
            for name, rep in named_reports.items()
        }
    return {"models": models, "metrics": metrics, "per_level": per_level}


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def format_comparison_table(comparison: dict) -> str:
    models = comparison["models"]
    width = 26
    header = "Métrique".ljust(width) + "".join(m.ljust(14) for m in models)
    lines = [header, "-" * len(header)]
    for key, label in COMPARE_METRICS:
        row = label.ljust(width) + "".join(_fmt(comparison["metrics"][key][m]).ljust(14) for m in models)
        lines.append(row)
    lines.append("-" * len(header))
    lines.append("Exactitude par niveau :")
    for lvl in PRIORITY_LEVELS:
        row = ("  " + lvl).ljust(width) + "".join(_fmt(comparison["per_level"][lvl][m]).ljust(14) for m in models)
        lines.append(row)
    return "\n".join(lines)
