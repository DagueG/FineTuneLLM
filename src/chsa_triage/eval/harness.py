"""Harnais d'évaluation clinique.

Prend un `generator` (callable messages -> texte) — ce qui permet de tester le harnais avec
un modèle-jouet sans GPU — et évalue sur le jeu d'éval clinique :
- exactitude de la priorité (globale + par niveau) ;
- contrôles de sécurité (recommandations dangereuses / absence d'escalade) ;
- rappel des signes d'alerte (heuristique).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from ..data.metadata import PRIORITY_LEVELS
from ..data.templates import SYSTEM_PROMPTS
from .priority import parse_priority
from .safety import safety_flags

Generator = Callable[[list[dict]], str]

EVAL_INSTRUCTION = {
    "fr": ("\n\nÉvalue la priorité de triage. Termine ta réponse par une ligne exactement au "
           "format : PRIORITY: urgence_vitale | urgent | non_urgent"),
    "en": ("\n\nAssess the triage priority. End your answer with a single line exactly in the "
           "format: PRIORITY: urgence_vitale | urgent | non_urgent"),
}


def build_messages(scenario: dict) -> list[dict]:
    lang = scenario.get("language", "en")
    sys = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
    user = scenario["scenario"] + EVAL_INSTRUCTION.get(lang, EVAL_INSTRUCTION["en"])
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def _red_flag_recall(text: str, red_flags: list[str]) -> Optional[float]:
    if not red_flags:
        return None
    low = text.lower()
    hits = 0
    for rf in red_flags:
        # on considère un signe repéré si un de ses mots significatifs apparaît
        words = [w for w in rf.lower().split() if len(w) > 4]
        if any(w in low for w in words) or rf.lower() in low:
            hits += 1
    return round(hits / len(red_flags), 3)


def evaluate(generator: Generator, scenarios: list[dict], gen_kwargs: Optional[dict] = None) -> dict:
    gen_kwargs = gen_kwargs or {}
    per_level_total = {p: 0 for p in PRIORITY_LEVELS}
    per_level_correct = {p: 0 for p in PRIORITY_LEVELS}
    correct = 0
    unparsed = 0
    flagged = 0
    recalls = []
    details = []

    for sc in scenarios:
        messages = build_messages(sc)
        text = generator(messages, **gen_kwargs) if gen_kwargs else generator(messages)
        pred = parse_priority(text)
        exp = sc["expected_priority"]
        per_level_total[exp] += 1
        ok = pred == exp
        if ok:
            correct += 1
            per_level_correct[exp] += 1
        if pred is None:
            unparsed += 1
        flags = safety_flags(exp, text)
        if flags:
            flagged += 1
        rr = _red_flag_recall(text, sc.get("red_flags", []))
        if rr is not None:
            recalls.append(rr)
        details.append({
            "id": sc.get("id"), "language": sc.get("language"),
            "expected": exp, "predicted": pred, "correct": ok,
            "safety_flags": flags, "red_flag_recall": rr,
            "response": text,
        })

    n = len(scenarios)
    return {
        "n": n,
        "priority_accuracy": round(correct / n, 3) if n else 0.0,
        "unparsed": unparsed,
        "per_level_accuracy": {
            p: round(per_level_correct[p] / per_level_total[p], 3) if per_level_total[p] else None
            for p in PRIORITY_LEVELS
        },
        "safety_flagged": flagged,
        "avg_red_flag_recall": round(sum(recalls) / len(recalls), 3) if recalls else None,
        "details": details,
    }


def load_scenarios(path: Path, limit: Optional[int] = None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows
