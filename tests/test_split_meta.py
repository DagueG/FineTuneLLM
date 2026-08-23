import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from chsa_triage.data.split import stratified_split, leakage_report
from chsa_triage.data.metadata import TriageInput, TriageAssessment, PRIORITY_LEVELS


def _mk(source, lang, i):
    return {"messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": f"question {source} {lang} {i}"},
        {"role": "assistant", "content": "a"},
    ], "meta": {"source": source, "language": lang, "kind": "qa"}}


def test_split_ratios_and_no_leakage():
    recs = [_mk("medquad", "en", i) for i in range(100)] + [_mk("frenchmedmcqa", "fr", i) for i in range(100)]
    sp = stratified_split(recs, ratios=(0.8, 0.1, 0.1), seed=42)
    assert len(sp["train"]) == 160 and len(sp["val"]) == 20 and len(sp["test"]) == 20
    leak = leakage_report(sp)
    assert all(v == 0 for v in leak.values())
    # stratification : chaque split contient les 2 langues
    for name in ("train", "val", "test"):
        langs = {r["meta"]["language"] for r in sp[name]}
        assert langs == {"en", "fr"}


def test_no_leakage_with_duplicate_questions():
    # 20 questions uniques, chacune dupliquée 3x (comme après anonymisation).
    recs = []
    for i in range(20):
        for _ in range(3):
            recs.append(_mk("medquad", "en", i))  # même contenu -> même question
    sp = stratified_split(recs, ratios=(0.8, 0.1, 0.1), seed=1)
    leak = leakage_report(sp)
    assert all(v == 0 for v in leak.values()), leak
    # aucune donnée perdue
    assert len(sp["train"]) + len(sp["val"]) + len(sp["test"]) == 60


def test_split_is_deterministic():
    recs = [_mk("medquad", "en", i) for i in range(50)]
    a = stratified_split(recs, seed=7)
    b = stratified_split(recs, seed=7)
    assert [r["messages"][1]["content"] for r in a["train"]] == \
           [r["messages"][1]["content"] for r in b["train"]]


def test_metadata_schema_valid():
    ti = TriageInput(chief_complaint="douleur thoracique", symptoms=["sueurs"],
                     vitals={"heart_rate_bpm": 110, "spo2_percent": 94.0}, language="fr")
    assert ti.vitals.heart_rate_bpm == 110
    ta = TriageAssessment(priority="urgence_vitale", confidence=0.9, red_flags=["dyspnée"])
    assert ta.priority in PRIORITY_LEVELS


def test_metadata_rejects_bad_values():
    with pytest.raises(ValidationError):
        TriageAssessment(priority="tres_urgent", confidence=0.5)  # priorité invalide
    with pytest.raises(ValidationError):
        TriageAssessment(priority="urgent", confidence=1.5)       # confiance hors [0,1]


def test_clinical_eval_file_is_wellformed():
    path = Path(__file__).resolve().parents[1] / "data" / "eval" / "clinical_eval.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) >= 12
    for r in rows:
        assert r["expected_priority"] in PRIORITY_LEVELS
        assert r["language"] in ("fr", "en")
        assert r["scenario"] and isinstance(r["red_flags"], list)
