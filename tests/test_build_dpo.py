import json

from chsa_triage.data.templates import to_dpo_record, SYSTEM_PROMPTS
from chsa_triage.data.build_dpo import build, format_stats


def test_dpo_record_shape():
    ex = {"instruction": "Douleur thoracique ?", "chosen": "Évaluer en urgence.",
          "rejected": "Ignorer.", "language": "fr", "source": "ultramedical_pref", "kind": "preference"}
    rec = to_dpo_record(ex)
    assert [m["role"] for m in rec["prompt"]] == ["system", "user"]
    assert rec["prompt"][0]["content"] == SYSTEM_PROMPTS["fr"]
    assert rec["chosen"][0]["content"] == "Évaluer en urgence."
    assert rec["rejected"][0]["content"] == "Ignorer."


def test_dpo_record_rejects_incomplete():
    assert to_dpo_record({"instruction": "q", "chosen": "a", "language": "en"}) is None  # rejected manquant


def test_build_consistency_and_dedup(tmp_path, monkeypatch):
    import chsa_triage.data.build_dpo as m
    raw = tmp_path / "raw"; raw.mkdir()
    proc = tmp_path / "processed"; proc.mkdir()

    rows = [
        {"instruction": "Q1 chest pain?", "chosen": "See a clinician now.", "rejected": "Do nothing.",
         "language": "en", "source": "ultramedical_pref", "kind": "preference"},
        # doublon exact du précédent
        {"instruction": "Q1 chest pain?", "chosen": "See a clinician now.", "rejected": "Do nothing.",
         "language": "en", "source": "ultramedical_pref", "kind": "preference"},
        # chosen == rejected -> rejeté
        {"instruction": "Q2?", "chosen": "same", "rejected": "same",
         "language": "en", "source": "ultramedical_pref", "kind": "preference"},
        # valide FR
        {"instruction": "Q3 fièvre ?", "chosen": "Surveiller, signes d'alerte.", "rejected": "Antibiotiques immédiats.",
         "language": "fr", "source": "ultramedical_pref", "kind": "preference"},
    ]
    (raw / "ultramedical_pref.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    class Cfg:
        class data:
            raw_dir = str(raw); processed_dir = str(proc); dpo_target_pairs = 100
        class model: seed = 42
        class audit: log_path = str(tmp_path / "audit.jsonl")
    monkeypatch.setattr(m, "load_config", lambda *a, **k: Cfg)
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)

    stats = build(sources=["ultramedical_pref"], target=100)
    assert stats["total"] == 2                 # Q1 (dédupliqué) + Q3
    assert stats["dropped_duplicates"] == 1
    assert stats["dropped_identical"] == 1
    assert stats["final_by_language"] == {"en": 1, "fr": 1}
    assert "Total DPO" in format_stats(stats)

    lines = [json.loads(l) for l in (proc / "dpo.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    for r in lines:
        assert set(r.keys()) >= {"prompt", "chosen", "rejected", "meta"}
        assert r["chosen"][0]["content"] != r["rejected"][0]["content"]
