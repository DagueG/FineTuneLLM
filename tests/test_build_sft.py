import json

from chsa_triage.data.templates import to_sft_messages, SYSTEM_PROMPTS
from chsa_triage.data.build_sft import build, format_stats


def test_templating_qa_and_mcqa():
    qa = {"instruction": "What is hypertension?", "output": "High blood pressure.",
          "language": "en", "source": "medquad", "kind": "qa", "input": ""}
    rec = to_sft_messages(qa)
    assert rec["messages"][0]["role"] == "system"
    assert rec["messages"][0]["content"] == SYSTEM_PROMPTS["en"]
    assert rec["messages"][1]["content"] == "What is hypertension?"
    assert rec["messages"][2]["content"] == "High blood pressure."

    mcqa = {"instruction": "Signe de choc ?", "input": "A. X\nB. Y", "output": "Réponse : A",
            "language": "fr", "source": "frenchmedmcqa", "kind": "mcqa"}
    rec2 = to_sft_messages(mcqa)
    assert "A. X" in rec2["messages"][1]["content"]  # options intégrées au message user
    assert rec2["messages"][0]["content"] == SYSTEM_PROMPTS["fr"]


def test_templating_rejects_empty():
    assert to_sft_messages({"instruction": "", "output": "x", "language": "en"}) is None
    assert to_sft_messages({"instruction": "q", "output": "", "language": "en"}) is None


def test_build_dedup_quality_and_balance(tmp_path, monkeypatch):
    # Prépare un faux data/raw et une config pointant dessus.
    import chsa_triage.data.build_sft as m
    raw = tmp_path / "raw"; raw.mkdir()
    proc = tmp_path / "processed"; proc.mkdir()

    # medquad : 1 valide + 1 doublon + 1 dégénéré (q==a)
    (raw / "medquad.jsonl").write_text("\n".join(json.dumps(x) for x in [
        {"instruction": "What is fever?", "output": "Elevated body temperature.", "language": "en", "source": "medquad", "kind": "qa", "input": ""},
        {"instruction": "what is FEVER?", "output": "Something else.", "language": "en", "source": "medquad", "kind": "qa", "input": ""},
        {"instruction": "same", "output": "same", "language": "en", "source": "medquad", "kind": "qa", "input": ""},
    ]), encoding="utf-8")
    # frenchmedmcqa : 2 valides
    (raw / "frenchmedmcqa.jsonl").write_text("\n".join(json.dumps(x) for x in [
        {"instruction": "Question A ?", "input": "A. x\nB. y", "output": "Réponse : A", "language": "fr", "source": "frenchmedmcqa", "kind": "mcqa"},
        {"instruction": "Question B ?", "input": "A. x\nB. y", "output": "Réponse : B", "language": "fr", "source": "frenchmedmcqa", "kind": "mcqa"},
    ]), encoding="utf-8")

    class Cfg:
        class data: raw_dir = str(raw); processed_dir = str(proc); sft_target_pairs = 100
        class model: seed = 42
        class audit: log_path = str(tmp_path / "audit.jsonl")
    monkeypatch.setattr(m, "load_config", lambda *a, **k: Cfg)
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)

    stats = build(sources=["medquad", "frenchmedmcqa"], target=100)
    # 1 (fever) dédupliqué du doublon, 1 dégénéré rejeté, 2 FR => total 3
    assert stats["total"] == 3
    assert stats["dropped_duplicates"] == 1
    assert stats["dropped_quality"] == 1
    assert stats["final_by_language"] == {"en": 1, "fr": 2}
    # format lisible
    assert "Total SFT" in format_stats(stats)
    # fichier écrit et valide
    lines = [json.loads(l) for l in (proc / "sft.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3
    assert all(len(r["messages"]) == 3 for r in lines)


def test_mcqa_share_is_capped(tmp_path, monkeypatch):
    import chsa_triage.data.build_sft as m
    raw = tmp_path / "raw"; raw.mkdir()
    proc = tmp_path / "processed"; proc.mkdir()

    # 100 QCM (frenchmedmcqa) + 100 rédigés (medquad)
    mcqa = [{"instruction": f"Question {i} ?", "input": "A. x\nB. y", "output": "Réponse : A",
             "language": "fr", "source": "frenchmedmcqa", "kind": "mcqa"} for i in range(100)]
    qa = [{"instruction": f"What is topic {i}?", "output": f"Explanation number {i} here.",
           "language": "en", "source": "medquad", "kind": "qa", "input": ""} for i in range(100)]
    (raw / "frenchmedmcqa.jsonl").write_text("\n".join(json.dumps(x) for x in mcqa), encoding="utf-8")
    (raw / "medquad.jsonl").write_text("\n".join(json.dumps(x) for x in qa), encoding="utf-8")

    class Cfg:
        class data: raw_dir = str(raw); processed_dir = str(proc); sft_target_pairs = 100
        class model: seed = 42
        class audit: log_path = str(tmp_path / "audit.jsonl")
    monkeypatch.setattr(m, "load_config", lambda *a, **k: Cfg)
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)

    stats = build(sources=["medquad", "frenchmedmcqa"], target=100, max_share_mcqa=0.3)
    assert stats["total"] == 100
    # au plus 30 % de QCM
    assert stats["final_by_kind"].get("mcqa", 0) <= 30
    assert stats["final_by_kind"].get("qa", 0) >= 70
