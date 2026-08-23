import importlib.util
import json
from pathlib import Path

from chsa_triage.data.anonymize import Anonymizer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "anonymize_dataset.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("anon_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pipeline_regex_backend(tmp_path, monkeypatch):
    m = _load_script_module()
    proc = tmp_path / "processed"; proc.mkdir()

    # SFT avec un email dans le message user et un prompt système à préserver.
    sft = [{
        "messages": [
            {"role": "system", "content": "Assistant du Centre Hospitalier Saint-Aurélien."},
            {"role": "user", "content": "Contacter j.dupont@chu.fr au sujet des symptômes."},
            {"role": "assistant", "content": "Repos et hydratation."},
        ],
        "meta": {"language": "fr", "source": "medquad", "kind": "qa"},
    }]
    (proc / "sft.jsonl").write_text("\n".join(json.dumps(x) for x in sft), encoding="utf-8")

    # Force le backend regex (déterministe, sans modèle lourd).
    def _regex_anon(*a, **k):
        an = Anonymizer(*a, **k)
        an._backend = "regex"
        return an
    monkeypatch.setattr(m, "Anonymizer", _regex_anon)

    class Cfg:
        class data: processed_dir = str(proc)
        class audit: log_path = str(tmp_path / "audit.jsonl")
    monkeypatch.setattr(m, "load_config", lambda *a, **k: Cfg)
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)

    res = m.run(strategy="replace", qc_sample=10)
    assert res["backend"] == "regex"
    rep = [r for r in res["reports"] if r["file"] == "sft.jsonl"][0]
    assert rep["qc_passed"] is True
    assert rep["qc_residual_structured"] == 0

    out = [json.loads(l) for l in (proc / "sft_anonymized.jsonl").read_text(encoding="utf-8").splitlines()]
    msgs = out[0]["messages"]
    # système préservé, email masqué dans le message user
    assert msgs[0]["content"] == "Assistant du Centre Hospitalier Saint-Aurélien."
    assert "j.dupont@chu.fr" not in msgs[1]["content"]
    assert "<EMAIL_ADDRESS>" in msgs[1]["content"]
