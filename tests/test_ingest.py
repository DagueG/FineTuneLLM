import json
from pathlib import Path

from chsa_triage.data.ingest import ingest_source, format_inventory_table
from chsa_triage.data.loaders import load_source
from chsa_triage.data.sources import REGISTRY
from chsa_triage.audit import AuditLogger, verify_chain


def test_all_sources_have_fallback():
    from chsa_triage.config import PROJECT_ROOT
    for key, src in REGISTRY.items():
        p = PROJECT_ROOT / "data" / "fallback" / src.fallback_file
        assert p.exists(), f"fallback manquant pour {key}"


def test_normalizers_on_fallback():
    # Chaque source doit produire au moins un exemple exploitable depuis son fallback.
    for key, src in REGISTRY.items():
        rows, mode, err = load_source(src, force_fallback=True)
        assert mode == "fallback"
        usable = [src.normalize(r, i, src) for i, r in enumerate(rows)]
        usable = [e for e in usable if e is not None and e.is_usable()]
        assert usable, f"aucun exemple exploitable pour {key}"
        # cohérence du type
        for e in usable:
            if src.kind == "preference":
                assert e.chosen and e.rejected and e.instruction
            else:
                assert e.instruction and e.output


def test_ingest_source_writes_jsonl_and_audits(tmp_path):
    audit = AuditLogger(tmp_path / "audit.jsonl")
    out_dir = tmp_path / "raw"
    entry = ingest_source(REGISTRY["medquad"], out_dir, audit, max_rows=None, force_fallback=True)

    assert entry["mode"] == "fallback"
    assert entry["usable_examples"] >= 1
    jsonl = out_dir / "medquad.jsonl"
    assert jsonl.exists()
    lines = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == entry["usable_examples"]
    assert lines[0]["source"] == "medquad" and lines[0]["language"] == "en"

    ok, n = verify_chain(tmp_path / "audit.jsonl")
    assert ok and n >= 1


def test_preference_normalizer_rejects_incomplete():
    src = REGISTRY["ultramedical_pref"]
    assert src.normalize({"prompt": "q", "chosen": "good"}, 0, src) is None  # rejected manquant


def test_format_inventory_table():
    inv = [{"source": "medquad", "language": "en", "kind": "qa", "mode": "fallback",
            "raw_rows": 4, "usable_examples": 4}]
    table = format_inventory_table(inv)
    assert "medquad" in table and "fallback" in table
