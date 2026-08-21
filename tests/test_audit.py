import json

from chsa_triage.audit import AuditLogger, verify_chain


def test_log_and_verify_ok(tmp_path):
    log = tmp_path / "audit.jsonl"
    logger = AuditLogger(log)
    logger.log("data.ingested", {"source": "MedQuAD", "rows": 47})
    logger.log("data.anonymized", {"tool": "presidio", "entities_masked": 12})
    logger.log("triage.evaluated", {"priority": "modérée"}, actor="agent")

    ok, n = verify_chain(log)
    assert ok is True
    assert n == 3


def test_empty_or_missing_is_valid(tmp_path):
    ok, n = verify_chain(tmp_path / "nope.jsonl")
    assert ok is True and n == 0


def test_tampering_is_detected(tmp_path):
    log = tmp_path / "audit.jsonl"
    logger = AuditLogger(log)
    logger.log("a", {"x": 1})
    logger.log("b", {"x": 2})
    logger.log("c", {"x": 3})

    # Falsification : on modifie le payload de la 2e ligne sans recalculer les hash.
    lines = log.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[1])
    rec["payload"]["x"] = 999
    lines[1] = json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, idx = verify_chain(log)
    assert ok is False
    assert idx == 1  # la première ligne incohérente est bien la 2e (index 1)
