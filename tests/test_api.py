import os
from fastapi.testclient import TestClient

from chsa_triage.api.app import create_app, Backend
from chsa_triage.api.fallback import fallback_triage


def _client():
    # Backend forcé en repli (dossier modèle inexistant) -> testable sans GPU.
    backend = Backend(model_dir="___no_such_model___", device="cpu")
    return TestClient(create_app(backend)), backend


def test_fallback_backend_used():
    _, b = _client()
    assert b.mode == "fallback"


def test_health():
    client, _ = _client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["mode"] == "fallback"


def test_triage_vital_and_nonurgent():
    client, _ = _client()
    r = client.post("/triage", json={"text": "Homme 55 ans, douleur thoracique et sueurs", "language": "fr"})
    assert r.status_code == 200
    body = r.json()
    assert body["priority"] == "urgence_vitale"
    assert body["mode"] == "fallback"
    assert body["request_id"]

    r2 = client.post("/triage", json={"text": "petite éraflure superficielle au doigt", "language": "fr"})
    assert r2.json()["priority"] == "non_urgent"


def test_fallback_triage_direct():
    assert "urgence_vitale" in fallback_triage("severe bleeding", "en")
    assert "non_urgent" in fallback_triage("mild sore throat", "en")
