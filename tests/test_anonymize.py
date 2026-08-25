from chsa_triage.data.anonymize import Anonymizer, _mask_value


def test_regex_fallback_email_phone(monkeypatch):
    anon = Anonymizer(strategy="replace")
    # Force le repli regex (pas de Presidio) pour un test déterministe et léger.
    anon._backend = "regex"
    text = "Contact: jean.dupont@chu.fr, tel 06 12 34 56 78, le 05/03/2024."
    clean, ents = anon.anonymize(text, language="fr")
    assert "jean.dupont@chu.fr" not in clean
    assert "06 12 34 56 78" not in clean
    assert "05/03/2024" not in clean
    assert "EMAIL_ADDRESS" in ents and "PHONE_NUMBER" in ents and "DATE_TIME" in ents
    # QC : plus aucune entité résiduelle
    assert anon.residual_entities(clean, language="fr") == []


def test_strategies():
    assert _mask_value("PERSON", "Marie", "replace") == "<PERSON>"
    assert _mask_value("PERSON", "Marie", "mask") == "*****"
    assert _mask_value("PERSON", "Marie", "redact") == ""


def test_empty_text():
    anon = Anonymizer()
    anon._backend = "regex"
    assert anon.anonymize("", language="en") == ("", [])


def test_training_entities_exclude_noisy_ner():
    from chsa_triage.data.anonymize import TRAINING_ENTITIES
    # Le mode entraînement ne masque QUE les PII structurées (préserve le contenu médical).
    for noisy in ("PERSON", "LOCATION", "DATE_TIME"):
        assert noisy not in TRAINING_ENTITIES
    for structured in ("EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE"):
        assert structured in TRAINING_ENTITIES
