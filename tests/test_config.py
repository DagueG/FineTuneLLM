from chsa_triage.config import AppConfig, load_config


def test_defaults():
    cfg = AppConfig()
    assert cfg.app_name == "chsa-triage"
    assert cfg.model.base_model_id == "Qwen/Qwen3-1.7B-Base"
    assert cfg.data.sft_target_pairs == 5000
    assert cfg.data.languages == ["fr", "en"]


def test_load_from_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "app_name: test-app\nenvironment: pilot\nmodel:\n  seed: 7\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.app_name == "test-app"
    assert cfg.environment == "pilot"
    assert cfg.model.seed == 7
    # les champs non fournis gardent leurs valeurs par défaut
    assert cfg.model.base_model_id == "Qwen/Qwen3-1.7B-Base"


def test_missing_file_falls_back_to_defaults(tmp_path):
    cfg = load_config(tmp_path / "does_not_exist.yaml")
    assert cfg.app_name == "chsa-triage"
