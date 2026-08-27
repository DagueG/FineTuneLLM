import json

from chsa_triage.train import dpo as dpomod


def _pair(user, chosen, rejected):
    return {"prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": user}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "meta": {"source": "ultramedical_pref", "language": "en", "kind": "preference"}}


def test_run_dpo_dry_run(tmp_path, monkeypatch):
    proc = tmp_path / "processed"; (proc / "splits").mkdir(parents=True)
    (proc / "splits" / "dpo_train.jsonl").write_text(
        "\n".join(json.dumps(_pair(f"q{i}", "good", "bad")) for i in range(10)), encoding="utf-8")
    (proc / "splits" / "dpo_val.jsonl").write_text(json.dumps(_pair("v", "g", "b")), encoding="utf-8")

    class Cfg:
        class data: processed_dir = str(proc); raw_dir = str(tmp_path / "raw")
        class model: seed = 42; base_model_id = "Qwen/Qwen3-1.7B-Base"
        class audit: log_path = str(tmp_path / "audit.jsonl")
    monkeypatch.setattr(dpomod, "load_config", lambda *a, **k: Cfg)
    monkeypatch.setattr(dpomod, "PROJECT_ROOT", tmp_path)

    meta = dpomod.run_dpo(dry_run=True, smoke=True)
    assert meta["status"] == "dry_run_ok"
    assert meta["profile"] == "smoke"
    assert meta["train_pairs"] == 10
    assert meta["val_pairs"] == 1
    # pas d'adaptateur SFT dans le tmp -> signalé
    assert meta["sft_adapter_present"] is False


def test_build_dpo_config_filters_unsupported():
    # _construct est utilisé : on vérifie qu'un faux DPOConfig ne reçoit pas d'arg inconnu.
    from dataclasses import dataclass
    from chsa_triage.train.sft import _construct

    @dataclass
    class FakeDPO:
        output_dir: str = ""
        beta: float = 0.1
        learning_rate: float = 0.0
        # pas de warmup_ratio, pas de max_prompt_length

    cfg, dropped = _construct(FakeDPO, {"output_dir": "o", "beta": 0.2, "learning_rate": 5e-6,
                                        "warmup_ratio": 0.03, "max_prompt_length": 512})
    assert cfg.beta == 0.2
    assert set(dropped) == {"warmup_ratio", "max_prompt_length"}
