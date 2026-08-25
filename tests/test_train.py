import json
import os

from chsa_triage.train.hf_utils import (
    PROFILES, select_profile, detect_vram_gb, bridge_hf_token,
)
from chsa_triage.train import sft as sftmod
from chsa_triage.train.sft import load_jsonl, resolve_profile


def test_select_profile_by_vram():
    assert select_profile(None).name == "smoke"
    assert select_profile(8).name == "low"
    assert select_profile(12).name == "mid"
    assert select_profile(16).name == "mid"
    assert select_profile(24).name == "high"
    assert select_profile(80).name == "high"


def test_resolve_profile_precedence():
    assert resolve_profile(None, smoke=True).name == "smoke"      # smoke prioritaire
    assert resolve_profile("high", smoke=False).name == "high"    # override CLI


def test_low_profile_uses_qlora():
    assert PROFILES["low"].load_4bit is True
    assert PROFILES["mid"].load_4bit is False


def test_detect_vram_no_torch_or_gpu():
    # Sans GPU (bac à sable), doit renvoyer None sans lever.
    assert detect_vram_gb() is None


def test_bridge_hf_token(monkeypatch):
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CHSA_HF_TOKEN", "hf_test_123")
    assert bridge_hf_token() is True
    assert os.environ["HF_TOKEN"] == "hf_test_123"


def test_load_jsonl_limit(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps({"i": i}) for i in range(10)), encoding="utf-8")
    assert len(load_jsonl(p)) == 10
    assert len(load_jsonl(p, limit=3)) == 3


def test_run_sft_dry_run(tmp_path, monkeypatch):
    proc = tmp_path / "processed"; (proc / "splits").mkdir(parents=True)
    def _mk(user):
        return {"messages": [{"role": "system", "content": "s"},
                             {"role": "user", "content": user},
                             {"role": "assistant", "content": "a"}],
                "meta": {"source": "medquad", "language": "en", "kind": "qa"}}
    (proc / "splits" / "sft_train.jsonl").write_text(
        "\n".join(json.dumps(_mk(f"q{i}")) for i in range(20)), encoding="utf-8")
    (proc / "splits" / "sft_val.jsonl").write_text(json.dumps(_mk("v")), encoding="utf-8")

    class Cfg:
        class data: processed_dir = str(proc); raw_dir = str(tmp_path / "raw")
        class model: seed = 42; base_model_id = "Qwen/Qwen3-1.7B-Base"
        class audit: log_path = str(tmp_path / "audit.jsonl")
    monkeypatch.setattr(sftmod, "load_config", lambda *a, **k: Cfg)
    monkeypatch.setattr(sftmod, "PROJECT_ROOT", tmp_path)

    meta = sftmod.run_sft(dry_run=True, smoke=True)
    assert meta["status"] == "dry_run_ok"
    assert meta["profile"] == "smoke"
    assert meta["train_examples"] == 20
    assert meta["val_examples"] == 1


def test_construct_filters_and_aliases():
    from dataclasses import dataclass
    from chsa_triage.train.sft import _construct

    @dataclass
    class FakeConfig:
        output_dir: str = ""
        max_seq_length: int = 0          # alias attendu de max_length
        evaluation_strategy: str = "no"  # alias attendu de eval_strategy
        learning_rate: float = 0.0
        # PAS de warmup_ratio (comme la version TRL de l'utilisateur)

    desired = {
        "output_dir": "out",
        "max_length": 2048,              # doit être renommé -> max_seq_length
        "eval_strategy": "steps",        # doit être renommé -> evaluation_strategy
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,            # doit être IGNORÉ
    }
    cfg, dropped = _construct(FakeConfig, desired)
    assert cfg.max_seq_length == 2048
    assert cfg.evaluation_strategy == "steps"
    assert cfg.learning_rate == 2e-4
    assert dropped == ["warmup_ratio"]


def test_supports_assistant_mask():
    from chsa_triage.train.sft import supports_assistant_mask

    class T:
        chat_template = None
    assert supports_assistant_mask(T()) is False

    class T2:
        # contient le mot "generation" (via add_generation_prompt) mais PAS le bloc de masquage
        chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}x{% endif %}"
    assert supports_assistant_mask(T2()) is False

    class T3:
        chat_template = "{% generation %}{{ m.content }}{% endgeneration %}"
    assert supports_assistant_mask(T3()) is True

    class T4:
        chat_template = "{%- generation -%}{{ m.content }}{%- endgeneration -%}"
    assert supports_assistant_mask(T4()) is True
