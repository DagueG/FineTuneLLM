"""Fine-tuning supervisé (SFT) LoRA de Qwen3-1.7B-Base avec TRL.

Conception (voir ADR-0007) :
- Format conversationnel `messages` (nos JSONL) géré nativement par TRL `SFTTrainer`.
- LoRA (bf16) par défaut ; QLoRA 4-bit optionnel pour petite VRAM.
- Profil auto-détecté depuis la VRAM ; override CLI possible.
- Checkpoints + reprise, seed, logging optionnel (W&B avec repli hors-ligne).
- Modes `--smoke` (minuscule, CPU-friendly) et `--dry-run` (tout préparer sans entraîner)
  pour valider la pipeline avant un vrai run.

Les imports lourds (torch, transformers, trl, peft) sont PARESSEUX : ce module s'importe
sans eux, ce qui permet de tester la logique pure sans GPU.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from ..audit import AuditLogger
from ..config import PROJECT_ROOT, load_config
from .hf_utils import PROFILES, TrainProfile, bridge_hf_token, detect_vram_gb, select_profile

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_jsonl(path: Path, limit: Optional[int] = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_profile(profile_name: Optional[str], smoke: bool) -> TrainProfile:
    """Détermine le profil effectif (smoke > override CLI > auto-détection VRAM)."""
    if smoke:
        return PROFILES["smoke"]
    if profile_name:
        return PROFILES[profile_name]
    return select_profile(detect_vram_gb())


def _construct(cls, desired: dict[str, Any]):
    """Construit `cls(**desired)` en ne gardant que les arguments RÉELLEMENT supportés par
    la version installée, et en gérant les renommages connus. Robuste aux dérives d'API.

    Retourne (instance, liste_des_arguments_ignorés).
    """
    import inspect
    params = set(inspect.signature(cls).parameters)
    aliases = {
        "max_length": ["max_length", "max_seq_length"],
        "eval_strategy": ["eval_strategy", "evaluation_strategy"],
    }
    kept: dict[str, Any] = {}
    dropped: list[str] = []
    for key, val in desired.items():
        for name in aliases.get(key, [key]):
            if name in params:
                kept[name] = val
                break
        else:
            dropped.append(key)
    return cls(**kept), dropped


def build_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Construit la LoraConfig PEFT (import paresseux)."""
    from peft import LoraConfig
    return LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
    )


def supports_assistant_mask(tokenizer) -> bool:
    """Vrai si le chat template contient le VRAI bloc de masquage Jinja `{% generation %}`
    (et pas seulement le mot « generation », présent dans `add_generation_prompt`).
    Sinon, on entraîne sur toute la séquence."""
    import re
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return False
    return re.search(r"\{%-?\s*generation\s*-?%\}", template) is not None


def build_sft_config(
    output_dir: str,
    profile: TrainProfile,
    epochs: float,
    learning_rate: float,
    eos_token: str,
    report_to: str,
    seed: int,
    assistant_only_loss: bool = False,
    logging_steps: int = 10,
    save_steps: int = 200,
    eval_steps: int = 200,
    max_steps: int = -1,
):
    """Construit la SFTConfig TRL (import paresseux), en ne passant que les arguments
    supportés par la version installée. Retourne (config, arguments_ignorés)."""
    from trl import SFTConfig
    desired = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=profile.per_device_batch_size,
        per_device_eval_batch_size=profile.per_device_batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation_steps,
        gradient_checkpointing=profile.gradient_checkpointing,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=profile.max_length,
        bf16=profile.bf16,
        packing=False,
        assistant_only_loss=assistant_only_loss,  # activé seulement si le template le supporte
        eos_token=eos_token,           # aligne l'EOS sur le chat template (Qwen : <|im_end|>)
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        report_to=report_to,
        seed=seed,
    )
    return _construct(SFTConfig, desired)


def run_sft(
    profile_name: Optional[str] = None,
    smoke: bool = False,
    dry_run: bool = False,
    epochs: float = 3.0,
    learning_rate: float = 2e-4,
    report_to: str = "none",
    resume: bool = False,
    config_path: Optional[str] = None,
) -> dict:
    """Lance (ou prépare) le SFT LoRA. Retourne un dict de métadonnées du run."""
    cfg = load_config(config_path)
    audit = AuditLogger(PROJECT_ROOT / cfg.audit.log_path)
    seed = cfg.model.seed
    base_model = cfg.model.base_model_id
    proc = PROJECT_ROOT / cfg.data.processed_dir
    out_dir = PROJECT_ROOT / "models" / "sft-lora"
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = resolve_profile(profile_name, smoke)
    has_token = bridge_hf_token()
    vram = detect_vram_gb()

    train_path = proc / "splits" / "sft_train.jsonl"
    val_path = proc / "splits" / "sft_val.jsonl"
    limit = 64 if smoke else None
    train_rows = load_jsonl(train_path, limit=limit)
    val_rows = load_jsonl(val_path, limit=16 if smoke else None)

    meta = {
        "base_model": base_model,
        "profile": profile.name,
        "profile_params": asdict(profile),
        "vram_gb": vram,
        "hf_token_bridged": has_token,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "output_dir": str(out_dir),
        "dry_run": dry_run,
        "smoke": smoke,
    }
    audit.log("sft.train.start", meta)

    if dry_run:
        meta["status"] = "dry_run_ok"
        audit.log("sft.train.dryrun", {"status": "ok"})
        return meta

    # --- À partir d'ici : imports lourds et exécution réelle ---
    from datasets import Dataset
    import torch  # noqa: F401  (présence requise)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # Qwen : EOS attendu par le chat template.
    eos_token = "<|im_end|>" if "<|im_end|>" in tokenizer.get_vocab() else (tokenizer.eos_token or "</s>")
    # N'entraîner que sur les tokens de l'assistant seulement si le template le permet.
    assistant_only = supports_assistant_mask(tokenizer)
    print(f"[info] assistant_only_loss = {assistant_only} "
          f"({'template compatible' if assistant_only else 'template sans masque -> loss sur toute la séquence'})")

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(val_rows) if val_rows else None

    model_kwargs: dict[str, Any] = {}
    if profile.load_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    max_steps = 4 if smoke else -1
    sft_config, dropped_args = build_sft_config(
        output_dir=str(out_dir), profile=profile, epochs=epochs,
        learning_rate=learning_rate, eos_token=eos_token, report_to=report_to,
        seed=seed, max_steps=max_steps, assistant_only_loss=assistant_only,
    )
    if dropped_args:
        print(f"[info] arguments SFTConfig ignorés (non supportés par ta version TRL) : {dropped_args}")
        audit.log("sft.train.config_dropped", {"dropped": dropped_args})

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=build_lora_config(),
    )
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    meta["status"] = "trained"
    audit.log("sft.train.done", {"output_dir": str(out_dir), "profile": profile.name})
    return meta
