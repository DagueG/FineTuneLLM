"""Alignement par préférences (DPO) au-dessus du modèle SFT.

Principe (voir ADR-0009) :
- On part du modèle **SFT** : on charge le modèle de base + l'adaptateur SFT et on le
  **fusionne** (`merge_and_unload`) pour obtenir le modèle de politique de départ.
- On entraîne un **nouvel adaptateur LoRA** par DPO sur les paires (chosen/rejected). Le
  modèle de référence est le même modèle avec l'adaptateur désactivé (pas de second modèle
  en mémoire → tient sur 12 Go pour un 1.7B).
- Mêmes garde-fous que le SFT : profils VRAM, construction de config résiliente aux versions
  (`_construct`), modes `--dry-run` / `--smoke`, reprise, pont token HF.

Imports lourds paresseux : la logique pure reste testable sans GPU.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ..audit import AuditLogger
from ..config import PROJECT_ROOT, load_config
from .hf_utils import PROFILES, bridge_hf_token, detect_vram_gb
from .sft import _construct, build_lora_config, load_jsonl, resolve_profile


def build_dpo_config(
    output_dir: str,
    profile,
    epochs: float,
    learning_rate: float,
    beta: float,
    report_to: str,
    seed: int,
    logging_steps: int = 10,
    save_steps: int = 200,
    eval_steps: int = 200,
    max_steps: int = -1,
):
    """Construit la DPOConfig TRL en ne passant que les arguments supportés. Retourne
    (config, arguments_ignorés)."""
    from trl import DPOConfig
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
        beta=beta,
        max_length=profile.max_length,
        max_prompt_length=profile.max_length // 2,
        bf16=profile.bf16,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        report_to=report_to,
        seed=seed,
    )
    return _construct(DPOConfig, desired)


def run_dpo(
    profile_name: Optional[str] = None,
    smoke: bool = False,
    dry_run: bool = False,
    epochs: float = 1.0,
    learning_rate: float = 5e-6,
    beta: float = 0.1,
    report_to: str = "none",
    resume: bool = False,
    sft_dir: str = "models/sft-lora",
    config_path: Optional[str] = None,
) -> dict:
    """Lance (ou prépare) le DPO. Retourne un dict de métadonnées du run."""
    cfg = load_config(config_path)
    audit = AuditLogger(PROJECT_ROOT / cfg.audit.log_path)
    seed = cfg.model.seed
    base_model = cfg.model.base_model_id
    proc = PROJECT_ROOT / cfg.data.processed_dir
    out_dir = PROJECT_ROOT / "models" / "dpo-lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = PROJECT_ROOT / sft_dir

    profile = resolve_profile(profile_name, smoke)
    has_token = bridge_hf_token()
    vram = detect_vram_gb()

    limit = 64 if smoke else None
    train_rows = load_jsonl(proc / "splits" / "dpo_train.jsonl", limit=limit)
    val_rows = load_jsonl(proc / "splits" / "dpo_val.jsonl", limit=16 if smoke else None)

    meta = {
        "base_model": base_model,
        "sft_dir": str(sft_path),
        "sft_adapter_present": (sft_path / "adapter_config.json").exists(),
        "profile": profile.name,
        "profile_params": asdict(profile),
        "vram_gb": vram,
        "hf_token_bridged": has_token,
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "beta": beta,
        "seed": seed,
        "output_dir": str(out_dir),
        "dry_run": dry_run,
        "smoke": smoke,
    }
    audit.log("dpo.train.start", meta)

    if dry_run:
        meta["status"] = "dry_run_ok"
        audit.log("dpo.train.dryrun", {"status": "ok",
                                       "sft_adapter_present": meta["sft_adapter_present"]})
        return meta

    if not meta["sft_adapter_present"]:
        raise FileNotFoundError(
            f"Adaptateur SFT introuvable dans {sft_path}. Lance d'abord le SFT (étape 7)."
        )

    # --- Exécution réelle ---
    from datasets import Dataset
    import torch  # noqa: F401
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(str(sft_path))
    dtype = torch.bfloat16 if profile.bf16 else None
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    # Fusion de l'adaptateur SFT -> modèle de politique de départ.
    policy = PeftModel.from_pretrained(base, str(sft_path)).merge_and_unload()
    # merge_and_unload() laisse le modèle sur CPU : on le remet sur le GPU si dispo,
    # sinon l'entraînement DPO tournerait sur CPU (extrêmement lent).
    if torch.cuda.is_available():
        policy = policy.to("cuda")
    print(f"[info] modèle de politique sur : {next(policy.parameters()).device}")

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(val_rows) if val_rows else None

    max_steps = 4 if smoke else -1
    dpo_config, dropped = build_dpo_config(
        output_dir=str(out_dir), profile=profile, epochs=epochs,
        learning_rate=learning_rate, beta=beta, report_to=report_to,
        seed=seed, max_steps=max_steps,
    )
    if dropped:
        print(f"[info] arguments DPOConfig ignorés (non supportés par ta version TRL) : {dropped}")
        audit.log("dpo.train.config_dropped", {"dropped": dropped})

    trainer = DPOTrainer(
        model=policy,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=build_lora_config(),
    )
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    meta["status"] = "trained"
    audit.log("dpo.train.done", {"output_dir": str(out_dir), "profile": profile.name})
    return meta
