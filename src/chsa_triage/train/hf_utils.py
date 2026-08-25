"""Utilitaires d'entraînement : pont token HF, détection VRAM, profils matériels.

Ces fonctions sont volontairement SANS dépendance lourde (pas de torch/trl importés au
niveau module) afin d'être testables sans GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from ..config import load_secrets


def bridge_hf_token() -> bool:
    """Fait le pont entre notre secret `CHSA_HF_TOKEN` et les variables attendues par les
    librairies Hugging Face (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`).

    Retourne True si un token a été trouvé et exporté.
    """
    token = load_secrets().hf_token
    if not token:
        return False
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return True


def detect_vram_gb() -> Optional[float]:
    """VRAM totale du GPU 0 en Go, ou None si pas de CUDA/torch. Import paresseux."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return round(props.total_memory / (1024 ** 3), 1)
    except Exception:
        return None


@dataclass
class TrainProfile:
    """Hyperparamètres dépendant du matériel."""

    name: str
    load_4bit: bool
    per_device_batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    gradient_checkpointing: bool
    bf16: bool = True


PROFILES: dict[str, TrainProfile] = {
    # ≤ 8 Go : QLoRA 4-bit, séquences courtes, forte accumulation.
    "low": TrainProfile("low", load_4bit=True, per_device_batch_size=1,
                        gradient_accumulation_steps=16, max_length=1024,
                        gradient_checkpointing=True),
    # 12–16 Go (T4, 3060, 4060…) : bf16 LoRA.
    "mid": TrainProfile("mid", load_4bit=False, per_device_batch_size=2,
                        gradient_accumulation_steps=8, max_length=2048,
                        gradient_checkpointing=True),
    # ≥ 24 Go (3090, 4090, A100…) : bf16 LoRA, gros batch.
    "high": TrainProfile("high", load_4bit=False, per_device_batch_size=8,
                         gradient_accumulation_steps=2, max_length=2048,
                         gradient_checkpointing=False),
    # Sans GPU / test rapide : minuscule, tourne sur CPU.
    "smoke": TrainProfile("smoke", load_4bit=False, per_device_batch_size=1,
                          gradient_accumulation_steps=1, max_length=512,
                          gradient_checkpointing=False, bf16=False),
}


def select_profile(vram_gb: Optional[float]) -> TrainProfile:
    """Choisit un profil à partir de la VRAM détectée."""
    if vram_gb is None:
        return PROFILES["smoke"]
    if vram_gb <= 8:
        return PROFILES["low"]
    if vram_gb <= 16:
        return PROFILES["mid"]
    return PROFILES["high"]
