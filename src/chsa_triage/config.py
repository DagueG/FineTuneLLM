"""Configuration du projet.

Principe de conception (voir ADR-0001) :
- La configuration "métier" (non sensible) vit dans `config/config.yaml`, versionnée dans Git.
- Les secrets (tokens API, clés) vivent dans des variables d'environnement (préfixe `CHSA_`)
  ou un fichier `.env` NON versionné. On ne mélange jamais secrets et config versionnée.

Un fallback gracieux est prévu : si le fichier de config est absent, on repart sur des
valeurs par défaut saines plutôt que de planter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du projet = deux niveaux au-dessus de ce fichier (src/chsa_triage/config.py -> racine)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ModelConfig(BaseModel):
    """Paramètres liés au modèle de langage."""

    base_model_id: str = "Qwen/Qwen3-1.7B-Base"
    max_seq_length: int = 2048
    seed: int = 42


class DataConfig(BaseModel):
    """Paramètres liés aux données."""

    languages: list[str] = Field(default_factory=lambda: ["fr", "en"])
    sft_target_pairs: int = 5000
    dpo_target_pairs: int = 3000
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


class AuditConfig(BaseModel):
    """Paramètres du journal d'audit (traçabilité)."""

    log_path: str = "logs/audit/audit_log.jsonl"
    enabled: bool = True


class AppConfig(BaseModel):
    """Configuration applicative complète (issue du YAML versionné)."""

    # `protected_namespaces=()` : autorise le champ `model` sans avertissement pydantic.
    model_config = ConfigDict(protected_namespaces=())

    app_name: str = "chsa-triage"
    environment: Literal["dev", "pilot", "prod"] = "dev"
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)


class Secrets(BaseSettings):
    """Secrets chargés depuis l'environnement / `.env` (jamais versionnés).

    Exemple : `export CHSA_HF_TOKEN=hf_xxx` alimente `Secrets().hf_token`.
    """

    model_config = SettingsConfigDict(env_prefix="CHSA_", env_file=".env", extra="ignore")

    hf_token: str | None = None
    wandb_api_key: str | None = None


def load_config(path: str | Path | None = None) -> AppConfig:
    """Charge la configuration applicative depuis un YAML.

    Fallback gracieux : si le fichier n'existe pas, retourne la config par défaut.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        return AppConfig()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(**raw)


def load_secrets() -> Secrets:
    """Charge les secrets depuis l'environnement / `.env`."""
    return Secrets()
