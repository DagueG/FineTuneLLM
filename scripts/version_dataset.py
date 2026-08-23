"""Versionne le dataset : manifeste d'empreintes (SHA-256) + carte dataset.

Produit un manifeste reproductible (hash + nombre de lignes par fichier) qui sert de
« version » auditable du dataset, sans dépendance à un service externe. Peut ensuite être
poussé tel quel vers https://huggingface.co/datasets (voir README).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from chsa_triage.audit import AuditLogger
from chsa_triage.config import PROJECT_ROOT, load_config


def _sha256(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
            n += chunk.count(b"\n")
    return h.hexdigest(), n


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def build_manifest(version: str) -> dict:
    cfg = load_config()
    proc = PROJECT_ROOT / cfg.data.processed_dir
    eval_dir = PROJECT_ROOT / "data" / "eval"

    candidates = [
        proc / "sft_anonymized.jsonl",
        proc / "dpo_anonymized.jsonl",
        proc / "splits" / "sft_train.jsonl",
        proc / "splits" / "sft_val.jsonl",
        proc / "splits" / "sft_test.jsonl",
        proc / "splits" / "dpo_train.jsonl",
        proc / "splits" / "dpo_val.jsonl",
        proc / "splits" / "dpo_test.jsonl",
        eval_dir / "clinical_eval.jsonl",
    ]

    files = {}
    for p in candidates:
        if p.exists():
            digest, rows = _sha256(p)
            files[str(p.relative_to(PROJECT_ROOT))] = {"sha256": digest, "rows": rows}

    # Empreinte globale = hash des empreintes (ordre trié -> déterministe).
    fingerprint = hashlib.sha256(
        "".join(f"{k}:{v['sha256']}" for k, v in sorted(files.items())).encode()
    ).hexdigest()[:16]

    return {
        "name": "chsa-triage-medical-bilingual",
        "version": version,
        "fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "files": files,
    }


DATASET_CARD = """# Dataset — CHSA Triage médical bilingue

- **Version** : {version} (empreinte `{fingerprint}`)
- **Créé le** : {created_at}
- **Commit** : {git_commit}
- **Langues** : français, anglais
- **Contenu** : paires SFT (instruction→réponse, format conversationnel), paires DPO
  (chosen/rejected), splits train/val/test, jeu d'évaluation clinique séparé.
- **Sources & licences** : voir `data/raw/inventory.json` et les ADR
  (MedQuAD, FrenchMedMCQA, MedMCQA, UltraMedical-Preference).
- **Anonymisation** : Presidio (fr/en) + repli regex ; voir `docs/RGPD.md`.
- **Métadonnées** : schéma de triage dans `src/chsa_triage/data/metadata.py`.

## Fichiers versionnés
{files_block}

## Usage
```python
from datasets import load_dataset
ds = load_dataset("json", data_files={{
    "train": "data/processed/splits/sft_train.jsonl",
    "validation": "data/processed/splits/sft_val.jsonl",
    "test": "data/processed/splits/sft_test.jsonl",
}})
```

> Avertissement : outil d'**aide à la décision**, ne pose pas de diagnostic et ne remplace
> pas un professionnel de santé.
"""


def write_card(manifest: dict, path: Path) -> None:
    files_block = "\n".join(
        f"- `{k}` — {v['rows']} lignes — `{v['sha256'][:12]}…`"
        for k, v in sorted(manifest["files"].items())
    )
    path.write_text(DATASET_CARD.format(files_block=files_block, **manifest), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Versionnement du dataset CHSA.")
    p.add_argument("--version", default="1.0.0")
    args = p.parse_args()

    cfg = load_config()
    proc = PROJECT_ROOT / cfg.data.processed_dir
    audit = AuditLogger(PROJECT_ROOT / cfg.audit.log_path)

    manifest = build_manifest(args.version)
    (proc / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_card(manifest, PROJECT_ROOT / "docs" / "DATASET_CARD.md")
    audit.log("dataset.versioned", {"version": manifest["version"],
                                    "fingerprint": manifest["fingerprint"],
                                    "n_files": len(manifest["files"])})

    print(f"\n== Dataset versionné : {manifest['version']} (empreinte {manifest['fingerprint']}) ==")
    for k, v in sorted(manifest["files"].items()):
        print(f"  {v['rows']:>6} lignes  {v['sha256'][:12]}…  {k}")
    print("\nManifeste : data/processed/manifest.json | Carte : docs/DATASET_CARD.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
