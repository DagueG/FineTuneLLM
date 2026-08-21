"""Smoke test de bout en bout — ne nécessite ni GPU ni réseau.

Objectif : prouver que l'ossature tient debout.
1. charge la configuration,
2. écrit quelques évènements dans le journal d'audit,
3. vérifie l'intégrité de la chaîne,
4. démontre que toute falsification est détectée.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from chsa_triage import __version__
from chsa_triage.audit import AuditLogger, verify_chain
from chsa_triage.config import load_config, load_secrets


def main() -> int:
    print(f"== CHSA Triage — smoke test (v{__version__}) ==")

    cfg = load_config()
    print(f"[config] app={cfg.app_name} env={cfg.environment} "
          f"modèle={cfg.model.base_model_id} cible_SFT={cfg.data.sft_target_pairs}")

    secrets = load_secrets()
    hf = "présent" if secrets.hf_token else "absent (normal à ce stade)"
    print(f"[secrets] token Hugging Face : {hf}")

    # On travaille dans un dossier temporaire pour ne rien polluer.
    with tempfile.TemporaryDirectory() as d:
        log_path = Path(d) / "audit.jsonl"
        logger = AuditLogger(log_path)
        logger.log("pipeline.start", {"step": "smoke_test"})
        logger.log("data.ingested", {"source": "demo", "rows": 3})
        logger.log("triage.evaluated", {"priority": "différée"}, actor="agent")

        ok, n = verify_chain(log_path)
        print(f"[audit] chaîne vérifiée : ok={ok}, enregistrements={n}")
        if not ok or n != 3:
            print("[audit] ÉCHEC : la chaîne devrait être valide avec 3 entrées.")
            return 1

        # Démonstration de la détection de falsification.
        lines = log_path.read_text(encoding="utf-8").splitlines()
        lines[1] = lines[1].replace('"rows":3', '"rows":9999')
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ok2, idx = verify_chain(log_path)
        print(f"[audit] après falsification : ok={ok2} (attendu False), "
              f"1re ligne incohérente = index {idx}")
        if ok2:
            print("[audit] ÉCHEC : la falsification aurait dû être détectée.")
            return 1

    print("\n✅ Smoke test réussi : config OK, audit OK, intégrité OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
