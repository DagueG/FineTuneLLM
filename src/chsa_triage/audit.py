"""Journal d'audit inviolable (tamper-evident).

La mission exige la traçabilité de chaque interaction pour les audits médicaux, et
l'auditabilité de chaque transformation de données. On implémente un journal en
append-only au format JSONL où chaque enregistrement est chaîné au précédent par un
hachage SHA-256 (comme une mini-blockchain locale).

Conséquence : si quelqu'un modifie, insère ou supprime une ligne a posteriori, la
vérification de la chaîne échoue et pointe la première ligne incohérente. On obtient
une preuve d'intégrité sans dépendance externe.

Note RGPD : ce journal ne doit contenir QUE des données déjà anonymisées ou des
métadonnées non identifiantes. L'anonymisation (Presidio) sera branchée en amont dans
une étape ultérieure ; ce module reste volontairement agnostique du contenu.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS_HASH = "0" * 64


def _canonical(obj: dict[str, Any]) -> str:
    """Sérialisation JSON déterministe (clés triées) pour un hachage reproductible."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hash(prev_hash: str, record: dict[str, Any]) -> str:
    """Hache le lien (hash précédent + enregistrement courant sans son propre hash)."""
    return hashlib.sha256((prev_hash + _canonical(record)).encode("utf-8")).hexdigest()


class AuditLogger:
    """Écrit des évènements d'audit chaînés dans un fichier JSONL append-only."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        """Retourne le hash du dernier enregistrement, ou le hash génésis si vide."""
        if not self.log_path.exists():
            return GENESIS_HASH
        last = GENESIS_HASH
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = json.loads(line)["record_hash"]
        return last

    def log(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        """Ajoute un évènement au journal et retourne l'enregistrement écrit."""
        prev_hash = self._last_hash()
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "actor": actor,
            "payload": payload or {},
            "prev_hash": prev_hash,
        }
        record["record_hash"] = _hash(prev_hash, record)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(_canonical(record) + "\n")
        return record


def verify_chain(log_path: str | Path) -> tuple[bool, int]:
    """Vérifie l'intégrité de la chaîne d'audit.

    Retourne (ok, n) :
    - si ok=True  : n = nombre d'enregistrements valides vérifiés.
    - si ok=False : n = index (0-based) de la première ligne incohérente.
    """
    path = Path(log_path)
    if not path.exists():
        return True, 0
    prev = GENESIS_HASH
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stored = rec.pop("record_hash", None)
            if rec.get("prev_hash") != prev:
                return False, i
            if stored is None or _hash(prev, rec) != stored:
                return False, i
            prev = stored
            count += 1
    return True, count
