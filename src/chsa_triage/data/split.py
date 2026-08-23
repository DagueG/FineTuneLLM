"""Découpage des datasets en train / val / test, stratifié et sans fuite.

Point de vigilance clé du brief : ne jamais mélanger entraînement et évaluation. On :
- stratifie par (source, langue) pour préserver l'équilibre dans chaque split ;
- utilise un seed déterministe (reproductibilité) ;
- vérifie l'absence de fuite : aucune même question (hash normalisé) présente à la fois
  dans train et dans val/test.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def _user_text(rec: dict) -> str:
    msgs = rec.get("messages") or rec.get("prompt") or []
    for m in msgs:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _strata_key(rec: dict) -> str:
    m = rec.get("meta", {})
    return f"{m.get('source','?')}|{m.get('language','?')}"


def _qhash(rec: dict) -> str:
    return hashlib.sha256(_norm(_user_text(rec)).encode()).hexdigest()


def stratified_split(
    records: list[dict[str, Any]],
    ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Découpe en train/val/test en stratifiant par (source, langue).

    Le découpage est **conscient des groupes** : toutes les lignes partageant la même
    question (hash normalisé) sont placées dans le MÊME split. Cela garantit l'absence de
    fuite même quand l'anonymisation a recréé des doublons (questions distinctes devenues
    identiques après masquage), sans supprimer de données.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "les ratios doivent sommer à 1"
    rng = random.Random(seed)

    # strate -> {hash_question -> [enregistrements]}
    strata: dict[str, dict[str, list[dict]]] = {}
    for rec in records:
        strata.setdefault(_strata_key(rec), {}).setdefault(_qhash(rec), []).append(rec)

    train, val, test = [], [], []
    for _, groups_map in sorted(strata.items()):
        groups = list(groups_map.values())
        rng.shuffle(groups)
        total = sum(len(g) for g in groups)
        n_train = int(total * ratios[0])
        n_val = int(total * ratios[1])
        acc = 0
        for g in groups:
            if acc < n_train:
                train += g
            elif acc < n_train + n_val:
                val += g
            else:
                test += g
            acc += len(g)
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def leakage_report(splits: dict[str, list[dict]]) -> dict[str, int]:
    """Compte les questions partagées entre train et val/test (doit être 0)."""
    def hashes(recs):
        return {hashlib.sha256(_norm(_user_text(r)).encode()).hexdigest() for r in recs}
    h_train = hashes(splits["train"])
    h_val = hashes(splits["val"])
    h_test = hashes(splits["test"])
    return {
        "train_val_overlap": len(h_train & h_val),
        "train_test_overlap": len(h_train & h_test),
        "val_test_overlap": len(h_val & h_test),
    }


def write_splits(splits: dict[str, list[dict]], out_dir: Path, prefix: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, recs in splits.items():
        p = out_dir / f"{prefix}_{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths[name] = str(p)
    return paths
