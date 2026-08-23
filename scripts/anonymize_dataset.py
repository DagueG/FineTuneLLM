"""Anonymise les datasets SFT et DPO (RGPD), avec rapport et contrôle qualité.

Pour chaque fichier (`sft.jsonl`, `dpo.jsonl`) :
1. anonymise tous les champs texte (user/assistant, prompt/chosen/rejected) selon la
   langue de l'exemple ;
2. écrit `*_anonymized.jsonl` ;
3. compte les entités détectées par type ;
4. **contrôle qualité** : re-scanne un échantillon anonymisé et vérifie qu'aucune PII
   ciblée ne subsiste (nombre d'entités résiduelles) ;
5. trace le tout dans le journal d'audit et écrit `anonymization_report.json`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

from chsa_triage.audit import AuditLogger
from chsa_triage.config import PROJECT_ROOT, load_config
from chsa_triage.data.anonymize import Anonymizer


def _lang_of(rec: dict) -> str:
    return rec.get("meta", {}).get("language", "en")


# PII structurées, détectables de façon déterministe : exigence stricte de 0 résiduel.
# Les entités NER (PERSON, LOCATION, DATE_TIME) sont bruitées avec les petits modèles
# spaCy (faux positifs sur du vocabulaire médical) : on les suit comme métrique, sans
# faire échouer le contrôle qualité dessus.
STRUCTURED_ENTITIES = {"EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE", "CREDIT_CARD", "FR_NIR"}


def _anonymize_messages(msgs: list[dict], anon: Anonymizer, lang: str, counter: dict) -> list[dict]:
    out = []
    for m in msgs:
        # On n'anonymise pas le prompt système (texte que NOUS écrivons, sans PII patient
        # ; il mentionne volontairement l'établissement déployeur).
        if m.get("role") == "system":
            out.append(m)
            continue
        clean, ents = anon.anonymize(m.get("content", ""), language=lang)
        for e in ents:
            counter[e] = counter.get(e, 0) + 1
        out.append({**m, "content": clean})
    return out


def _anonymize_record(rec: dict, anon: Anonymizer, counter: dict) -> dict:
    lang = _lang_of(rec)
    new = dict(rec)
    if "messages" in rec:  # format SFT
        new["messages"] = _anonymize_messages(rec["messages"], anon, lang, counter)
    else:  # format DPO
        for key in ("prompt", "chosen", "rejected"):
            if key in rec:
                new[key] = _anonymize_messages(rec[key], anon, lang, counter)
    return new


def _iter_texts(rec: dict):
    if "messages" in rec:
        for m in rec["messages"]:
            if m.get("role") == "system":
                continue
            yield m.get("content", ""), _lang_of(rec)
    else:
        for key in ("prompt", "chosen", "rejected"):
            for m in rec.get(key, []):
                if m.get("role") == "system":
                    continue
                yield m.get("content", ""), _lang_of(rec)


def anonymize_file(
    in_path: Path, out_path: Path, anon: Anonymizer, audit: AuditLogger, qc_sample: int
) -> dict:
    if not in_path.exists():
        return {"file": in_path.name, "status": "absent"}

    counter: dict[str, int] = {}
    records = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(_anonymize_record(json.loads(line), anon, counter))

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Contrôle qualité : re-scan d'un échantillon anonymisé.
    rng = random.Random(42)
    sample = rng.sample(records, min(qc_sample, len(records))) if records else []
    residual_structured = 0
    residual_ner = 0
    for rec in sample:
        for text, lang in _iter_texts(rec):
            for ent in anon.residual_entities(text, language=lang):
                if ent in STRUCTURED_ENTITIES:
                    residual_structured += 1
                else:
                    residual_ner += 1

    report = {
        "file": in_path.name,
        "output": out_path.name,
        "records": len(records),
        "entities_masked_by_type": counter,
        "entities_masked_total": sum(counter.values()),
        "qc_sample_size": len(sample),
        "qc_residual_structured": residual_structured,   # doit être 0
        "qc_residual_ner": residual_ner,                  # métrique (faux positifs possibles)
        "qc_passed": residual_structured == 0,
        "backend": anon.backend,
    }
    audit.log("data.anonymized", report)
    return report


def run(strategy: str = "replace", qc_sample: int = 200, config_path: Optional[str] = None) -> dict:
    cfg = load_config(config_path)
    proc = PROJECT_ROOT / cfg.data.processed_dir
    audit = AuditLogger(PROJECT_ROOT / cfg.audit.log_path)
    anon = Anonymizer(strategy=strategy)

    audit.log("anonymize.start", {"strategy": strategy, "backend": anon.backend,
                                  "init": anon.init_reason})

    reports = []
    for name in ("sft.jsonl", "dpo.jsonl"):
        in_path = proc / name
        out_path = proc / name.replace(".jsonl", "_anonymized.jsonl")
        reports.append(anonymize_file(in_path, out_path, anon, audit, qc_sample))

    result = {"backend": anon.backend, "init_reason": anon.init_reason, "reports": reports}
    with (proc / "anonymization_report.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    audit.log("anonymize.done", {"files": [r.get("file") for r in reports]})
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="Anonymisation RGPD des datasets CHSA.")
    p.add_argument("--strategy", choices=["replace", "mask", "redact"], default="replace")
    p.add_argument("--qc-sample", type=int, default=200)
    args = p.parse_args()

    res = run(strategy=args.strategy, qc_sample=args.qc_sample)
    print(f"\n== Anonymisation ({res['backend']}) ==")
    print(f"init : {res['init_reason']}")
    for r in res["reports"]:
        if r.get("status") == "absent":
            print(f"  - {r['file']} : absent (lance d'abord build_sft / build_dpo)")
            continue
        print(f"  - {r['file']} -> {r['output']} | {r['records']} enreg. | "
              f"{r['entities_masked_total']} entités masquées {r['entities_masked_by_type']}")
        print(f"      QC : PII structurées résiduelles={r['qc_residual_structured']} "
              f"-> {'OK' if r['qc_passed'] else 'ÉCHEC'} | "
              f"entités NER résiduelles={r['qc_residual_ner']} (faux positifs possibles, "
              f"sur {r['qc_sample_size']} échantillons)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
