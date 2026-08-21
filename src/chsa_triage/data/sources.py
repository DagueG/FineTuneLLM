"""Registre des sources de données et leurs normaliseurs.

Chaque source décrit : son identifiant sur le Hub, la langue, le type, la licence, le
split, et une fonction `normalize(raw_row, idx) -> Example | None` qui convertit une
ligne brute vers le schéma canonique. `normalize` renvoie None si la ligne est
inexploitable (elle sera ignorée proprement).

Robustesse : les normaliseurs cherchent les colonnes de façon tolérante (insensible à
la casse, plusieurs noms possibles), car les schémas amont varient et peuvent changer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .schema import Example, Kind

# --------------------------------------------------------------------------------------
# Helpers de recherche tolérante de colonnes
# --------------------------------------------------------------------------------------

def _get(row: dict[str, Any], *candidates: str) -> Optional[Any]:
    """Retourne la 1re valeur non vide parmi des noms de colonnes candidats (casse ignorée)."""
    lower = {str(k).lower(): v for k, v in row.items()}
    for c in candidates:
        v = lower.get(c.lower())
        if v not in (None, "", []):
            return v
    return None


def _as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(_as_text(x) for x in v)
    return str(v).strip()


# --------------------------------------------------------------------------------------
# Normaliseurs par source
# --------------------------------------------------------------------------------------

def _norm_qa(row: dict[str, Any], idx: int, src: "Source") -> Optional[Example]:
    q = _as_text(_get(row, "question", "Question", "instruction", "input", "query"))
    a = _as_text(_get(row, "answer", "Answer", "output", "response", "long_answer"))
    if not q or not a:
        return None
    return Example(
        id=f"{src.key}:{idx}",
        source=src.key,
        language=src.language,
        kind="qa",
        instruction=q,
        output=a,
        meta={"license": src.license, "qtype": _as_text(_get(row, "qtype", "question_type"))},
    )


def _render_options(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in options.items() if text)


def _norm_frenchmedmcqa(row: dict[str, Any], idx: int, src: "Source") -> Optional[Example]:
    q = _as_text(_get(row, "question"))
    options = {
        "A": _as_text(_get(row, "answer_a", "opa", "option_a")),
        "B": _as_text(_get(row, "answer_b", "opb", "option_b")),
        "C": _as_text(_get(row, "answer_c", "opc", "option_c")),
        "D": _as_text(_get(row, "answer_d", "opd", "option_d")),
        "E": _as_text(_get(row, "answer_e", "ope", "option_e")),
    }
    options = {k: v for k, v in options.items() if v}
    correct = _get(row, "correct_answers", "correct_answer", "answer", "cop")
    if isinstance(correct, str):
        correct_letters = [c.strip().upper() for c in correct.replace(",", " ").split() if c.strip()]
    elif isinstance(correct, (list, tuple)):
        correct_letters = [str(c).strip().upper() for c in correct]
    else:
        correct_letters = []
    if not q or not options or not correct_letters:
        return None
    correct_text = "; ".join(f"{l}. {options.get(l, '')}".strip() for l in correct_letters)
    return Example(
        id=f"{src.key}:{idx}",
        source=src.key,
        language=src.language,
        kind="mcqa",
        instruction=q,
        input=_render_options(options),
        output=f"Réponse(s) : {', '.join(correct_letters)}. {correct_text}".strip(),
        meta={"license": src.license, "correct_letters": correct_letters},
    )


def _norm_medmcqa(row: dict[str, Any], idx: int, src: "Source") -> Optional[Example]:
    """MedMCQA (défaut pour le créneau 'MediQA') : colonnes question, opa..opd, cop (0-3), exp."""
    q = _as_text(_get(row, "question"))
    options = {
        "A": _as_text(_get(row, "opa", "answer_a", "option_a")),
        "B": _as_text(_get(row, "opb", "answer_b", "option_b")),
        "C": _as_text(_get(row, "opc", "answer_c", "option_c")),
        "D": _as_text(_get(row, "opd", "answer_d", "option_d")),
    }
    options = {k: v for k, v in options.items() if v}
    cop = _get(row, "cop", "correct_option", "answer_idx")
    letters = ["A", "B", "C", "D"]
    correct_letter = None
    if isinstance(cop, int) and 0 <= cop < len(letters):
        correct_letter = letters[cop]
    elif isinstance(cop, str) and cop.strip().upper() in letters:
        correct_letter = cop.strip().upper()
    if not q or not options or correct_letter is None:
        return None
    exp = _as_text(_get(row, "exp", "explanation"))
    out = f"Answer: {correct_letter}. {options.get(correct_letter, '')}".strip()
    if exp:
        out += f"\nExplanation: {exp}"
    return Example(
        id=f"{src.key}:{idx}",
        source=src.key,
        language=src.language,
        kind="mcqa",
        instruction=q,
        input=_render_options(options),
        output=out,
        meta={"license": src.license, "correct_letter": correct_letter},
    )


def _extract_pref_texts(row: dict[str, Any]) -> tuple[str, str, str]:
    """Extrait (prompt, chosen, rejected) de façon tolérante à plusieurs schémas.

    Gère : (a) colonnes plates chosen/rejected (+ prompt/question), (b) chosen/rejected
    sous forme de listes de messages [{role/content} ou {from/value}].
    """
    def msg_text(x: Any) -> str:
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, list):
            # prend le dernier message (réponse de l'assistant)
            for m in reversed(x):
                if isinstance(m, dict):
                    return _as_text(m.get("content") or m.get("value"))
            return ""
        if isinstance(x, dict):
            return _as_text(x.get("content") or x.get("value") or x.get("text"))
        return _as_text(x)

    prompt = _as_text(_get(row, "prompt", "question", "instruction", "input"))
    chosen = msg_text(_get(row, "chosen", "chosen_response", "response_chosen"))
    rejected = msg_text(_get(row, "rejected", "rejected_response", "response_rejected"))
    return prompt, chosen, rejected


def _norm_preference(row: dict[str, Any], idx: int, src: "Source") -> Optional[Example]:
    prompt, chosen, rejected = _extract_pref_texts(row)
    if not (prompt and chosen and rejected):
        # Schéma non reconnu : on garde le brut en meta pour diagnostic à l'étape 4.
        return None
    return Example(
        id=f"{src.key}:{idx}",
        source=src.key,
        language=src.language,
        kind="preference",
        instruction=prompt,
        chosen=chosen,
        rejected=rejected,
        meta={"license": src.license},
    )


# --------------------------------------------------------------------------------------
# Définition des sources
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Source:
    key: str
    hf_id: str
    language: str            # "fr" | "en"
    kind: Kind
    license: str
    split: str
    normalize: Callable[[dict, int, "Source"], Optional[Example]]
    trust_remote_code: bool = False
    fallback_file: str = ""  # nom du fichier sous data/fallback/


REGISTRY: dict[str, Source] = {
    "medquad": Source(
        key="medquad", hf_id="lavita/MedQuAD", language="en", kind="qa",
        license="NIH/NLM — voir carte du dataset", split="train",
        normalize=_norm_qa, fallback_file="medquad.json",
    ),
    "frenchmedmcqa": Source(
        key="frenchmedmcqa", hf_id="qanastek/frenchmedmcqa", language="fr", kind="mcqa",
        license="Apache-2.0", split="train",
        normalize=_norm_frenchmedmcqa, trust_remote_code=True,
        fallback_file="frenchmedmcqa.json",
    ),
    # Créneau "MediQA" (ambigu dans le brief) : défaut = MedMCQA (EN, QCM, parquet propre).
    # Modifiable : il suffit de changer hf_id / normalize ici une fois la source confirmée.
    "mediqa": Source(
        key="mediqa", hf_id="openlifescienceai/medmcqa", language="en", kind="mcqa",
        license="voir carte du dataset (MedMCQA)", split="train",
        normalize=_norm_medmcqa, fallback_file="mediqa.json",
    ),
    "ultramedical_pref": Source(
        key="ultramedical_pref", hf_id="TsinghuaC3I/UltraMedical-Preference",
        language="en", kind="preference", license="MIT", split="train",
        normalize=_norm_preference, fallback_file="ultramedical_pref.json",
    ),
}
