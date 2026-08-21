"""Templates de mise en forme SFT (format conversationnel `messages`).

Choix de conception (voir ADR-0003) :

- On produit un format **conversationnel** (`messages`: system/user/assistant), standard
  moderne géré nativement par TRL `SFTTrainer` (application du chat template, possibilité
  d'entraîner sur les tokens de l'assistant uniquement via `assistant_only_loss`).

- Le prompt système encadre un assistant **d'aide à la décision**, prudent : il répond de
  façon exacte, signale les signes d'alerte urgents, et renvoie au clinicien en cas de
  doute. On NE force PAS un format rigide « niveau de priorité » que les données QA/QCM ne
  portent pas : l'étiquetage explicite en 3 niveaux (vital / urgent / non urgent) relève de
  données annotées par des cliniciens (phases 2/3). Le POC installe d'abord un comportement
  médical sûr et exact. C'est un choix honnête vis-à-vis des données disponibles.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPTS: dict[str, str] = {
    "fr": (
        "Tu es un assistant d'aide au triage médical destiné au personnel soignant du "
        "Centre Hospitalier Saint-Aurélien. Tu n'établis pas de diagnostic définitif et "
        "tu ne remplaces pas un professionnel de santé. Réponds de façon claire et exacte, "
        "signale les signes d'alerte nécessitant une prise en charge immédiate, et "
        "recommande une évaluation par un clinicien en cas de doute ou d'information "
        "insuffisante."
    ),
    "en": (
        "You are a medical triage decision-support assistant for the clinical staff of "
        "Centre Hospitalier Saint-Aurélien. You do not provide a definitive diagnosis and "
        "you do not replace a healthcare professional. Answer clearly and accurately, flag "
        "red-flag symptoms that require immediate care, and recommend clinician evaluation "
        "when in doubt or when information is insufficient."
    ),
}


def build_user_content(example: dict[str, Any]) -> str:
    """Construit le message utilisateur (question + options si QCM)."""
    instruction = (example.get("instruction") or "").strip()
    extra = (example.get("input") or "").strip()
    if extra:
        return f"{instruction}\n\n{extra}"
    return instruction


def target_answer(example: dict[str, Any]) -> str:
    """Réponse cible : `output` pour qa/mcqa, `chosen` pour une source de préférences."""
    return (example.get("output") or example.get("chosen") or "").strip()


def to_sft_messages(example: dict[str, Any]) -> dict[str, Any] | None:
    """Convertit un exemple normalisé en enregistrement SFT conversationnel.

    Retourne None si le contenu minimal (question + réponse) est absent.
    """
    lang = example.get("language", "en")
    if lang not in SYSTEM_PROMPTS:
        lang = "en"
    user = build_user_content(example)
    answer = target_answer(example)
    if not user or not answer:
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[lang]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "source": example.get("source", ""),
            "language": lang,
            "kind": example.get("kind", ""),
        },
    }
