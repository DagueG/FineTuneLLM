"""Anonymisation des données (RGPD) via Presidio, avec repli regex.

Conception (voir ADR-0005) :
- **Presidio** est le moteur principal : `AnalyzerEngine` (détection d'entités via NER
  spaCy) + `AnonymizerEngine` (masquage). Multilingue FR/EN.
- Modèles spaCy : `fr_core_news_md` / `en_core_web_lg` demandés par défaut (cf. brief),
  avec **repli automatique** sur les variantes `sm` si les `md/lg` ne sont pas installées.
- **Filet de sécurité regex** : si Presidio ou spaCy n'est pas disponible du tout, on
  bascule sur un détecteur regex (emails, téléphones, dates, NIR) pour ne jamais laisser
  passer les PII structurées et pour que la pipeline tourne même en CI léger.
- **3 stratégies** de masquage : `replace` (`<PERSON>`), `mask` (`****`), `redact` (suppr.).
- **Contrôle qualité** : `residual_entities()` re-scanne le texte anonymisé pour vérifier
  qu'aucune PII ciblée ne subsiste.

Ce module reste agnostique du contenu : il s'applique aussi bien aux questions qu'aux
réponses, dans les deux langues.
"""

from __future__ import annotations

import re
from typing import Optional

# Entités ciblées par défaut (PERSON = minimum requis par le brief : nom/prénom patients).
DEFAULT_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "DATE_TIME",
    "LOCATION", "CREDIT_CARD", "IBAN_CODE",
]

# Entités "préservant l'utilité" pour les données D'ENTRAÎNEMENT : uniquement les PII
# structurées (vrai risque, détection fiable). On évite PERSON/LOCATION/DATE_TIME dont les
# petits/moyens modèles NER font de nombreux faux positifs sur du vocabulaire médical, ce qui
# dégrade la qualité du modèle entraîné. Justifié car les corpus sont publics et déjà
# dé-identifiés à la source. Le masquage NER complet reste disponible (mode "full"), destiné
# aux vraies données patients en production.
TRAINING_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE"]

_MODEL_CANDIDATES = {
    "fr": ["fr_core_news_md", "fr_core_news_sm"],
    "en": ["en_core_web_lg", "en_core_web_sm"],
}

# --- Regex du filet de sécurité (entités structurées, indépendantes de la langue) ---
_REGEXES: dict[str, re.Pattern] = {
    "EMAIL_ADDRESS": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE_NUMBER": re.compile(r"(?:(?:\+|00)33|0)\s*[1-9](?:[\s.\-]*\d{2}){4}"),
    "DATE_TIME": re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b"),
    "FR_NIR": re.compile(r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}(?:\s?\d{2})?\b"),
    "IBAN_CODE": re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}\b"),
}


def _mask_value(entity_type: str, value: str, strategy: str) -> str:
    if strategy == "redact":
        return ""
    if strategy == "mask":
        return "*" * len(value)
    return f"<{entity_type}>"  # replace


class Anonymizer:
    """Anonymiseur Presidio-first avec repli regex."""

    def __init__(
        self,
        strategy: str = "replace",
        languages: tuple[str, ...] = ("fr", "en"),
        entities: Optional[list[str]] = None,
        score_threshold: float = 0.5,
    ):
        assert strategy in {"replace", "mask", "redact"}
        self.strategy = strategy
        self.languages = languages
        self.entities = entities or DEFAULT_ENTITIES
        self.score_threshold = score_threshold
        self._backend: Optional[str] = None       # "presidio" | "regex"
        self._analyzer = None
        self._anonymizer = None
        self._supported: list[str] = []
        self._init_reason: str = ""

    # ------------------------------------------------------------------ init
    def _resolve_models(self) -> dict[str, str]:
        import spacy.util
        resolved = {}
        for lang in self.languages:
            for name in _MODEL_CANDIDATES.get(lang, []):
                if spacy.util.is_package(name):
                    resolved[lang] = name
                    break
        return resolved

    def _ensure(self) -> None:
        if self._backend is not None:
            return
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine

            resolved = self._resolve_models()
            if not resolved:
                raise RuntimeError("aucun modèle spaCy fr/en installé")

            nlp_conf = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": lg, "model_name": nm} for lg, nm in resolved.items()],
            }
            nlp_engine = NlpEngineProvider(nlp_configuration=nlp_conf).create_engine()
            self._analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine, supported_languages=list(resolved)
            )
            self._anonymizer = AnonymizerEngine()
            self._supported = list(resolved)
            self._backend = "presidio"
            self._init_reason = f"modèles={resolved}"
        except Exception as e:  # noqa: BLE001
            self._backend = "regex"
            self._init_reason = f"repli regex ({type(e).__name__}: {e})"

    @property
    def backend(self) -> str:
        self._ensure()
        return self._backend  # type: ignore[return-value]

    @property
    def init_reason(self) -> str:
        self._ensure()
        return self._init_reason

    # ------------------------------------------------------------- operators
    def _presidio_operators(self):
        from presidio_anonymizer.entities import OperatorConfig
        if self.strategy == "redact":
            return {"DEFAULT": OperatorConfig("redact", {})}
        if self.strategy == "mask":
            return {"DEFAULT": OperatorConfig("custom", {"lambda": lambda x: "*" * len(x)})}
        return None  # "replace" -> Presidio met <ENTITY_TYPE> par défaut

    # --------------------------------------------------------------- anonymize
    def anonymize(self, text: str, language: str = "en") -> tuple[str, list[str]]:
        """Retourne (texte_anonymisé, liste des types d'entités détectées)."""
        if not text:
            return text, []
        self._ensure()
        if self._backend == "presidio" and language in self._supported:
            results = self._analyzer.analyze(
                text=text, language=language,
                entities=self.entities, score_threshold=self.score_threshold,
            )
            out = self._anonymizer.anonymize(
                text=text, analyzer_results=results, operators=self._presidio_operators()
            )
            return out.text, [r.entity_type for r in results]
        # Repli regex (ou langue non couverte par Presidio)
        return self._regex_anonymize(text)

    def _regex_anonymize(self, text: str) -> tuple[str, list[str]]:
        found: list[str] = []
        out = text
        for etype, pat in _REGEXES.items():
            def _sub(m, et=etype):
                found.append(et)
                return _mask_value(et, m.group(0), self.strategy)
            out = pat.sub(_sub, out)
        return out, found

    def residual_entities(self, text: str, language: str = "en") -> list[str]:
        """Contrôle qualité : re-scanne un texte et retourne les entités encore présentes."""
        self._ensure()
        if self._backend == "presidio" and language in self._supported:
            results = self._analyzer.analyze(
                text=text, language=language,
                entities=self.entities, score_threshold=self.score_threshold,
            )
            return [r.entity_type for r in results]
        return self._regex_anonymize(text)[1]
