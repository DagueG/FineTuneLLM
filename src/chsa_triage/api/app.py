"""API FastAPI de l'agent de triage.

- `GET /health`  : état du service (modèle chargé ou mode repli).
- `POST /triage` : évalue une situation patient et renvoie priorité + explication.

Robustesse (fallback-first) : si le modèle est indisponible (pas de GPU, poids absents,
erreur), l'API bascule sur un repli rule-based et **répond quand même**. Chaque interaction
est tracée dans le journal d'audit (métadonnées seulement — pas de texte patient brut, RGPD).
"""

from __future__ import annotations

import os
import uuid
from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..audit import AuditLogger
from ..config import PROJECT_ROOT, load_config
from ..data.templates import SYSTEM_PROMPTS
from ..eval.priority import parse_priority
from .fallback import fallback_triage

DEFAULT_MODEL_DIR = os.environ.get("CHSA_MODEL_DIR", "models/dpo-merged")
DEVICE = os.environ.get("CHSA_DEVICE", "cuda")

_PRIORITY_INSTRUCTION = {
    "fr": "\n\nÉvalue la priorité de triage et termine par une ligne : PRIORITY: urgence_vitale | urgent | non_urgent",
    "en": "\n\nAssess the triage priority and end with a line: PRIORITY: urgence_vitale | urgent | non_urgent",
}


class Backend:
    """Charge le modèle si possible, sinon bascule en repli."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, device: str = DEVICE):
        self.mode = "fallback"
        self.model_id = "fallback"
        self.reason = ""
        self._model = None
        try:
            from ..infer.generate import TriageModel
            self._model = TriageModel.load(model_dir, device=device)
            self.mode = "model"
            self.model_id = model_dir
        except Exception as e:  # noqa: BLE001
            self.reason = f"{type(e).__name__}: {e}"

    def generate(self, text: str, language: str) -> str:
        if self.mode == "model" and self._model is not None:
            sys = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["en"])
            user = text + _PRIORITY_INSTRUCTION.get(language, _PRIORITY_INSTRUCTION["en"])
            messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
            return self._model.generate(messages)
        return fallback_triage(text, language)


class TriageRequest(BaseModel):
    text: str = Field(..., description="Description de la situation / symptômes du patient.")
    language: Literal["fr", "en"] = "fr"


class TriageResponse(BaseModel):
    request_id: str
    priority: Optional[str]
    response: str
    mode: str
    model: str


def create_app(backend: Optional[Backend] = None) -> FastAPI:
    app = FastAPI(title="CHSA Triage API", version="0.1.0")
    cfg = load_config()
    audit = AuditLogger(PROJECT_ROOT / cfg.audit.log_path)
    # Backend construit paresseusement au 1er accès si non fourni (utile pour les tests).
    state: dict = {"backend": backend}

    def get_backend() -> Backend:
        if state["backend"] is None:
            state["backend"] = Backend()
        return state["backend"]

    @app.get("/health")
    def health() -> dict:
        b = get_backend()
        return {"status": "ok", "mode": b.mode, "model": b.model_id,
                "reason": b.reason or None}

    @app.post("/triage", response_model=TriageResponse)
    def triage(req: TriageRequest) -> TriageResponse:
        b = get_backend()
        text = b.generate(req.text, req.language)
        priority = parse_priority(text)
        request_id = str(uuid.uuid4())
        # Traçabilité RGPD : on journalise des métadonnées, PAS le texte patient brut.
        audit.log("api.triage", {"request_id": request_id, "language": req.language,
                                 "priority": priority, "mode": b.mode, "model": b.model_id})
        return TriageResponse(request_id=request_id, priority=priority, response=text,
                              mode=b.mode, model=b.model_id)

    return app


# Instance par défaut pour `uvicorn chsa_triage.api.app:app`.
app = create_app()
