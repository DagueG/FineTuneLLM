"""Inférence propre pour l'agent de triage.

Encapsule le chargement du modèle (base + adaptateur LoRA) et une génération robuste :
décodage glouton, pénalité de répétition, blocage des n-grammes répétés, gestion de l'EOS
Qwen (`<|im_end|>`), et nettoyage des quelques tokens parasites que le modèle base peut
émettre en tout début de génération.

Ce module est réutilisé par l'évaluation (étape 8) ET par l'API de déploiement (semaine 4).
Les imports lourds sont paresseux pour permettre de tester la logique de nettoyage sans GPU.
"""

from __future__ import annotations

import re
from typing import Optional

_HAS_LETTER = re.compile(r"[A-Za-zÀ-ÿ]")


def clean_generation(text: str) -> str:
    """Retire les lignes parasites en tête (glyphes / tokens non textuels du modèle base)
    jusqu'à la première ligne réellement textuelle, puis nettoie les espaces."""
    lines = text.splitlines()
    while lines:
        first = lines[0].strip()
        # ligne vide, ou trop peu de vraies lettres (ex. "𫟦", "魔龙令牌", "= ; 0.")
        letters = len(_HAS_LETTER.findall(first))
        if not first or (len(first) <= 12 and letters < 3):
            lines.pop(0)
            continue
        break
    return "\n".join(lines).strip()


class TriageModel:
    """Charge un modèle SFT/DPO (adaptateur LoRA) et génère des réponses de triage."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self._eos_id is None or self._eos_id < 0:
            self._eos_id = tokenizer.eos_token_id

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda") -> "TriageModel":
        """Charge l'adaptateur LoRA + son tokenizer (import paresseux)."""
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        model = AutoPeftModelForCausalLM.from_pretrained(model_dir).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return cls(model, tokenizer, device=device)

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 160,
        min_new_tokens: int = 30,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        """Génère une réponse de triage pour une liste de messages (system/user)."""
        import torch  # noqa: F401
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=self._eos_id,
            pad_token_id=self._eos_id,
        )
        text = self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return clean_generation(text)
