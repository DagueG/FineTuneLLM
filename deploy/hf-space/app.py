"""Space Gradio (ZeroGPU) — démo de l'agent de triage CHSA.

Le modèle fusionné (SFT+DPO) est chargé depuis un dépôt de MODÈLE Hugging Face (à pousser au
préalable, voir docs/DEPLOYMENT.md). ZeroGPU attache un GPU H200 le temps de la génération.

Aide à la décision — ne pose pas de diagnostic, ne remplace pas un soignant.
"""

import os
import re

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ⚠️ Remplace par TON dépôt de modèle (ex. "ton-user/chsa-triage-dpo").
MODEL_ID = os.environ.get("CHSA_MODEL_ID", "DagueGG/chsa-triage-dpo")

SYSTEM_PROMPTS = {
    "fr": ("Tu es un assistant d'aide au triage médical destiné au personnel soignant du "
           "Centre Hospitalier Saint-Aurélien. Tu n'établis pas de diagnostic définitif et tu "
           "ne remplaces pas un professionnel de santé. Réponds de façon claire et exacte, "
           "signale les signes d'alerte nécessitant une prise en charge immédiate, et "
           "recommande une évaluation par un clinicien en cas de doute."),
    "en": ("You are a medical triage decision-support assistant for the clinical staff of "
           "Centre Hospitalier Saint-Aurélien. You do not provide a definitive diagnosis and "
           "you do not replace a healthcare professional. Answer clearly and accurately, flag "
           "red-flag symptoms that require immediate care, and recommend clinician evaluation "
           "when in doubt."),
}

_HAS_LETTER = re.compile(r"[A-Za-zÀ-ÿ]")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
_EOS = tokenizer.convert_tokens_to_ids("<|im_end|>")
if _EOS is None or _EOS < 0:
    _EOS = tokenizer.eos_token_id


def _clean(text: str) -> str:
    lines = text.splitlines()
    while lines:
        first = lines[0].strip()
        if not first or (len(first) <= 12 and len(_HAS_LETTER.findall(first)) < 3):
            lines.pop(0)
            continue
        break
    return "\n".join(lines).strip()


@spaces.GPU
def triage(text: str, language: str) -> str:
    if not text.strip():
        return "Décris une situation / des symptômes."
    model.to("cuda")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["en"])},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(
        **enc, max_new_tokens=160, min_new_tokens=30, do_sample=False,
        repetition_penalty=1.3, no_repeat_ngram_size=3,
        eos_token_id=_EOS, pad_token_id=_EOS,
    )
    resp = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return _clean(resp)


demo = gr.Interface(
    fn=triage,
    inputs=[
        gr.Textbox(label="Situation / symptômes du patient", lines=4,
                   placeholder="A 55-year-old man has chest pain radiating to the left arm with sweating."),
        gr.Radio(["en", "fr"], value="en", label="Langue (meilleures réponses en anglais)"),
    ],
    outputs=gr.Textbox(label="Évaluation de triage (aide à la décision)"),
    title="🏥 CHSA — Agent d'aide au triage (POC)",
    description=("Aide à la décision destinée au personnel soignant. **Ne pose pas de "
                 "diagnostic et ne remplace pas un professionnel de santé.** Modèle "
                 "Qwen3-1.7B affiné (SFT+DPO). Limite connue : qualité moindre en français."),
    examples=[
        ["A 55-year-old man has chest pain radiating to the left arm with sweating.", "en"],
        ["A 3-year-old child is unresponsive and difficult to wake.", "en"],
        ["A 35-year-old with mild cold symptoms, no fever, feeling well.", "en"],
    ],
)

if __name__ == "__main__":
    demo.launch()
