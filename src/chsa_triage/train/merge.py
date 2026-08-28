"""Fusion des adaptateurs LoRA en modèles complets.

Nécessaire pour :
- **le déploiement** (l'API / vLLM servent un modèle complet, pas un empilement d'adaptateurs) ;
- **l'évaluation correcte du DPO** : le DPO a été entraîné au-dessus du SFT fusionné, donc
  l'adaptateur DPO doit être appliqué sur le modèle SFT (et non sur la base brute).

`merge(base, [sft])`        -> modèle SFT complet.
`merge(base, [sft, dpo])`   -> modèle SFT+DPO complet (empilement correct).

Imports lourds paresseux.
"""

from __future__ import annotations

from pathlib import Path


def merge(base_model: str, adapter_dirs: list[str], out_dir: str, bf16: bool = True) -> str:
    """Applique séquentiellement des adaptateurs (avec fusion à chaque étape) et sauvegarde
    le modèle complet + le tokenizer du dernier adaptateur."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if bf16 else None
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    tok_src = adapter_dirs[-1]
    for d in adapter_dirs:
        model = PeftModel.from_pretrained(model, d).merge_and_unload()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(out_dir)
    return out_dir
