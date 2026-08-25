# ADR-0007 — Fine-tuning supervisé (SFT) LoRA

Statut : accepté (étape 7)
Date : 2026

## Décisions
1. **TRL `SFTTrainer`** sur nos JSONL conversationnels (`messages`), gérés nativement.
   `assistant_only_loss=True` : on n'entraîne que sur les tokens de l'assistant.
2. **LoRA bf16 par défaut** (r=16, alpha=32, dropout=0.05, target = q/k/v/o + MLP).
   Pas de bitsandbytes requis → robuste (notamment sous Windows). **QLoRA 4-bit** en option
   pour petite VRAM (profil `low`).
3. **Profils matériels auto-détectés** depuis la VRAM (low ≤8 Go / mid 12–16 Go / high ≥24 Go),
   override CLI possible. Fixe batch, accumulation, longueur de séquence, gradient checkpointing.
4. **Modes de validation progressifs** : `--dry-run` (prépare tout sans torch ni entraînement)
   puis `--smoke` (4 pas, sous-échantillon, CPU-friendly) avant le vrai run. Applique la reco
   « petits runs LoRA pour valider la pipeline avant la montée en charge ».
5. **Reproductibilité & reprise** : seed fixé, checkpoints (`save_steps`, `save_total_limit=2`),
   `--resume`. Logging optionnel (W&B, repli `WANDB_MODE=offline`), défaut `none`.
6. **Pont token HF** : `CHSA_HF_TOKEN` → `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` (le modèle base
   Qwen3-1.7B-Base est public, donc aucun token requis pour l'entraînement lui-même).
7. **Anti-sur-apprentissage** (point de vigilance du brief) : évaluation périodique sur le
   split `val` (`eval_strategy="steps"`) pour surveiller la généralisation.

## Limites de test
Impossible de valider le run réel dans l'environnement du binôme (pas de GPU, pas d'accès au
Hub, torch trop volumineux). Sont testés : logique de profils, pont token, chargement des
JSONL, `--dry-run` de bout en bout. Le run réel (`--smoke` puis complet) se valide côté GPU
de l'utilisateur ; les versions exactes de `transformers/trl/peft` seront confirmées là.

## Sortie
Adaptateur LoRA + tokenizer dans `models/sft-lora/`. La fusion LoRA→base pour le déploiement
sera traitée à l'étape déploiement.
