# ADR-0012 — Déploiement cloud (HF Spaces + vLLM)

Statut : accepté (étape 14)
Date : 2026

## Contexte
Le brief demande un endpoint déployé sur le cloud (fournisseur au choix) et une inférence
rapide via vLLM. Contrainte : vLLM exige un GPU/Linux ; HF Spaces gratuit est CPU-only.

## Décisions
1. **HF Spaces (Docker SDK)** pour un endpoint public gratuit — tourne en **mode repli**
   (CPU), suffisant pour une démo d'endpoint qui ne tombe jamais. Réutilise notre Dockerfile.
2. **vLLM sur instance GPU** documenté (`docs/DEPLOYMENT.md`) : fusion du modèle
   (`dpo-merged`) puis `vllm serve` (API compatible OpenAI) pour l'inférence rapide.
3. **Architecture cible** : façade FastAPI stable (traçabilité, repli) devant un backend
   d'inférence remplaçable (transformers en local, vLLM en production).
4. **Checklist go/no-go** fournie (latence, secrets, audit, limites d'usage, supervision).

## Conséquences
Le brief est couvert de façon pragmatique et honnête, sans coût cloud obligatoire : URL
publique (Spaces) + procédure vLLM reproductible pour la performance.
