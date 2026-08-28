# ADR-0010 — API de déploiement (FastAPI) avec repli

Statut : accepté (étape 11)
Date : 2026

## Décisions
1. **FastAPI** expose `GET /health` et `POST /triage`. Le module d'inférence `TriageModel`
   (étape 8) est réutilisé — un seul chemin de génération, testé.
2. **Fallback-first** : si le modèle est indisponible (pas de GPU, poids absents, erreur de
   chargement), l'API bascule sur un **repli rule-based** prudent (escalade sur signe
   d'alerte, renvoi au clinicien) et **répond quand même**. Une démo ne tombe jamais en panne.
3. **Traçabilité RGPD** : chaque appel émet un évènement `api.triage` dans le journal d'audit
   avec des **métadonnées seulement** (request_id, langue, priorité, mode, modèle) — jamais
   le texte patient brut.
4. **Modèle servi** : `models/dpo-merged` par défaut (modèle complet), surchargable via
   `CHSA_MODEL_DIR`. `TriageModel` accepte adaptateur ou modèle complet.

## vLLM
Le mode modèle utilise `transformers` (portable, Windows OK). **vLLM** (inférence rapide,
Linux) sera branché à l'étape de déploiement cloud comme backend alternatif — l'API restant
la même façade. vLLM n'est pas installé côté Windows.

## Limites
Le repli par mots-clés ne gère pas la négation (« pas de fièvre » -> urgent). Choix assumé :
sur-triage = direction prudente ; le repli n'est qu'un filet de secours.
