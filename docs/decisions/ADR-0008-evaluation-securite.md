# ADR-0008 — Évaluation clinique et contrôles de sécurité

Statut : accepté (étape 8)
Date : 2026

## Décisions
1. **Module d'inférence propre** (`infer/generate.py`) réutilisable par l'éval ET l'API :
   décodage glouton, pénalité de répétition, `no_repeat_ngram_size`, gestion de l'EOS Qwen
   (`<|im_end|>`), et nettoyage des tokens parasites de début (glyphes du modèle base).
2. **Harnais découplé** : l'évaluation prend un `generator` (callable messages -> texte), ce
   qui permet de tester toute la logique avec un modèle-jouet (`--mock`) sans GPU.
3. **Métrique clinique** sur le jeu d'éval séparé : exactitude du niveau de priorité (globale
   + par niveau), via un tag `PRIORITY:` demandé au modèle (parsing fiable) avec repli
   heuristique par mots-clés FR/EN. Plus un rappel heuristique des signes d'alerte.
4. **Contrôles de sécurité conservateurs** : sur les cas graves (urgence_vitale/urgent), on
   lève un drapeau si la réponse minimise la prise en charge sans orienter vers l'urgence
   (`dangerous_downplay`) ou n'escalade pas (`missing_escalation`). Principe : préférer un
   faux positif à un cas grave manqué.
5. **Traçabilité** : évènements `eval.start`/`eval.done` + rapport complet
   `data/processed/eval_report.json` (avec les réponses, pour audit clinique).

## Limites
Les métriques automatiques (parsing par mots-clés, rappel des signes d'alerte) sont des
approximations : une validation clinique humaine reste nécessaire. Le jeu d'éval (18 scénarios)
est un POC à enrichir par des cliniciens.
