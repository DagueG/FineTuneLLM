# ADR-0003 — Conception du dataset SFT

Statut : accepté (étape 3)
Date : 2026

## Décisions

1. **Format conversationnel `messages`** (system/user/assistant) plutôt que prompt-completion.
   Standard moderne géré nativement par TRL `SFTTrainer` : application automatique du chat
   template et entraînement possible sur les seuls tokens de l'assistant (`assistant_only_loss`).
   Directement chargeable via `datasets.load_dataset("json", data_files="sft.jsonl")`.
2. **Prompt système d'aide à la décision, prudent et bilingue** (FR/EN selon la langue de
   l'exemple) : réponse exacte, signalement des signes d'alerte, renvoi au clinicien en cas
   de doute. On **n'invente pas** d'étiquettes de priorité (vital/urgent/non urgent) absentes
   des données QA/QCM : l'étiquetage explicite relève de données annotées par des cliniciens
   (phases 2/3). Le POC installe d'abord un comportement médical sûr et exact — choix honnête.
3. **Sources SFT** : QA + QCM (MedQuAD, FrenchMedMCQA, MedMCQA). Les réponses `chosen` du jeu
   de préférences (UltraMedical) sont **aussi** utilisées comme cibles SFT (haute qualité),
   activable/désactivable (`--no-preference-chosen`). Le jeu de préférences reste par ailleurs
   dédié au DPO (étape 4).
4. **Qualité** : filtres de longueur (min question/réponse, max global) et rejet des exemples
   dégénérés (question == réponse).
5. **Déduplication** globale sur la question normalisée (minuscules + espaces), 1re occurrence
   conservée. Évite la sur-représentation et les fuites triviales.
6. **Équilibrage** : round-robin entre sources (mélange déterministe via `seed`) puis
   plafonnement à la cible (`sft_target_pairs`, défaut 5 000). Favorise un mélange de sources
   et de langues au lieu de laisser une source dominer.

## Traçabilité

Évènements `sft.build.start` / `sft.build.done` dans le journal d'audit (volumétries,
rejets qualité, doublons, répartition par langue, cible atteinte ou non). Statistiques
détaillées écrites dans `data/processed/sft_stats.json`.

## Conséquences

`data/processed/sft.jsonl` prêt pour le SFT. Sur les mini-jeux de repli la cible n'est pas
atteinte (peu d'exemples) : c'est normal. Sur données réelles, augmenter la volumétrie
d'ingestion (`--max-rows 0`) permet d'atteindre 5 000. Les splits train/val/test et le jeu
d'évaluation clinique seront constitués à l'étape 6 (séparation stricte train/éval).

## Mise à jour — plafonnement de la part QCM

`max_share_mcqa` (défaut 0.3) limite les sources QCM (FrenchMedMCQA, MedMCQA) à ~30 % du
mélange. Motivation : entraîné à 50 % sur du QCM, le modèle apprenait à *générer des questions
à choix multiples* au lieu de répondre en prose. En plafonnant le QCM et en laissant dominer
les réponses rédigées (MedQuAD + `chosen` d'UltraMedical), le comportement cible devient une
réponse d'aide au triage rédigée, sans perdre la couverture médicale apportée par le QCM.
Le savoir médical provient surtout du pré-entraînement ; le SFT ne fait qu'orienter la forme.
