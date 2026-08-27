# Fiche de soutenance — POC Agent IA de triage médical (CHSA)

*Support de présentation. À lire de haut en bas : contexte → architecture → décisions →
itérations → résultats → limites → roadmap.*

---

## 1. Contexte & objectif

Le service des urgences du Centre Hospitalier Saint-Aurélien est en surcharge. Objectif :
un **POC** d'agent IA d'**aide au triage** qui assiste le personnel soignant (collecte de
symptômes, évaluation de priorité, explications), **sans poser de diagnostic** ni remplacer
un soignant.

**Approche** : post-training d'un `Qwen3-1.7B-Base` — fine-tuning supervisé (SFT) + LoRA,
puis alignement par préférences (DPO) — sur un dataset médical bilingue, puis déploiement
d'un endpoint. **Ce qui compte pour un POC : démontrer la faisabilité de bout en bout.**

---

## 2. Architecture de la solution

```
Sources publiques (FR/EN)
  MedQuAD · FrenchMedMCQA · MedMCQA · UltraMedical-Preference
        │  ingestion + normalisation (schéma canonique)
        ▼
  Dataset SFT (5 000) ── DPO (3 000)
        │  anonymisation RGPD · dédup · splits anti-fuite · éval clinique séparée
        ▼
  SFT LoRA (Qwen3-1.7B)  ──►  DPO (alignement)
        │  évaluation clinique + contrôles de sécurité
        ▼
  Endpoint FastAPI + vLLM · Docker · CI/CD (à venir)
```

Fil rouge transverse : **traçabilité auditable** (journal à chaîne de hachage) à chaque
étape, et **robustesse *fallback-first*** (rien ne casse la démo).

---

## 3. Décisions techniques clés (et pourquoi)

| Décision | Justification |
|---|---|
| **Format conversationnel `messages`** (SFT & DPO) | Standard TRL, même cadre SFT↔DPO |
| **LoRA bf16 + profils VRAM auto** | Portable (T4 Colab → RTX 3060 → A100), peu de VRAM |
| **`datasets < 4`** | Les versions récentes ont supprimé les scripts de chargement (FrenchMedMCQA) |
| **Normalisation précoce → JSONL** | Découple des sources et des versions de libs |
| **Journal d'audit à chaîne de hachage** | Traçabilité **inviolable** exigée en contexte médical |
| **Split conscient des groupes** | Garantit l'absence de fuite train/éval même avec doublons |
| **Anonymisation à 2 modes (full / training)** | Arbitrage confidentialité ↔ utilité du modèle |
| **Plafonnement QCM à 30 %** | Le modèle doit répondre en prose, pas générer des QCM |
| **Contrôles de sécurité conservateurs** | Ne jamais minimiser un cas grave (faux positif toléré) |

---

## 4. Les itérations de debug (le cœur de la démarche)

Chaque problème a été **diagnostiqué à la racine** puis **verrouillé par un test** :

1. **Fuite train/test** détectée par un contrôle automatique → cause : l'anonymisation
   recréait des doublons après déduplication → **split conscient des groupes**.
2. **Modèle qui génère `<PERSON>` en boucle** → cause : sur-masquage NER de termes médicaux
   → **mode d'anonymisation `training`** (PII structurées seulement).
3. **Réponses en format QCM** → cause : 50 % du dataset était du QCM → **plafonnement à 30 %**.
4. **Caractères parasites / bouclage en génération** → cause : modèle base + échantillonnage
   → **décodage glouton + anti-répétition + gestion EOS**, encapsulés dans un module testé.
5. **Faux positif du contrôle de sécurité** (réponse correcte signalée à tort) → cause :
   liste de mots d'escalade trop littérale → **élargissement des synonymes**.

> Message clé : *on ne fait confiance ni au modèle ni aux métriques aveuglément — on inspecte,
> on diagnostique, on corrige, on teste.*

---

## 5. Résultats (évaluation clinique, jeu séparé, 18 scénarios)

- Exactitude de priorité **globale ~0.39** ; **urgent 0.67**, **urgence_vitale 0.5**,
  **non_urgent 0.0**.
- **Meilleur en anglais qu'en français** (cohérent avec la couverture des données).
- **Sur-triage** systématique (non_urgent = 0) : le modèle **escalade** → comportement
  *prudent*, préférable à la sous-évaluation d'un cas grave.
- Le **filet de sécurité** a correctement signalé un cas vital FR non escaladé (AVC).

*Lecture honnête : scores modestes attendus pour un 1.7B peu aligné ; l'important est la
chaîne complète et une évaluation objective des forces/faiblesses.*

---

## 6. Points forts à mettre en avant

- **Chaîne complète et reproductible** : données → entraînement → éval → (déploiement).
- **RGPD & traçabilité par conception** : anonymisation documentée, secrets isolés, audit
  inviolable de bout en bout.
- **Robustesse** : fallbacks partout (datasets, inférence), profils matériels, reprise
  d'entraînement.
- **Qualité logicielle** : ~44 tests, un commit par étape, décisions documentées (ADR).

---

## 7. Limites connues & roadmap

| Limite | Piste d'amélioration |
|---|---|
| Faible qualité en **français** | Ajouter des Q/R FR **rédigées** (traduction MedQuAD, source FR en prose) |
| **Sur-triage** (non_urgent raté) | DPO + données de triage annotées par des cliniciens |
| Métriques automatiques approximatives | Évaluation clinique humaine, plus de scénarios |
| Modèle compact (1.7B) | Passage à l'échelle (32B+) une fois le POC validé (phase 3) |
| Anonymisation NER bruitée | Modèle NER médical dédié + allow-list |

**Prochaines étapes du projet** : alignement **DPO** (amélioration), **comparaison SFT vs DPO**
chiffrée, puis **déploiement** (FastAPI + vLLM, Docker, CI/CD GitHub Actions) et **rapport final**.

---

## 8. Le pitch en 3 phrases

> Nous avons construit, de bout en bout, un POC d'agent de triage : un dataset médical
> bilingue anonymisé (RGPD) et versionné, un `Qwen3-1.7B` spécialisé par SFT+LoRA, et une
> évaluation clinique orientée sécurité. La démarche a été rigoureuse — chaque problème
> diagnostiqué à la racine et verrouillé par un test, chaque étape tracée de façon auditable.
> Le POC démontre la faisabilité technique et clinique, et trace une roadmap claire vers la
> mise à l'échelle.
