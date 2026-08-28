# Guide de déploiement

> ⚠️ **Offre HF (mi-2026)** : les Spaces qui exécutent du code (Gradio/Docker) requièrent
> désormais un plan payant, SAUF **ZeroGPU** (GPU H200 gratuit, quota) et **Static**. Les deux
> chemins gratuits ci-dessous en tiennent compte.

Trois options, de la plus « démo » à la plus « production » :
1. **HF Space ZeroGPU** — gratuit, sert le **vrai modèle** sur GPU (recommandé pour la démo).
2. **Render** — gratuit, sert l'**API FastAPI en mode repli** (URL publique, sans modèle).
3. **vLLM sur GPU** — inférence rapide en production (procédure reproductible).

---

## 1. HF Space ZeroGPU (gratuit, vrai modèle) — RECOMMANDÉ

Sert le modèle SFT+DPO sur GPU H200 gratuit (quota). Fichiers prêts dans `deploy/hf-space/`.

### a. Pousser le modèle fusionné sur un dépôt de MODÈLE HF
(une fois, depuis ta machine où se trouve `models/dpo-merged`)
```
huggingface-cli login          # colle ton token HF (write)
huggingface-cli upload TON_USER/chsa-triage-dpo models\dpo-merged . --repo-type model
```
(Le dépôt est créé automatiquement s'il n'existe pas.)

### b. Créer le Space
- https://huggingface.co/new-space -> SDK : Gradio, Hardware : ZeroGPU, visibilité Public.

### c. Pousser les 3 fichiers du Space
Clone le repo du Space, copie-y les fichiers de `deploy/hf-space/` :
```
git clone https://huggingface.co/spaces/TON_USER/chsa-triage
copy deploy\hf-space\app.py chsa-triage\
copy deploy\hf-space\requirements.txt chsa-triage\
copy deploy\hf-space\README.md chsa-triage\
```
Dans `app.py`, remplace `REMPLACER_PAR_TON_USER/chsa-triage-dpo` par ton dépôt de modèle
(ou définis la variable `CHSA_MODEL_ID` dans Settings -> Variables du Space).
```
cd chsa-triage
git add -A && git commit -m "CHSA triage demo" && git push
```

### d. Résultat
Le Space se construit puis expose une URL publique avec une interface de démo :
`https://TON_USER-chsa-triage.hf.space`. Idéal pour la soutenance (démo en anglais).

> Si le build ZeroGPU réclame une version précise de `torch`, suis le message d'erreur (HF
> indique la version CUDA attendue) ; sinon les dépendances par défaut suffisent.

---

## 2. Render (gratuit) - API FastAPI en mode repli

Donne une URL publique de l'API (mode repli, sans modèle) à partir de notre `Dockerfile`.

1. Pousse le dépôt sur GitHub (voir plus bas si pas encore fait).
2. https://render.com -> New -> Web Service -> connecte le repo GitHub.
3. Render détecte le `Dockerfile` (l'image écoute déjà sur `$PORT`). Plan Free.
4. Déploie -> URL type `https://chsa-triage.onrender.com`.
5. Teste : `/health` (-> `mode: fallback`), `/triage`, `/docs`.

> Le tier gratuit se met en veille après inactivité (démarrage à froid ~30 s au 1er appel).

---

## 3. Inférence rapide avec vLLM (production, GPU Linux)

vLLM exige un GPU et Linux. Sur une instance GPU (RunPod, Lambda, serveur CHSA) :
```
python scripts/merge_model.py --which dpo        # produit models/dpo-merged
pip install vllm
vllm serve /chemin/models/dpo-merged \
    --host 0.0.0.0 --port 8001 \
    --served-model-name chsa-triage --max-model-len 2048
```
Interrogation (API compatible OpenAI) :
```
curl http://localhost:8001/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model":"chsa-triage",
  "messages":[{"role":"system","content":"You are a medical triage decision-support assistant for CHSA."},
              {"role":"user","content":"A 55-year-old man has chest pain radiating to the left arm."}],
  "max_tokens":160,"temperature":0}'
```
vLLM apporte le batching continu et une inférence bien plus rapide - l'optimisation demandée
par le brief. Architecture cible : façade FastAPI stable (traçabilité, repli) devant vLLM.

---

## Pousser le dépôt sur GitHub (si pas encore fait)
```
git remote add origin https://github.com/TON_USER/chsa-triage.git
git branch -M main
git push -u origin main
```
Le workflow CI/CD (`.github/workflows/ci.yml`) se lancera automatiquement (onglet Actions).

---

## Checklist « go / no-go » (mise en production)
- [ ] Endpoint accessible et `/health` OK.
- [ ] Latence mesurée en conditions réalistes (p50/p95).
- [ ] Secrets hors du dépôt (Settings -> Secrets/Variables ; secrets GitHub).
- [ ] Journal d'audit activé et vérifiable (`verify_chain`).
- [ ] Limites d'usage documentées (aide à la décision, pas de diagnostic ; faiblesse FR).
- [ ] Supervision post-déploiement (monitoring, revue clinique périodique).
