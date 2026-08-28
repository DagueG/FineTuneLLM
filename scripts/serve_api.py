"""Lance l'API de triage avec uvicorn.

  python scripts/serve_api.py           # http://127.0.0.1:8000  (docs: /docs)
Variables : CHSA_MODEL_DIR (défaut models/dpo-merged), CHSA_DEVICE (défaut cuda).
Sans modèle/GPU, l'API démarre quand même en mode repli.
"""
import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run("chsa_triage.api.app:app",
                host=os.environ.get("CHSA_HOST", "127.0.0.1"),
                port=int(os.environ.get("CHSA_PORT", "8000")),
                reload=False)
