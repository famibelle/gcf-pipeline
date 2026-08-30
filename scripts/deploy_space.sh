#!/usr/bin/env bash
# Reconstruit l'index puis pousse space/ vers le Space Hugging Face.
#   scripts/deploy_space.sh [POTOMITAN/annotation-gcf] [--public]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
SPACE="${1:-POTOMITAN/annotation-gcf}"
VISIBILITE="prive"
[ "${2:-}" = "--public" ] && VISIBILITE="public"

# Le moteur de suggestion n'est stocké qu'une fois, dans docs/ : on le recopie
# au moment du déploiement plutôt que de le dupliquer dans le dépôt.
mkdir -p space/static/assets
cp -r docs/assets/kreyol space/static/assets/

.venv/bin/python scripts/build_pages.py
.venv/bin/python scripts/build_space_index.py
.venv/bin/python - "$SPACE" "$VISIBILITE" <<'PY'
import os, sys
from huggingface_hub import HfApi

espace, visibilite = sys.argv[1], sys.argv[2]
prive = visibilite != "public"
if not prive:
    # Le Space relaie l'audio du corpus gaté pour le compte de son visiteur.
    # Public, il rendrait librement écoutable ce que le gating protège : ce
    # n'est pas un réglage à prendre à la légère.
    print("ATTENTION : un Space public ouvre le corpus gaté à tout venant.")
    if input("  taper OUI pour confirmer : ").strip() != "OUI":
        raise SystemExit("annulé")

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(espace, repo_type="space", space_sdk="docker",
                private=prive, exist_ok=True)
# Le dépôt d'annotations doit exister avant la première correction, sinon le
# service garde tout en mémoire et le travail part à la mise en veille.
annotations = os.environ.get("ANNOT_DATASET", "POTOMITAN/gcf-annotations")
api.create_repo(annotations, repo_type="dataset", private=True, exist_ok=True)
print(f"dépôt d'annotations : {annotations}")
api.upload_folder(
    folder_path="space", repo_id=espace, repo_type="space",
    commit_message="Interface d'annotation et index des segments",
    ignore_patterns=["__pycache__/*", "*.pyc"],
)
print(f"déployé ({'privé' if prive else 'public'}) : https://huggingface.co/spaces/{espace}")
print("Reste à régler le secret HF_TOKEN dans Settings → Variables and secrets.")
PY
