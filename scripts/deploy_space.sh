#!/usr/bin/env bash
# Reconstruit l'index puis pousse space/ vers le Space Hugging Face.
#   scripts/deploy_space.sh [POTOMITAN/annotation-gcf]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
SPACE="${1:-POTOMITAN/annotation-gcf}"

# Le moteur de suggestion n'est stocké qu'une fois, dans docs/ : on le recopie
# au moment du déploiement plutôt que de le dupliquer dans le dépôt.
mkdir -p space/static/assets
cp -r docs/assets/kreyol space/static/assets/

.venv/bin/python scripts/build_space_index.py
.venv/bin/python - "$SPACE" <<'PY'
import os, sys
from huggingface_hub import HfApi
espace = sys.argv[1]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(espace, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(
    folder_path="space", repo_id=espace, repo_type="space",
    commit_message="Interface d'annotation et index des segments",
    ignore_patterns=["__pycache__/*", "*.pyc"],
)
print(f"déployé : https://huggingface.co/spaces/{espace}")
print("Pense au secret HF_TOKEN dans Settings → Variables and secrets.")
PY
