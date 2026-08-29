#!/usr/bin/env bash
# Resynchronise le moteur de suggestion depuis le dépôt KreyolKeyb.
set -euo pipefail
SRC="${1:-../KreyolKeyb}/docs/assets"
DST="$(dirname "$0")/../docs/assets/kreyol"
[ -d "$SRC" ] || { echo "Introuvable : $SRC" >&2; exit 1; }
for f in simulateur-engine.js creole_dict.json creole_ngrams.json french_simple_dict.json; do
  cp -v "$SRC/$f" "$DST/"
done
echo "Pense à relancer la construction de l'interface artifact (audio embarqué)."
