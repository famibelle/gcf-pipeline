#!/usr/bin/env bash
# Regarder défiler les transcriptions au fil de leur écriture.
#   scripts/whisper_watch.sh          toutes
#   scripts/whisper_watch.sh -n 50    les 50 dernières puis la suite
cd "$(dirname "$0")/.." || exit 1
DEPUIS=0
[ "${1:-}" = "-n" ] && DEPUIS="${2:-20}"

tail -n "$DEPUIS" -f data/metadata.whisper.jsonl | .venv/bin/python -u -c '
import json, sys
V, J, G, R = "\033[0m", "\033[33m", "\033[90m", "\033[36m"
for ligne in sys.stdin:
    ligne = ligne.strip()
    if not ligne:
        continue
    try:
        r = json.loads(ligne)
    except json.JSONDecodeError:
        continue
    nom = r["file_name"]
    court = nom.split("/")[-1]
    if len(court) > 46:
        court = court[:43] + "..."
    dossier = nom.split("/")[0]
    txt = r["transcription"].strip()
    if not txt:
        print(f"{G}{dossier:>7} {court:<46} — vide ou écarté{V}", flush=True)
    else:
        print(f"{R}{dossier:>7}{V} {court:<46} {J}{txt}{V}", flush=True)
'
