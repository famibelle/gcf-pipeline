#!/usr/bin/env bash
# Passe Whisper complète, détachée : elle survit à la fermeture du terminal.
# La passe reprend d'elle-même là où la sortie s'arrête ; rien n'est recalculé.
set -uo pipefail
cd /home/medhi/SourceCode/gcf-pipeline

# Se replacer hors du terminal appelant, sinon sa fermeture emporte la passe.
if [ "${WHISPER_DETACHE:-}" != "1" ]; then
  TS=$(date +%Y%m%d-%H%M%S)
  LOG="artifacts/logs/whisper-$TS.log"
  mkdir -p artifacts/logs
  : > "$LOG"
  ln -sf "whisper-$TS.log" artifacts/logs/whisper-latest.log
  WHISPER_DETACHE=1 WHISPER_LOG="$LOG" setsid nohup "$0" </dev/null >>"$LOG" 2>&1 &
  disown
  echo "passe lancée (PID $!) — journal : $LOG"
  echo "suivi : scripts/whisper_status.sh   |   scripts/whisper_watch.sh -n 20"
  exit 0
fi

LOG="$WHISPER_LOG"
set -a; . ./.env; set +a
{
  echo "=== lancement $(date -Is) ==="
  echo "sortie : data/metadata.whisper.jsonl"
} >> "$LOG"
PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/transcribe_corpus.py \
  --hf POTOMITAN/potomitan-gcf-transcription \
  --preserve data/metadata_hub.jsonl \
  --out data/metadata.whisper.jsonl >> "$LOG" 2>&1
echo "=== fin $(date -Is) — code $? ===" >> "$LOG"
