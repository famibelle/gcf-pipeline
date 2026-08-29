#!/usr/bin/env bash
# Statut de la passe Whisper en cours.
cd "$(dirname "$0")/.." || exit 1
OUT=data/metadata.whisper.jsonl
LOG=artifacts/logs/whisper-latest.log
TOT=40333

if pgrep -f transcribe_corpus >/dev/null; then
  ETAT="en cours (PID $(pgrep -f transcribe_corpus | head -1))"
else
  ETAT="ARRÊTÉE"
fi

N=$([ -f "$OUT" ] && wc -l < "$OUT" || echo 0)
REJ=$([ -f "$OUT.rejected.jsonl" ] && wc -l < "$OUT.rejected.jsonl" || echo 0)
LARGEUR=40
PLEIN=$((N * LARGEUR / TOT))

printf '\n  passe Whisper : %s\n\n' "$ETAT"
printf '  ['
for i in $(seq 1 $LARGEUR); do [ "$i" -le "$PLEIN" ] && printf '#' || printf '.'; done
printf ']  %d / %d  (%.1f%%)\n\n' "$N" "$TOT" "$(echo "scale=2; 100*$N/$TOT" | bc)"

printf '  cadence   : %s\n' "$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep 'fichier/s' | tail -1 | sed 's/^ *//')"
printf '  écartés   : %s lignes (hallucinations et français)\n' "$REJ"
printf '  GPU       : %s\n' "$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)"
echo
echo "  dernière transcription :"
[ -f "$OUT" ] && tail -1 "$OUT" | cut -c1-150 | sed 's/^/    /'
echo
