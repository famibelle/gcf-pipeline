#!/usr/bin/env python3
"""Barre de progression tqdm pour la passe Whisper en cours.

Se branche sur le fichier de sortie plutôt que sur le processus : on peut la
lancer, la fermer et la relancer à volonté sans rien perturber, et elle marche
sur une passe démarrée avant elle.

    python scripts/whisper_progress.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "data/metadata.whisper.jsonl"
REJETS = RACINE / "data/metadata.whisper.jsonl.rejected.jsonl"
TOTAL = 40_333


def lignes(p: Path) -> int:
    try:
        with p.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def tourne() -> bool:
    return subprocess.run(["pgrep", "-f", "transcribe_corpus"],
                          capture_output=True, check=False).returncode == 0


def main() -> int:
    n = lignes(SORTIE)
    barre = tqdm(total=TOTAL, initial=n, unit="seg", dynamic_ncols=True,
                 smoothing=0.05, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    try:
        while True:
            time.sleep(3)
            nouveau = lignes(SORTIE)
            if nouveau > n:
                barre.update(nouveau - n)
                n = nouveau
            r = lignes(REJETS)
            barre.set_postfix_str(f"{r} écartés", refresh=False)
            if n >= TOTAL:
                break
            if not tourne():
                barre.set_postfix_str(f"{r} écartés — PASSE ARRÊTÉE", refresh=True)
                break
    except KeyboardInterrupt:
        pass
    finally:
        barre.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
