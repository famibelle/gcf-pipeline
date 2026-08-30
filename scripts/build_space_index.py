#!/usr/bin/env python3
"""Index des segments pour le Space d'annotation.

Le Space a besoin de savoir quoi proposer : chemin, transcription Whisper,
motif de rejet éventuel, durée. L'audio, lui, reste dans le dataset gaté et
n'est jamais copié — c'est toute la raison d'être de cette passerelle.

    scripts/build_space_index.py [--out space/data/index.jsonl]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
SEGMENT = re.compile(r"_segment_(\d+)_(\d+)\.mp3$")


def duree_ms(chemin: str) -> int:
    """Les bornes sont dans le nom du fichier ; inutile d'ouvrir l'audio."""
    m = SEGMENT.search(chemin)
    return int(m.group(2)) - int(m.group(1)) if m else 0


def lignes(fichier: Path):
    if not fichier.exists():
        return
    with fichier.open(encoding="utf-8-sig") as fh:
        for ligne in fh:
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                yield json.loads(ligne)
            except json.JSONDecodeError:
                continue


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--whisper", default="data/metadata.whisper.jsonl")
    p.add_argument("--out", default="space/data/index.jsonl")
    args = p.parse_args()

    source = RACINE / args.whisper
    rejets = RACINE / (args.whisper + ".rejected.jsonl")
    sortie = RACINE / args.out
    if not source.exists():
        print(f"introuvable : {source}")
        return 2

    # Les rejets d'abord : ce sont eux qui portent le motif, et le segment
    # rejeté n'apparaît pas toujours dans la sortie principale.
    motifs: dict[str, str] = {}
    textes_rejetes: dict[str, str] = {}
    for row in lignes(rejets):
        motifs[row["file_name"]] = row.get("motif", "rejet")
        textes_rejetes[row["file_name"]] = row.get("texte_rejete", "")

    index: dict[str, dict] = {}
    for row in lignes(source):
        chemin = row["file_name"]
        texte = (row.get("transcription") or "").strip()
        index[chemin] = {
            "c": chemin,
            "t": texte,
            "m": motifs.get(chemin, "" if texte else "vide"),
            "d": duree_ms(chemin),
        }
    # Un segment écarté n'a pas de transcription retenue : on garde le texte
    # rejeté sous les yeux de l'annotateur, c'est lui qu'il doit corriger.
    for chemin, motif in motifs.items():
        entree = index.setdefault(chemin, {"c": chemin, "t": "", "m": motif,
                                           "d": duree_ms(chemin)})
        entree["m"] = motif
        if not entree["t"]:
            entree["t"] = textes_rejetes.get(chemin, "")

    sortie.parent.mkdir(parents=True, exist_ok=True)
    with sortie.open("w", encoding="utf-8") as fh:
        for chemin in sorted(index):
            fh.write(json.dumps(index[chemin], ensure_ascii=False) + "\n")

    compte: dict[str, int] = {}
    for e in index.values():
        compte[e["m"] or "transcrit"] = compte.get(e["m"] or "transcrit", 0) + 1
    total_ms = sum(e["d"] for e in index.values())
    print(f"{sortie.relative_to(RACINE)} : {len(index):,} segments, "
          f"{sortie.stat().st_size/1024/1024:.1f} Mo, {total_ms/3600000:.1f} h d'audio")
    for motif, n in sorted(compte.items(), key=lambda kv: -kv[1]):
        print(f"  {motif:<16} {n:>7,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
