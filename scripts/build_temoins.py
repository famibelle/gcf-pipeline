#!/usr/bin/env python3
"""Constitue les segments-témoins du Space d'annotation.

Un témoin est un extrait dont la bonne transcription est connue. Glissé sans
prévenir dans le flux de travail, il mesure la qualité d'un annotateur sans
qu'un second annotateur ait à repasser derrière — ce qui est la seule voie
praticable quand ils sont deux ou trois.

La référence ne s'invente pas : elle vient d'un CSV de corrections déjà
validées, typiquement l'export du Space relu par le propriétaire du corpus.

    scripts/build_temoins.py --depuis data/references_validees.csv
    scripts/build_temoins.py --depuis export.csv --colonne corrected --lister
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depuis", required=True, help="CSV validé (colonnes segment_id et corrected)")
    p.add_argument("--colonne", default="corrected", help="colonne portant la référence")
    p.add_argument("--cle", default="segment_id", help="colonne portant le chemin du segment")
    p.add_argument("--out", default="space/data/temoins.jsonl")
    p.add_argument("--index", default="space/data/index.jsonl")
    p.add_argument("--lister", action="store_true", help="afficher les témoins retenus")
    args = p.parse_args()

    source = RACINE / args.depuis if not Path(args.depuis).is_absolute() else Path(args.depuis)
    if not source.is_file():
        print(f"introuvable : {source}", file=sys.stderr)
        return 2

    connus: set[str] = set()
    index = RACINE / args.index
    if index.exists():
        with index.open(encoding="utf-8") as fh:
            connus = {json.loads(l)["c"] for l in fh if l.strip()}

    retenus: dict[str, str] = {}
    hors_index = 0
    with source.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            chemin = (row.get(args.cle) or "").strip()
            ref = (row.get(args.colonne) or "").strip()
            if not chemin or not ref:
                continue
            # Un témoin absent de l'index ne sera jamais servi : autant le dire
            # plutôt que de croire à un contrôle qui ne tourne pas.
            if connus and chemin not in connus:
                hors_index += 1
                continue
            retenus[chemin] = ref

    sortie = RACINE / args.out
    sortie.parent.mkdir(parents=True, exist_ok=True)
    with sortie.open("w", encoding="utf-8") as fh:
        for chemin in sorted(retenus):
            fh.write(json.dumps({"c": chemin, "ref": retenus[chemin]}, ensure_ascii=False) + "\n")

    print(f"{sortie.relative_to(RACINE)} : {len(retenus)} témoins")
    if hors_index:
        print(f"  {hors_index} écartés : absents de l'index, ils ne seraient jamais servis")
    if args.lister:
        for chemin in sorted(retenus):
            print(f"  {chemin}\n      {retenus[chemin][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
