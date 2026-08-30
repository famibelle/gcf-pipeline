#!/usr/bin/env python3
"""Constitue un lot d'annotation pour la page statique (GitHub Pages).

Le corpus fait 2,3 Go et reste sous accès contrôlé : il ne peut ni tenir dans
le dépôt ni être lu par un visiteur non connecté. Ce script en extrait de quoi
travailler — les segments que la passe a écartés d'abord, ce sont eux qui
méritent un humain — les transcode en mono 32 kbps, et écrit un manifeste que
la page charge au démarrage.

Le jeton n'intervient qu'ici, à la construction : rien de gaté ne descend dans
la page publiée.

    scripts/build_pages_lot.py --max 500
    scripts/build_pages_lot.py --motif français --max 200 --lot francais
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
CORPUS = os.environ.get("CORPUS_DATASET", "POTOMITAN/potomitan-gcf-transcription")


def nom_local(chemin: str) -> str:
    """part01/x.mp3 devient part01__x.mp3 : un dossier plat, sans collision."""
    return chemin.replace("/", "__")


def transcoder(source: Path, cible: Path, debit: int) -> bool:
    # Mono 16 kHz : la parole n'y perd rien et le fichier divise par trois.
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-ac", "1",
         "-ar", "16000", "-b:a", f"{debit}k", str(cible)],
        capture_output=True,
    )
    return r.returncode == 0 and cible.exists()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", default="space/data/index.jsonl")
    p.add_argument("--lot", default="rejets", help="nom du lot")
    p.add_argument("--motif", action="append",
                   help="motifs retenus (défaut : hallucination, français, vide)")
    p.add_argument("--tous", action="store_true",
                   help="tirer dans tout le corpus transcrit, sans filtre de motif")
    p.add_argument("--max", type=int, default=500, help="nombre de segments")
    p.add_argument("--debit", type=int, default=32, help="kbps du transcodage")
    p.add_argument("--duree-max", type=float, default=25.0, help="secondes")
    p.add_argument("--graine", type=int, default=1)
    p.add_argument("--travailleurs", type=int, default=4)
    args = p.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN absent : le corpus est gaté, la construction en a besoin.",
              file=sys.stderr)
        return 2
    from huggingface_hub import hf_hub_download

    index = RACINE / args.index
    if not index.exists():
        print(f"introuvable : {index} (scripts/build_space_index.py)", file=sys.stderr)
        return 2
    segments = [json.loads(l) for l in index.open(encoding="utf-8") if l.strip()]

    if args.tous:
        # Tirage sur tout le corpus : mesure le taux d'erreur réel, là où un lot
        # de rejets ne montre que les défauts déjà connus.
        retenus = [s for s in segments if 0 < s["d"] <= args.duree_max * 1000]
    else:
        motifs = set(args.motif or ["hallucination", "français", "vide"])
        retenus = [s for s in segments
                   if s["m"] in motifs and 0 < s["d"] <= args.duree_max * 1000]
    # Tirage reproductible : reconstruire le lot deux fois donne le même lot,
    # donc pas de fichiers orphelins dans le dépôt.
    random.Random(args.graine).shuffle(retenus)
    retenus = retenus[:args.max]
    if not retenus:
        print("aucun segment ne correspond.", file=sys.stderr)
        return 1

    dossier = RACINE / "docs" / "audio"
    dossier.mkdir(parents=True, exist_ok=True)
    brut = RACINE / "artifacts" / "lot_brut"
    brut.mkdir(parents=True, exist_ok=True)

    faits, echecs = [], []

    def traiter(seg: dict) -> None:
        cible = dossier / nom_local(seg["c"])
        if cible.exists():
            faits.append(seg)
            return
        try:
            source = hf_hub_download(CORPUS, seg["c"], repo_type="dataset",
                                     local_dir=str(brut))
        except Exception as err:
            echecs.append((seg["c"], type(err).__name__))
            return
        if transcoder(Path(source), cible, args.debit):
            faits.append(seg)
        else:
            echecs.append((seg["c"], "transcodage"))
        Path(source).unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=args.travailleurs) as pool:
        for n, _ in enumerate(pool.map(traiter, retenus), 1):
            if n % 50 == 0:
                print(f"  {n}/{len(retenus)}…", flush=True)

    faits.sort(key=lambda s: s["c"])
    manifeste = RACINE / "docs" / "data" / f"lot-{args.lot}.json"
    manifeste.parent.mkdir(parents=True, exist_ok=True)
    manifeste.write_text(json.dumps({
        "lot": args.lot,
        "corpus": CORPUS,
        "debit": args.debit,
        "segments": [{"c": s["c"], "f": nom_local(s["c"]), "t": s["t"],
                      "m": s["m"], "d": s["d"]} for s in faits],
    }, ensure_ascii=False), encoding="utf-8")

    # La page découvre les lots publiés par ce fichier, et bascule par ?lot=.
    lots = sorted(f.stem[len("lot-"):] for f in manifeste.parent.glob("lot-*.json"))
    (manifeste.parent / "lots.json").write_text(json.dumps(lots, ensure_ascii=False),
                                                encoding="utf-8")

    octets = sum((dossier / nom_local(s["c"])).stat().st_size for s in faits)
    secondes = sum(s["d"] for s in faits) / 1000
    print(f"\n  lot « {args.lot} » : {len(faits)} segments, {secondes/60:.0f} min d'audio")
    print(f"  {dossier.relative_to(RACINE)} : {octets/1024/1024:.1f} Mo "
          f"({octets/len(faits)/1024:.0f} Ko par segment)")
    print(f"  {manifeste.relative_to(RACINE)} : "
          f"{manifeste.stat().st_size/1024:.0f} Ko")
    if echecs:
        print(f"  {len(echecs)} échec(s) :")
        for chemin, raison in echecs[:5]:
            print(f"    {raison:<16} {chemin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
