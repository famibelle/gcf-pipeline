#!/usr/bin/env python3
"""Qualité des annotations, sans second annotateur.

Avec deux ou trois personnes, personne ne repasse derrière personne : l'accord
inter-annotateurs est hors de portée. Trois mesures le remplacent.

- **Accord avec soi-même.** Un extrait resservi vierge des semaines plus tard
  produit une seconde version ; l'écart entre les deux dit la reproductibilité
  du travail aussi honnêtement qu'un accord entre deux personnes.
- **Écart aux témoins.** Des extraits dont la bonne transcription est connue,
  glissés sans prévenir dans le flux.
- **Drapeaux automatiques.** Ils ne disent pas « c'est faux », ils disent
  « à regarder » — et ce sont les seuls à tourner sur chaque segment.

    scripts/qualite_annotations.py --dir /chemin/corrections
    scripts/qualite_annotations.py --hf --csv artifacts/a_revoir.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import unicodedata
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "scripts"))
from transcribe_corpus import juger  # noqa: E402  (détecteur déjà écrit pour la passe)

MOT = None  # rempli à la première utilisation


# ---------------------------------------------------------------- distances

def normaliser(texte: str) -> str:
    """Les accents comptent en kréyòl : on ne les efface pas, seulement la
    ponctuation et la casse."""
    t = "".join(" " if unicodedata.category(c).startswith("P") else c for c in texte.lower())
    return " ".join(t.split())


def cer(reference: str, hypothese: str) -> float:
    """Taux d'erreur de caractères, borné à 1 quand tout est à refaire."""
    a, b = normaliser(reference), normaliser(hypothese)
    if not a:
        return 0.0 if not b else 1.0
    precedente = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        courante = [i]
        for j, cb in enumerate(b, 1):
            courante.append(min(precedente[j] + 1, courante[j - 1] + 1,
                                precedente[j - 1] + (ca != cb)))
        precedente = courante
    return min(1.0, precedente[-1] / len(a))


# ---------------------------------------------------------------- chargement

def charger_corrections(dossier: Path | None, depot: str) -> list[dict]:
    rows: list[dict] = []
    if dossier:
        fichiers = sorted(dossier.glob("*.jsonl"))
        if not fichiers:
            print(f"aucun .jsonl dans {dossier}", file=sys.stderr)
    else:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        fichiers = []
        for nom in api.list_repo_files(depot, repo_type="dataset"):
            if nom.startswith("corrections/") and nom.endswith(".jsonl"):
                fichiers.append(Path(api.hf_hub_download(depot, nom, repo_type="dataset")))
    for f in fichiers:
        with f.open(encoding="utf-8") as fh:
            for ligne in fh:
                ligne = ligne.strip()
                if ligne:
                    row = json.loads(ligne)
                    row.setdefault("annotateur", f.stem)
                    rows.append(row)
    return rows


def charger_jsonl(chemin: Path, cle: str) -> dict:
    index = {}
    if chemin.exists():
        with chemin.open(encoding="utf-8") as fh:
            for ligne in fh:
                if ligne.strip():
                    row = json.loads(ligne)
                    index[row[cle]] = row
    return index


def charger_lexique() -> set[str]:
    chemin = RACINE / "docs/assets/kreyol/creole_dict.json"
    if not chemin.exists():
        return set()
    return {m.lower() for m, _ in json.loads(chemin.read_text(encoding="utf-8"))}


# ---------------------------------------------------------------- drapeaux

def drapeaux(row: dict, seg: dict, lexique: set[str]) -> list[str]:
    texte = (row.get("corrected") or "").strip()
    if not texte:
        return []
    marques = []
    duree, ecoute = seg.get("d", 0), row.get("ecoute_ms", 0)
    lectures = row.get("lectures", 0)
    # L'inflation de notes est le seul travers que la note elle-même ne dit
    # pas : cinq étoiles sur un extrait qu'on n'a pas écouté ne valent rien.
    if (row.get("note") or 0) == 5 and duree and (not lectures or ecoute < 0.5 * duree):
        marques.append("5 étoiles sans avoir écouté")

    if lectures == 0:
        marques.append("jamais écouté")
    elif duree and ecoute < 0.4 * duree:
        marques.append("écoute partielle")
    # Reprendre la transcription telle quelle est légitime — si on l'a écoutée.
    if normaliser(texte) == normaliser(seg.get("t", "")) and duree and ecoute < 0.5 * duree:
        marques.append("validé sans écouter")

    garder, motif = juger(texte)
    if not garder:
        marques.append(f"correction en {motif}" if motif == "français" else f"correction : {motif}")

    # Le lexique ne juge que le texte inventé. Une transcription fidèle d'une
    # parole mêlée de français sort du lexique sans que ce soit une faute : on
    # n'alerte donc que si la correction s'éloigne aussi de ce qu'a entendu
    # Whisper.
    mots = [m for m in normaliser(texte).split() if len(m) > 2]
    if lexique and len(mots) >= 4 and cer(seg.get("t", ""), texte) > 0.5:
        inconnus = sum(1 for m in mots if m not in lexique)
        if inconnus / len(mots) > 0.7:
            marques.append("hors lexique et loin de la source")
    return marques


# ---------------------------------------------------------------- rapport

def mediane(valeurs: list[float]) -> float | str:
    return statistics.median(valeurs) if valeurs else "—"


def pourcent(v) -> str:
    return f"{100 * v:.1f} %" if isinstance(v, float) else str(v)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, help="dossier de corrections/*.jsonl")
    p.add_argument("--hf", action="store_true", help="télécharger depuis le dataset d'annotations")
    p.add_argument("--depot", default=os.environ.get("ANNOT_DATASET", "POTOMITAN/gcf-annotations"))
    p.add_argument("--csv", type=Path, help="où écrire les segments à revoir")
    args = p.parse_args()
    if not args.dir and not args.hf:
        p.error("précise --dir ou --hf")

    rows = charger_corrections(args.dir, args.depot)
    if not rows:
        print("aucune correction à examiner.")
        return 0
    index = charger_jsonl(RACINE / "space/data/index.jsonl", "c")
    temoins = charger_jsonl(RACINE / "space/data/temoins.jsonl", "c")
    lexique = charger_lexique()

    # (segment, annotateur) -> versions, dans l'ordre où elles ont été écrites
    versions: dict[tuple[str, str], list[dict]] = {}
    for row in sorted(rows, key=lambda r: r.get("at", 0)):
        versions.setdefault((row["id"], row["annotateur"]), []).append(row)

    par_annotateur: dict[str, dict] = {}
    a_revoir: list[dict] = []
    for (ident, annotateur), liste in sorted(versions.items()):
        seg = index.get(ident, {})
        stat = par_annotateur.setdefault(annotateur, {
            "corrections": 0, "rebuts": 0, "ecoute": [], "reprises": [], "temoins": [],
            "notes": [], "validees": 0,
        })
        derniere = liste[-1]
        note = derniere.get("note") or 0
        if note:
            stat["notes"].append(note)
        if note == 1:
            stat["rebuts"] += 1
        if note == 5:
            stat["validees"] += 1
        if not (derniere.get("corrected") or "").strip():
            continue
        stat["corrections"] += 1
        if seg.get("d"):
            stat["ecoute"].append(min(2.0, derniere.get("ecoute_ms", 0) / seg["d"]))
        # Accord avec soi-même : première version contre reprise.
        if len(liste) > 1 and (liste[0].get("corrected") or "").strip():
            stat["reprises"].append(cer(liste[0]["corrected"], derniere["corrected"]))
        if ident in temoins:
            stat["temoins"].append(cer(temoins[ident]["ref"], derniere["corrected"]))
        marques = drapeaux(derniere, seg, lexique)
        if marques:
            a_revoir.append({
                "segment_id": ident, "annotateur": annotateur,
                "drapeaux": " ; ".join(marques),
                "whisper": seg.get("t", ""), "correction": derniere["corrected"],
                "duree_ms": seg.get("d", 0), "ecoute_ms": derniere.get("ecoute_ms", 0),
                "lectures": derniere.get("lectures", 0), "rating": note,
            })

    largeur = max([10] + [len(a) for a in par_annotateur])
    print(f"\n  {'annotateur'.ljust(largeur)}  corrigés  ★ méd.  ★★★★★  1★   écoute   "
          f"reprises            témoins")
    print(f"  {'-' * largeur}  --------  ------  -----  --   ------   "
          f"-----------------   -----------------")
    for annotateur, s in sorted(par_annotateur.items()):
        note = mediane(s["notes"])
        print(f"  {annotateur.ljust(largeur)}  {s['corrections']:>8}  "
              f"{(f'{float(note):.1f}' if s['notes'] else '—'):>6}  "
              f"{s['validees']:>5}  {s['rebuts']:>2}   "
              f"{pourcent(mediane(s['ecoute'])):>6}   "
              f"{pourcent(mediane(s['reprises'])):>7} ({len(s['reprises']):>2} mesuré)   "
              f"{pourcent(mediane(s['temoins'])):>7} ({len(s['temoins']):>2} mesuré)")
    print("\n  ★ méd.   : confiance médiane déclarée ; ★★★★★ = textes qui font foi, 1★ = inexploitables")
    print("  écoute   : audio entendu rapporté à la durée (médiane) — 100 % = l'extrait entier,")
    print("             au-delà, il a été réécouté ; bien en deçà, il ne l'a pas été")
    print("  reprises : écart entre deux passages du même annotateur — plus c'est bas, plus c'est reproductible")
    print("  témoins  : écart à une transcription connue — la seule mesure absolue")
    if not any(s["temoins"] for s in par_annotateur.values()):
        print("\n  Aucun témoin traité : scripts/build_temoins.py en constitue depuis un CSV validé.")
    if not any(s["reprises"] for s in par_annotateur.values()):
        print("  Aucune reprise : le re-test se déclenche 14 jours après une correction (DELAI_RETEST).")

    print(f"\n  {len(a_revoir)} segment(s) à revoir")
    compte: dict[str, int] = {}
    for r in a_revoir:
        for m in r["drapeaux"].split(" ; "):
            compte[m] = compte.get(m, 0) + 1
    for m, n in sorted(compte.items(), key=lambda kv: -kv[1]):
        print(f"    {m:<34} {n:>5}")
    if args.csv and a_revoir:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(a_revoir[0]))
            w.writeheader()
            w.writerows(a_revoir)
        print(f"\n  détail : {args.csv}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
