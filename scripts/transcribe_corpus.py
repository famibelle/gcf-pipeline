#!/usr/bin/env python3
"""
Transcription Whisper du corpus audio POTOMITAN (première passe ht).

Produit un `metadata.jsonl` au format exact du dépôt Hugging Face —
`{"file_name": "part01/xxx.mp3", "transcription": "..."}` — de façon à
remplacer le fichier actuel sans rien casser en aval.

Deux propriétés comptent sur 40 000 fichiers :

- **Reprise.** Chaque ligne est écrite et synchronisée au fil de l'eau. Une
  interruption à la 30 000ᵉ ne coûte que le fichier en cours ; relancer la
  même commande repart où ça s'était arrêté.
- **Isolation des erreurs.** Un mp3 corrompu est journalisé et sauté, il
  n'arrête pas la passe.

Mesurer avant de lancer la nuit complète :

    python scripts/transcribe_corpus.py --root <dataset> --limit 100

Le débit affiché à la fin donne l'estimation réelle sur ta carte, ce qui
vaut mieux que n'importe quelle extrapolation théorique.

Note sur le coût : Whisper complète toute entrée à 30 secondes avant de
l'encoder. Des segments de 5 secondes paient donc six fois leur durée. Le
seul moyen de l'éviter serait de reconcaténer les segments en flux longs
puis de redécouper aux horodatages — beaucoup plus de plomberie, pour un
gain qui ne se justifie que si la mesure ci-dessus se révèle décevante.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def discover(root: Path, dirs: list[str] | None) -> list[Path]:
    """Tous les audios sous `root`, triés pour que la reprise soit déterministe."""
    if dirs:
        roots = [root / d for d in dirs]
    else:
        roots = sorted(p for p in root.iterdir() if p.is_dir())
    found: list[Path] = []
    for r in roots:
        if not r.is_dir():
            print(f"  (ignoré, introuvable : {r})", file=sys.stderr)
            continue
        found.extend(p for p in r.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    return sorted(found)


def key_of(path: Path, root: Path) -> str:
    """Clé au format du dépôt : chemin relatif en séparateurs POSIX."""
    return path.relative_to(root).as_posix()


def load_done(out: Path) -> dict[str, str]:
    """Lignes déjà écrites. Une ligne tronquée par une interruption est ignorée."""
    done: dict[str, str] = {}
    if not out.exists():
        return done
    # utf-8-sig et non utf-8 : le metadata.jsonl du dépôt porte un BOM, et en
    # utf-8 strict sa PREMIÈRE ligne échoue au parsage puis se fait avaler par
    # le `continue` ci-dessous — elle serait alors retranscrite en silence.
    with out.open(encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = row.get("file_name")
            if name:
                done[name] = row.get("transcription", "")
    return done


def human(seconds: float) -> str:
    seconds = int(seconds)
    h, m = divmod(seconds // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{seconds % 60:02d}s"


def main() -> int:
    p = argparse.ArgumentParser(description="Passe Whisper sur le corpus audio")
    p.add_argument("--root", required=True, help="racine contenant part01/, part02/, audio/…")
    p.add_argument("--out", default=None, help="jsonl de sortie (défaut : <root>/metadata.whisper.jsonl)")
    p.add_argument("--dirs", nargs="*", default=None, help="restreindre à ces sous-dossiers")
    p.add_argument("--model", default="large-v3")
    p.add_argument("--language", default="ht", help="ht = créole haïtien, la langue de la première passe")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="int8_float16")
    p.add_argument("--beam-size", type=int, default=5,
                   help="1 = glouton, nettement plus rapide et un peu moins bon")
    p.add_argument("--limit", type=int, default=None, help="s'arrêter après N fichiers (mesure)")
    p.add_argument("--redo", action="store_true", help="ignorer la reprise et tout retranscrire")
    p.add_argument("--retry-empty", action="store_true",
                   help="reprendre les entrées dont la transcription est vide "
                        "(après une panne d'installation, par exemple)")
    p.add_argument("--max-initial-failures", type=int, default=10,
                   help="arrêt si les N premiers fichiers échouent tous")
    p.add_argument("--preserve", default=None,
                   help="jsonl existant dont les transcriptions non vides sont "
                        "recopiées telles quelles et jamais recalculées "
                        "(typiquement le metadata.jsonl actuel du dépôt)")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Racine introuvable : {root}", file=sys.stderr)
        return 2
    out = Path(args.out).expanduser() if args.out else root / "metadata.whisper.jsonl"

    files = discover(root, args.dirs)
    if not files:
        print(f"Aucun audio trouvé sous {root}", file=sys.stderr)
        return 2

    done = {} if args.redo else load_done(out)

    # Transcriptions à conserver intactes : elles sont écrites dans la sortie si
    # elles n'y sont pas déjà, puis exclues du travail. La sortie devient ainsi un
    # remplaçant complet du metadata.jsonl, sans qu'aucune ligne existante ne soit
    # recalculée.
    preserved = 0
    if args.preserve:
        src = Path(args.preserve).expanduser()
        if not src.is_file():
            print(f"--preserve : fichier introuvable ({src})", file=sys.stderr)
            return 2
        keep = {k: v for k, v in load_done(src).items() if v.strip()}
        nouveaux = {k: v for k, v in keep.items() if k not in done}
        if nouveaux:
            with out.open("a", encoding="utf-8") as fh:
                for k, v in nouveaux.items():
                    fh.write(json.dumps({"file_name": k, "transcription": v},
                                        ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        done.update(keep)
        preserved = len(keep)

    if args.retry_empty:
        # Les lignes préservées restent hors du champ : elles ne sont pas vides.
        done = {k: v for k, v in done.items() if v.strip()}
    todo = [f for f in files if key_of(f, root) not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"racine        : {root}")
    print(f"sortie        : {out}")
    print(f"audios trouvés: {len(files):,}")
    print(f"déjà faits    : {len(done):,}")
    if preserved:
        print(f"  dont préservés : {preserved:,} (jamais recalculés)")
    print(f"à traiter     : {len(todo):,}")
    if not todo:
        print("Rien à faire.")
        return 0

    from faster_whisper import WhisperModel

    print(f"chargement de {args.model} sur {args.device} ({args.compute_type})…")
    t0 = time.time()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print(f"  chargé en {time.time() - t0:.0f}s")

    ok = failed = empty = 0
    last_error = ""
    audio_seconds = 0.0
    started = time.time()
    # Ouverture en ajout : la reprise ne réécrit jamais ce qui existe.
    with out.open("a", encoding="utf-8") as fh:
        for i, path in enumerate(todo, 1):
            name = key_of(path, root)
            try:
                segments, info = model.transcribe(
                    str(path),
                    language=args.language,
                    beam_size=args.beam_size,
                    # Segments indépendants : conditionner sur le précédent
                    # ferait dériver le texte d'un extrait à l'autre.
                    condition_on_previous_text=False,
                    vad_filter=False,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
                audio_seconds += info.duration
                ok += 1
                if not text:
                    empty += 1
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                print(f"  ÉCHEC {name} : {last_error}", file=sys.stderr)
                text = ""
                failed += 1

            # Un mp3 illisible se saute ; une installation cassée fait échouer
            # TOUT et doit s'arrêter là. Sans ce garde-fou, une bibliothèque CUDA
            # manquante produirait 40 000 lignes vides en plusieurs heures.
            if ok == 0 and failed >= args.max_initial_failures:
                fh.flush()
                os.fsync(fh.fileno())
                print(f"\nARRÊT : les {failed} premiers fichiers ont tous échoué.\n"
                      f"Dernière erreur — {last_error}\n"
                      f"Ce n'est pas un problème de données mais d'installation. "
                      f"Corriger, puis relancer avec --retry-empty pour reprendre "
                      f"les lignes vides déjà écrites.", file=sys.stderr)
                return 1

            fh.write(json.dumps({"file_name": name, "transcription": text},
                                ensure_ascii=False) + "\n")
            # Synchronisation périodique : une coupure ne perd qu'une poignée
            # de lignes, sans payer un fsync par fichier.
            if i % 25 == 0 or i == len(todo):
                fh.flush()
                os.fsync(fh.fileno())
                elapsed = time.time() - started
                rate = i / elapsed
                eta = (len(todo) - i) / rate if rate else 0
                print(f"  {i:>6,}/{len(todo):,}  {rate:5.2f} fichier/s  "
                      f"écoulé {human(elapsed)}  reste ~{human(eta)}", flush=True)

    elapsed = time.time() - started
    print()
    print(f"transcrits    : {ok:,}")
    print(f"dont vides    : {empty:,}")
    print(f"échecs        : {failed:,}")
    if ok and empty == ok:
        print("  ATTENTION : toutes les transcriptions sont vides. Vérifier le "
              "modèle et la langue avant de lancer la passe complète.", file=sys.stderr)
    print(f"durée calcul  : {human(elapsed)}")
    if audio_seconds:
        print(f"audio traité  : {human(audio_seconds)}")
        print(f"vitesse       : {audio_seconds / elapsed:.1f}× le temps réel")
        reste = len(files) - len(done) - len(todo)
        if reste > 0:
            par_fichier = elapsed / max(ok + failed, 1)
            print(f"→ estimation pour les {reste:,} restants : {human(reste * par_fichier)}")
    print(f"\nsortie : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
