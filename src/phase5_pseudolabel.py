"""
PHASE 5 — ITÉRATION PSEUDO-LABELING (optionnelle)

À ne lancer que si la Phase 4 laisse trop peu de segments à haute confiance, ou
si le WER estimé plafonne. Le gain vient de l'exposition du correcteur à de
vraies sorties Whisper plutôt qu'à des corruptions synthétiques.

Boucle :
  1. re-transcription n-best avec Whisper-ht
  2. rescoring des hypothèses par le LM gcf (interpolation acoustique/LM)
  3. correction ByT5 de chaque hypothèse
  4. filtre d'ACCORD : on ne garde que les segments où plusieurs hypothèses
     convergent après correction. C'est le garde-fou central — sans lui, le
     modèle réapprend ses propres erreurs et le pipeline dérive.
  5. fusion avec les paires synthétiques, ré-entraînement warm start (1-2 epochs,
     LR réduit), puis retour en Phase 4.

Le risque connu de cette phase est l'amplification de biais : à chaque tour, le
correcteur devient plus sûr de lui sur ce qu'il produit déjà. Trois protections
sont câblées : filtre d'accord, plafond `--max-rounds`, et conservation d'une
proportion minimale de paires synthétiques (`--synth-ratio`) qui ancre
l'entraînement sur du gcf attesté.

Usage :
    python -m src.phase5_pseudolabel transcribe --audio-dir ./audio --nbest 10
    python -m src.phase5_pseudolabel merge --agreement 0.85
    python -m src.phase5_pseudolabel retrain --epochs 2
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import CFG
from .io_utils import log, read_jsonl, write_json, write_jsonl
from .lm import load_lm
from .metrics import wer
from .normalize import normalize_surface

NBEST_PATH = CFG.model_dir.parent / "nbest_hypotheses.jsonl"
PSEUDO_PATH = CFG.model_dir.parent / "pseudo_pairs.csv"


# ---------------------------------------------------------------------------
# 1-2. Transcription n-best + rescoring
# ---------------------------------------------------------------------------

def cmd_transcribe(args) -> None:
    """
    Re-transcrit l'audio en n-best. Utilise faster-whisper si disponible
    (beaucoup plus rapide), sinon openai-whisper.
    """
    audio_files = _collect_audio(args)
    log.info("%d fichiers audio à transcrire", len(audio_files))
    lm = load_lm(CFG)

    model = _load_whisper(args.whisper_model, args.backend)
    rows = []
    for k, path in enumerate(audio_files):
        hyps = _transcribe_nbest(model, path, args.nbest, args.backend, args.language)
        for h in hyps:
            h["lm_logprob"] = lm.log10_prob(h["text"])
            h["score_fusion"] = (
                h["avg_logprob"] + args.lm_weight * h["lm_logprob"] / max(len(h["text"].split()), 1)
            )
        hyps.sort(key=lambda h: -h["score_fusion"])
        rows.append({"audio": str(path), "hypotheses": hyps})
        if (k + 1) % 50 == 0:
            log.info("  %d/%d", k + 1, len(audio_files))

    write_jsonl(args.out or NBEST_PATH, rows)


def _collect_audio(args) -> list[Path]:
    if args.audio_dir:
        d = Path(args.audio_dir)
        return sorted(p for p in d.rglob("*") if p.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"})
    # sinon : chemins présents dans dataset_corrected.jsonl
    paths = []
    for r in read_jsonl(CFG.dataset_corrected):
        if r.get("audio"):
            paths.append(Path(r["audio"]))
    return paths[:args.limit] if args.limit else paths


def _load_whisper(name: str, backend: str):
    if backend == "faster":
        from faster_whisper import WhisperModel
        return WhisperModel(name, device="auto", compute_type="float16")
    import whisper
    return whisper.load_model(name)


def _transcribe_nbest(model, path: Path, nbest: int, backend: str, language: str) -> list[dict]:
    """
    Whisper n'expose pas nativement une n-best list. On l'approxime par un
    échantillonnage à températures croissantes : c'est ce que fait le décodage
    de repli de Whisper lui-même, et cela suffit à mesurer l'accord.
    """
    temps = [0.0] + [round(0.2 * i, 2) for i in range(1, nbest)]
    hyps: list[dict] = []
    seen: set[str] = set()
    for t in temps[:nbest]:
        try:
            if backend == "faster":
                segs, _ = model.transcribe(str(path), language=language, temperature=t,
                                           beam_size=5 if t == 0 else 1)
                segs = list(segs)
                text = normalize_surface(" ".join(s.text for s in segs))
                lp = sum(s.avg_logprob for s in segs) / max(len(segs), 1) if segs else -5.0
            else:
                res = model.transcribe(str(path), language=language, temperature=t)
                text = normalize_surface(res.get("text", ""))
                segs = res.get("segments", [])
                lp = sum(s["avg_logprob"] for s in segs) / max(len(segs), 1) if segs else -5.0
        except Exception as exc:  # noqa: BLE001
            log.warning("Transcription échouée (%s, T=%.1f) : %s", path.name, t, exc)
            continue
        if text and text.lower() not in seen:
            seen.add(text.lower())
            hyps.append({"text": text, "temperature": t, "avg_logprob": float(lp)})
    return hyps


# ---------------------------------------------------------------------------
# 3-4. Correction des n-best + filtre d'accord
# ---------------------------------------------------------------------------

def cmd_merge(args) -> None:
    from .phase3_train import generate_batch, load_corrector

    model, tok = load_corrector(args.checkpoint)
    rows = list(read_jsonl(args.nbest_file or NBEST_PATH))
    log.info("%d segments n-best chargés", len(rows))

    kept, rejected = [], 0
    flat = [(i, h["text"]) for i, r in enumerate(rows) for h in r["hypotheses"]]
    preds = generate_batch(model, tok, [t for _, t in flat], batch_size=args.batch_size)

    grouped: dict[int, list[str]] = {}
    for (i, _), p in zip(flat, preds):
        grouped.setdefault(i, []).append(normalize_surface(p))

    for i, corrected in grouped.items():
        if len(corrected) < 2:
            rejected += 1
            continue
        best = corrected[0]                       # hypothèse la mieux rescorée
        agree = _agreement(best, corrected[1:])
        if agree < args.agreement:
            rejected += 1
            continue
        kept.append({
            "audio": rows[i]["audio"],
            "source_pseudo_ht": rows[i]["hypotheses"][0]["text"],
            "target_gcf": best,
            "accord": round(agree, 4),
            "n_hypotheses": len(corrected),
        })

    log.info("Pseudo-labels retenus : %d | rejetés pour désaccord : %d", len(kept), rejected)
    out = Path(args.out or PSEUDO_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source_pseudo_ht", "target_gcf", "accord", "audio"])
        for r in kept:
            w.writerow([r["source_pseudo_ht"], r["target_gcf"], r["accord"], r["audio"]])
    write_json(out.with_suffix(".stats.json"), {
        "retenus": len(kept), "rejetes": rejected,
        "seuil_accord": args.agreement,
        "taux_retention": round(len(kept) / max(len(rows), 1), 4),
    })


def _agreement(best: str, others: list[str]) -> float:
    """1 - WER moyen entre la meilleure hypothèse corrigée et les autres."""
    if not others:
        return 0.0
    return max(0.0, 1 - sum(wer(best, o) for o in others) / len(others))


# ---------------------------------------------------------------------------
# 5. Ré-entraînement warm start
# ---------------------------------------------------------------------------

def cmd_retrain(args) -> None:
    """Fusionne pseudo-paires + paires synthétiques, puis reprend l'entraînement."""
    import random

    with Path(args.pseudo or PSEUDO_PATH).open(encoding="utf-8") as fh:
        pseudo = list(csv.DictReader(fh))
    with Path(CFG.train_pairs).open(encoding="utf-8") as fh:
        synth = list(csv.DictReader(fh))
    rng = random.Random(CFG.seed)

    n_synth_min = int(len(pseudo) * args.synth_ratio)
    keep_synth = synth if len(synth) <= n_synth_min else rng.sample(synth, n_synth_min)
    merged = [{"source_pseudo_ht": r["source_pseudo_ht"], "target_gcf": r["target_gcf"]}
              for r in pseudo + keep_synth]
    rng.shuffle(merged)

    out = Path(CFG.train_pairs).with_name("train_pairs_round2.csv")
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source_pseudo_ht", "target_gcf"])
        w.writeheader()
        w.writerows(merged)
    log.info("%d paires fusionnées (%d pseudo + %d synthétiques) -> %s",
             len(merged), len(pseudo), len(keep_synth), out)

    from .phase3_train import build_parser as p3_parser
    from .phase3_train import train

    argv = [
        "--model", args.checkpoint or str(CFG.model_dir),   # warm start
        "--train", str(out),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
    ]
    metrics = train(p3_parser().parse_args(argv))
    print("\nTour de pseudo-labeling terminé. Relancer la Phase 4 pour mesurer le gain réel.")
    print(f"WER test après ré-entraînement : {metrics['wer_after']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 5 — pseudo-labeling itératif")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe")
    t.add_argument("--audio-dir", default=None)
    t.add_argument("--whisper-model", default="openai/whisper-small")
    t.add_argument("--backend", choices=["faster", "openai"], default="faster")
    t.add_argument("--language", default="ht")
    t.add_argument("--nbest", type=int, default=CFG.nbest)
    t.add_argument("--lm-weight", type=float, default=0.5)
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--out", default=None)
    t.set_defaults(func=cmd_transcribe)

    m = sub.add_parser("merge")
    m.add_argument("--nbest-file", default=None)
    m.add_argument("--checkpoint", default=None)
    m.add_argument("--agreement", type=float, default=CFG.pseudo_agreement_threshold)
    m.add_argument("--batch-size", type=int, default=32)
    m.add_argument("--out", default=None)
    m.set_defaults(func=cmd_merge)

    r = sub.add_parser("retrain")
    r.add_argument("--pseudo", default=None)
    r.add_argument("--checkpoint", default=None)
    r.add_argument("--epochs", type=float, default=CFG.pseudo_epochs)
    r.add_argument("--lr", type=float, default=CFG.lr / 3)
    r.add_argument("--synth-ratio", type=float, default=1.0,
                   help="paires synthétiques conservées par pseudo-paire (ancrage)")
    r.set_defaults(func=cmd_retrain)
    return p


if __name__ == "__main__":
    _a = build_parser().parse_args()
    _a.func(_a)
