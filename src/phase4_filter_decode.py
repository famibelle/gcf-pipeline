"""
PHASE 4 — FILTRAGE, CORRECTION, RESCORING

Chaîne : scorer tout le corpus Whisper-ht -> filtrer -> corriger -> re-scorer ->
partitionner en `dataset_corrected.jsonl` (tout, avec scores) et
`dataset_high_confidence.jsonl` (le sous-ensemble exploitable pour l'ASR).

Deux points méritent d'être explicites, parce qu'ils décident de la qualité du
livrable final :

1. Le WER annoncé est ESTIMÉ, jamais mesuré, sur les données réelles : il n'y a
   pas de vérité terrain sur le corpus Whisper. Le champ s'appelle donc
   `wer_estime` et son calibrage vient de test_pairs, où la vérité existe. Cette
   extrapolation suppose que les corruptions synthétiques ressemblent aux erreurs
   réelles de Whisper — hypothèse à vérifier en re-annotant un échantillon du
   livrable (`--audit-sample`).

2. `avg_logprob` et `compression_ratio` viennent de Whisper. S'ils sont absents
   du dataset, `compression_ratio` est recalculé par zlib (même définition que
   Whisper) et le filtre logprob est neutralisé, ce qui est signalé.

Usage :
    python -m src.phase4_filter_decode --calibrate
    python -m src.phase4_filter_decode --audit-sample 200
"""
from __future__ import annotations

import argparse
import math
import zlib
from pathlib import Path

from .config import CFG
from .io_utils import (
    detect_text_column,
    load_hf_audio_no_decode,
    log,
    write_json,
    write_jsonl,
)
from .lm import calibrate_threshold, load_lm
from .metrics import wer
from .normalize import enforce_gerec2, gerec2_violations, normalize_surface, tokenize

LOGPROB_FIELDS = ["avg_logprob", "avg_log_prob", "logprob", "whisper_avg_logprob"]
RATIO_FIELDS = ["compression_ratio", "whisper_compression_ratio"]
NOSPEECH_FIELDS = ["no_speech_prob", "nospeech_prob"]


def compression_ratio(text: str) -> float:
    """Définition de Whisper : taille brute / taille compressée zlib."""
    b = text.encode("utf-8")
    if not b:
        return 0.0
    return len(b) / max(len(zlib.compress(b)), 1)


def _first(row: dict, fields: list[str]):
    for f in fields:
        v = row.get(f)
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_corpus(ds, text_col: str, lm, cfg) -> list[dict]:
    rows: list[dict] = []
    missing_logprob = 0
    for i in range(len(ds)):
        r = ds[i]
        text = normalize_surface(r.get(text_col) or "")
        if not text:
            continue
        lp = _first(r, LOGPROB_FIELDS)
        if lp is None:
            missing_logprob += 1
        cr = _first(r, RATIO_FIELDS)
        if cr is None:
            cr = compression_ratio(text)
        ppl = lm.perplexity(text)
        audio = r.get(cfg.asr_audio_column)
        rows.append({
            "id": r.get("id", f"seg_{i:07d}"),
            "dataset_index": i,
            "audio": audio.get("path") if isinstance(audio, dict) else audio,
            "duration": r.get("duration"),
            "whisper_ht": text,
            "avg_logprob": float(lp) if lp is not None else None,
            "compression_ratio": round(float(cr), 4),
            "no_speech_prob": _first(r, NOSPEECH_FIELDS),
            "perplexite_lm": round(float(ppl), 3) if math.isfinite(ppl) else None,
            "n_tokens": len(tokenize(text)),
        })
    if missing_logprob:
        log.warning("avg_logprob absent pour %d/%d segments : le filtre logprob "
                    "sera neutralisé pour eux (ils ne sont pas rejetés à tort).",
                    missing_logprob, len(rows))
    return rows


def apply_filters(rows: list[dict], cfg, ppl_threshold: float) -> list[dict]:
    for r in rows:
        reasons = []
        if r["avg_logprob"] is not None and r["avg_logprob"] < cfg.min_avg_logprob:
            reasons.append(f"logprob<{cfg.min_avg_logprob}")
        if r["compression_ratio"] > cfg.max_compression_ratio:
            reasons.append(f"compression>{cfg.max_compression_ratio}")
        if r["perplexite_lm"] is not None and r["perplexite_lm"] > ppl_threshold:
            reasons.append(f"ppl>{ppl_threshold:.0f}")
        if r["no_speech_prob"] is not None and r["no_speech_prob"] > 0.6:
            reasons.append("no_speech>0.6")
        if r["n_tokens"] < 2:
            reasons.append("trop_court")
        r["filtre_passe"] = not reasons
        r["motifs_rejet"] = reasons
    kept = [r for r in rows if r["filtre_passe"]]
    log.info("Filtrage : %d/%d segments conservés (%.1f%%)",
             len(kept), len(rows), 100 * len(kept) / max(len(rows), 1))
    return kept


# ---------------------------------------------------------------------------
# Confiance composite
# ---------------------------------------------------------------------------

def _sig(x: float, mid: float, scale: float) -> float:
    return 1 / (1 + math.exp(-(x - mid) / scale))


def confidence(row: dict, cfg, ppl_threshold: float) -> float:
    """
    Score dans [0,1] combinant : acoustique (logprob), dégénérescence
    (compression), plausibilité linguistique de la SORTIE corrigée (perplexité),
    et stabilité de la correction (ampleur de l'édition appliquée par ByT5 —
    une réécriture massive signale soit une entrée aberrante, soit une
    hallucination du correcteur ; les deux justifient de se méfier).
    """
    parts, weights = [], []

    if row.get("avg_logprob") is not None:
        parts.append(_sig(row["avg_logprob"], cfg.min_avg_logprob, 0.25)); weights.append(0.30)
    parts.append(1 - _sig(row["compression_ratio"], cfg.max_compression_ratio, 0.20)); weights.append(0.15)
    ppl = row.get("perplexite_corrigee") or row.get("perplexite_lm")
    if ppl:
        parts.append(1 - _sig(math.log(max(ppl, 1.01)), math.log(max(ppl_threshold, 1.01)), 0.35))
        weights.append(0.35)
    edit = row.get("taux_edition", 0.0)
    parts.append(1 - _sig(edit, 0.45, 0.12)); weights.append(0.20)

    if row.get("violations_gerec2"):
        parts.append(0.0); weights.append(0.10)

    return round(sum(p * w for p, w in zip(parts, weights)) / sum(weights), 4)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(args) -> dict:
    cfg = CFG
    lm = load_lm(cfg)

    ppl_threshold = cfg.max_perplexity
    if args.calibrate:
        from .phase2_synthetic import iter_gcf_sentences
        sample = [s for _, s in zip(range(args.calibrate_n), iter_gcf_sentences(cfg))]
        ppl_threshold = calibrate_threshold(lm, sample, quantile=args.calibrate_quantile)

    ds = load_hf_audio_no_decode(cfg.hf_asr_dataset, cfg.hf_asr_split, cfg.hf_token)
    col = detect_text_column(ds, cfg.asr_text_column)
    rows = score_corpus(ds, col, lm, cfg)
    if args.limit:
        rows = rows[:args.limit]
    kept = apply_filters(rows, cfg, ppl_threshold)

    # --- correction neuronale du sous-ensemble filtré ------------------------
    from .phase3_train import generate_batch, load_corrector
    model, tok = load_corrector(args.checkpoint)
    texts = [r["whisper_ht"] for r in kept]
    log.info("Correction de %d segments…", len(texts))
    preds = generate_batch(model, tok, texts, batch_size=args.batch_size, beams=args.beams)

    for r, p in zip(kept, preds):
        p = enforce_gerec2(normalize_surface(p), mode="fix")
        r["gcf_corrige"] = p
        r["taux_edition"] = round(wer(r["whisper_ht"], p), 4)
        r["perplexite_corrigee"] = round(lm.perplexity(p), 3)
        r["violations_gerec2"] = gerec2_violations(p)
        r["gain_perplexite"] = (
            round(r["perplexite_lm"] - r["perplexite_corrigee"], 3)
            if r["perplexite_lm"] is not None else None
        )

    # --- calibration du WER estimé sur test_pairs ---------------------------
    wer_map = calibrate_wer_estimator(model, tok, cfg, n=args.calibration_pairs)
    for r in kept:
        r["confiance"] = confidence(r, cfg, ppl_threshold)
        r["wer_estime"] = estimate_wer(r["confiance"], wer_map)

    by_id = {r["id"]: r for r in kept}
    full = [by_id.get(r["id"], r) for r in rows]
    write_jsonl(cfg.dataset_corrected, full)

    high = [r for r in kept if r["confiance"] >= args.high_conf
            and r["wer_estime"] <= args.max_wer]
    high_out = [{
        "id": r["id"], "audio": r["audio"], "duration": r.get("duration"),
        "text": r["gcf_corrige"], "whisper_ht": r["whisper_ht"],
        "confiance": r["confiance"], "wer_estime": r["wer_estime"],
        "perplexite": r["perplexite_corrigee"],
    } for r in high]
    write_jsonl(cfg.dataset_high_conf, high_out)

    stats = {
        "segments_total": len(rows),
        "segments_apres_filtrage": len(kept),
        "segments_haute_confiance": len(high),
        "taux_retention_global": round(len(high) / max(len(rows), 1), 4),
        "seuil_perplexite": round(ppl_threshold, 2),
        "seuil_confiance": args.high_conf,
        "wer_estime_moyen_haute_confiance": round(
            sum(r["wer_estime"] for r in high) / max(len(high), 1), 4),
        "duree_totale_heures": round(
            sum(r.get("duration") or 0 for r in high) / 3600, 2),
        "calibration_wer": wer_map,
        "motifs_rejet": _reason_counts(rows),
    }
    write_json(Path(cfg.metrics_path).with_name("phase4_stats.json"), stats)

    print("\n=== PHASE 4 ===")
    for k, v in stats.items():
        if k not in ("calibration_wer", "motifs_rejet"):
            print(f"  {k:38s}: {v}")
    print(f"  motifs de rejet : {stats['motifs_rejet']}")

    if args.audit_sample:
        _write_audit_sheet(high, args.audit_sample, cfg)
    return stats


def _reason_counts(rows: list[dict]) -> dict:
    from collections import Counter
    c = Counter()
    for r in rows:
        for m in r.get("motifs_rejet", []):
            c[m] += 1
    return dict(c.most_common())


# ---------------------------------------------------------------------------
# Estimation du WER
# ---------------------------------------------------------------------------

def calibrate_wer_estimator(model, tok, cfg, n: int = 2000) -> dict:
    """
    Sur test_pairs (vérité terrain disponible), on mesure le WER réel par tranche
    de confiance. Ce tableau sert ensuite à attribuer un WER estimé aux segments
    réels. C'est une extrapolation, pas une mesure — d'où le nom des champs.
    """
    from .phase3_train import generate_batch, load_pairs_csv

    src, ref = load_pairs_csv(Path(cfg.test_pairs))
    src, ref = src[:n], ref[:n]
    if not src:
        log.warning("test_pairs vide : WER estimé indisponible.")
        return {}
    preds = generate_batch(model, tok, src, batch_size=32)
    lm = load_lm(cfg)

    buckets: dict[str, list[float]] = {}
    for s, p, r in zip(src, preds, ref):
        row = {
            "avg_logprob": None,
            "compression_ratio": compression_ratio(s),
            "perplexite_corrigee": lm.perplexity(p),
            "taux_edition": wer(s, p),
            "violations_gerec2": gerec2_violations(p),
        }
        c = confidence(row, cfg, cfg.max_perplexity)
        key = f"{int(c * 10) / 10:.1f}"
        buckets.setdefault(key, []).append(wer(r, p))

    table = {k: round(sum(v) / len(v), 4) for k, v in sorted(buckets.items())}
    log.info("Table de calibration WER par tranche de confiance : %s", table)
    return table


def estimate_wer(conf: float, table: dict) -> float:
    if not table:
        return round(max(0.0, 0.5 * (1 - conf)), 4)  # heuristique de secours
    key = f"{int(conf * 10) / 10:.1f}"
    if key in table:
        return table[key]
    keys = sorted(table, key=float)
    nearest = min(keys, key=lambda k: abs(float(k) - conf))
    return table[nearest]


def _write_audit_sheet(high: list[dict], n: int, cfg) -> None:
    """Échantillon à ré-annoter humainement : seul moyen de valider le WER estimé."""
    import csv
    import random

    rng = random.Random(cfg.seed)
    sample = rng.sample(high, min(n, len(high)))
    path = Path(cfg.dataset_high_conf).with_name("audit_humain.csv")
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "audio", "whisper_ht", "gcf_corrige",
                    "confiance", "wer_estime", "verite_terrain", "verdict"])
        for r in sample:
            w.writerow([r["id"], r["audio"], r["whisper_ht"], r["gcf_corrige"],
                        r["confiance"], r["wer_estime"], "", ""])
    log.info("Feuille d'audit humain (%d segments) -> %s", len(sample), path)
    print(f"\nAudit : remplir `verite_terrain` dans {path} pour confirmer le WER estimé.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 4 — filtrage et décodage")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--beams", type=int, default=4)
    p.add_argument("--high-conf", type=float, default=CFG.high_conf_threshold)
    p.add_argument("--max-wer", type=float, default=0.05,
                   help="WER estimé maximal pour dataset_high_confidence")
    p.add_argument("--calibrate", action="store_true",
                   help="calibrer le seuil de perplexité sur du gcf propre")
    p.add_argument("--calibrate-n", type=int, default=5000)
    p.add_argument("--calibrate-quantile", type=float, default=0.95)
    p.add_argument("--calibration-pairs", type=int, default=2000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--audit-sample", type=int, default=0)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
