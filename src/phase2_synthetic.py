"""
PHASE 2 — PAIRES D'ENTRAÎNEMENT SYNTHÉTIQUES

Principe : on part du gcf propre (PawolKreyol-gfc) et on le *dégrade* vers un
pseudo-Whisper-ht en appliquant les règles à l'envers. Le correcteur apprend
donc à remonter la pente, exactement la tâche demandée en Phase 4.

Trois précautions structurent le module :

1. Corruption STOCHASTIQUE (`--corruption-prob`, défaut 0.85). Si l'on appliquait
   les règles de façon déterministe, le correcteur apprendrait une bijection et
   s'effondrerait dès qu'un segment réel de Whisper échappe aux règles.

2. Bruit non-réglementaire (`--noise-prob`) : chutes d'accents, agglutination,
   segmentation erronée, hésitations. Whisper produit ce genre d'erreurs, les
   règles ne les couvrent pas, le correcteur doit y être exposé.

3. Split par EMPREINTE du texte cible, pas au hasard. PawolKreyol contient des
   phrases très proches les unes des autres ; un split aléatoire ferait fuiter
   du train vers le test et gonflerait artificiellement les scores.

Usage :
    python -m src.phase2_synthetic --n 80000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import unicodedata
from pathlib import Path

from .config import CFG
from .io_utils import detect_text_column, load_hf, log, write_json
from .normalize import normalize_surface, strip_accents, tokenize
from .rules import RuleSet

# ---------------------------------------------------------------------------
# Bruit « façon Whisper »
# ---------------------------------------------------------------------------

_FILLERS = ["euh", "hm", "ah", "eh", "bon"]


def add_asr_noise(text: str, p: float, rng: random.Random) -> str:
    """Applique un bruit léger et plausible. Chaque opération est indépendante."""
    if p <= 0:
        return text
    tokens = tokenize(text)
    if not tokens:
        return text
    out: list[str] = []
    for tok in tokens:
        r = rng.random()
        if r < p * 0.35:
            out.append(strip_accents(tok))                    # perte d'accent
        elif r < p * 0.50 and len(tok) > 3:
            i = rng.randrange(1, len(tok) - 1)
            out.append(tok[:i] + tok[i + 1:])                 # élision de caractère
        elif r < p * 0.62 and len(tok) > 3:
            i = rng.randrange(1, len(tok) - 1)
            out.append(tok[:i] + tok[i] + tok[i:])            # gémination
        elif r < p * 0.72:
            out.append(tok.replace("è", "e").replace("ò", "o"))
        else:
            out.append(tok)
        if rng.random() < p * 0.06:
            out.append(rng.choice(_FILLERS))                  # hésitation insérée

    # agglutination / segmentation erronée
    if len(out) > 3 and rng.random() < p * 0.25:
        i = rng.randrange(len(out) - 1)
        out[i:i + 2] = [out[i] + out[i + 1]]
    if rng.random() < p * 0.20:
        i = rng.randrange(len(out))
        w = out[i]
        if len(w) > 4:
            j = rng.randrange(2, len(w) - 1)
            out[i:i + 1] = [w[:j], w[j:]]

    s = " ".join(out)
    if rng.random() < p * 0.4:
        s = re.sub(r"[,;:]", "", s)                           # ponctuation perdue
    if rng.random() < p * 0.3:
        s = s.lower()
    return normalize_surface(s)


# ---------------------------------------------------------------------------
# Corpus cible
# ---------------------------------------------------------------------------

def iter_gcf_sentences(cfg, limit: int | None = None):
    ds = load_hf(cfg.hf_text_dataset, cfg.hf_text_split, cfg.hf_token)
    col = detect_text_column(ds, cfg.text_column)
    log.info("Colonne texte gcf : %s | %d lignes", col, len(ds))
    seen: set[str] = set()
    n = 0
    for raw in ds[col]:
        if not raw:
            continue
        for sent in _split_sentences(raw):
            sent = normalize_surface(sent)
            if not (cfg.min_chars <= len(sent) <= cfg.max_chars):
                continue
            if len(tokenize(sent)) < 3:
                continue
            key = _fingerprint(sent)
            if key in seen:
                continue
            seen.add(key)
            yield sent
            n += 1
            if limit and n >= limit:
                return


_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text.strip())
    return [p for p in parts if p.strip()]


def _fingerprint(text: str) -> str:
    """Empreinte insensible à la casse, aux accents et à la ponctuation :
    deux variantes d'une même phrase tombent dans le même split."""
    t = strip_accents(text.lower())
    t = re.sub(r"[^\w\s]", "", unicodedata.normalize("NFC", t))
    t = re.sub(r"\s+", " ", t).strip()
    return hashlib.blake2b(t.encode("utf-8"), digest_size=12).hexdigest()


def _split_of(fp: str, ratios: tuple) -> str:
    """Assigne un split de façon déterministe à partir de l'empreinte."""
    bucket = int(fp[:8], 16) / 0xFFFFFFFF
    train, val, _ = ratios
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Génération
# ---------------------------------------------------------------------------

def generate(args) -> dict:
    cfg = CFG
    rs = RuleSet.load(args.rules or cfg.rules_path)
    if not rs.rules:
        raise SystemExit("Aucune règle chargée : exécuter la Phase 1 d'abord.")
    log.info("%d règles chargées (inversion gcf -> pseudo-ht)", len(rs.rules))

    rng = random.Random(args.seed)
    buckets: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    n_unchanged = 0
    n_total = 0

    for sent in iter_gcf_sentences(cfg, limit=args.scan_limit):
        n_total += 1
        # variantes multiples par phrase : même cible, corruptions différentes.
        for v in range(args.variants_per_sentence):
            noisy = rs.apply_inverse(sent, prob=args.corruption_prob, rng=rng)
            noisy = add_asr_noise(noisy, args.noise_prob, rng)
            if normalize_surface(noisy, lower=True) == normalize_surface(sent, lower=True):
                n_unchanged += 1
                # On conserve tout de même une part de paires identité : le
                # correcteur doit apprendre à NE PAS toucher au gcf déjà correct.
                if rng.random() > args.identity_keep_rate:
                    continue
            split = _split_of(_fingerprint(sent), cfg.split_ratios)
            buckets[split].append((noisy, sent))
        if sum(len(v) for v in buckets.values()) >= args.n:
            break

    stats = {
        "phrases_gcf_uniques_parcourues": n_total,
        "variantes_par_phrase": args.variants_per_sentence,
        "paires_generees": {k: len(v) for k, v in buckets.items()},
        "total": sum(len(v) for v in buckets.values()),
        "paires_identite_rencontrees": n_unchanged,
        "corruption_prob": args.corruption_prob,
        "noise_prob": args.noise_prob,
        "n_rules": len(rs.rules),
        "split_par": "empreinte du texte cible (anti-fuite)",
    }

    for split, path in (("train", cfg.train_pairs), ("val", cfg.val_pairs), ("test", cfg.test_pairs)):
        _write_pairs(Path(args.outdir) / path.name if args.outdir else path, buckets[split])

    write_json(cfg.model_dir.parent / "phase2_stats.json", stats)
    log.info("Statistiques : %s", stats["paires_generees"])
    if stats["total"] < 50_000:
        log.warning(
            "Moins de 50k paires : augmenter --variants-per-sentence ou vérifier "
            "la taille de PawolKreyol-gfc après déduplication."
        )
    return stats


def _write_pairs(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source_pseudo_ht", "target_gcf"])
        w.writerows(pairs)
    log.info("%d paires -> %s", len(pairs), path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 2 — paires synthétiques pseudo-ht -> gcf")
    p.add_argument("--n", type=int, default=CFG.n_synthetic_pairs)
    p.add_argument("--rules", default=None)
    p.add_argument("--seed", type=int, default=CFG.seed)
    p.add_argument("--corruption-prob", type=float, default=CFG.corruption_prob)
    p.add_argument("--noise-prob", type=float, default=CFG.noise_prob)
    p.add_argument("--variants-per-sentence", type=int, default=2)
    p.add_argument("--identity-keep-rate", type=float, default=0.05,
                   help="part des paires source==cible conservées")
    p.add_argument("--scan-limit", type=int, default=None)
    p.add_argument("--outdir", default=None)
    return p


if __name__ == "__main__":
    generate(build_parser().parse_args())
