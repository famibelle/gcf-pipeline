"""
PHASE 1 — DIAGNOSTIC

Trois sous-commandes :

  sample : extrait N segments aléatoires du corpus Whisper-ht et écrit une feuille
           de correction CSV (colonne `corrected` à remplir à la main). Une
           pré-correction par les règles amorces est proposée pour gagner du temps :
           l'annotateur corrige une proposition au lieu de partir de zéro.

  mine   : lit la feuille corrigée, aligne mot-à-mot, fouille les substitutions
           systématiques, écrit substitution_rules_ht2gcf.json + residual_cases.json.

  audit  : rejoue les règles sur les paires annotées et rapporte la couverture.

Usage :
    python -m src.phase1_diagnostic sample --n 300
    # ... correction humaine de data/diagnostic_to_correct.csv ...
    python -m src.phase1_diagnostic mine  --min-support 5 --min-confidence 0.6
    python -m src.phase1_diagnostic audit
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from .config import CFG
from .io_utils import detect_text_column, load_hf_audio_no_decode, log, write_json
from .normalize import gerec2_violations, normalize_surface, tokenize
from .rules import RuleSet, merge_rulesets, mine_rules

SEED_RULES = Path(__file__).resolve().parent.parent / "data" / "seed_rules_ht2gcf.json"


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

def cmd_sample(args) -> None:
    ds = load_hf_audio_no_decode(CFG.hf_asr_dataset, CFG.hf_asr_split, CFG.hf_token)
    col = detect_text_column(ds, CFG.asr_text_column)
    log.info("Colonne texte : %s | %d segments disponibles", col, len(ds))

    rng = random.Random(args.seed)
    # On échantillonne parmi les segments de longueur exploitable : un segment de
    # 3 mots ne révèle aucune substitution systématique.
    eligible = [i for i, t in enumerate(ds[col]) if t and len(tokenize(t)) >= args.min_tokens]
    log.info("%d segments éligibles (>= %d tokens)", len(eligible), args.min_tokens)
    if len(eligible) < args.n:
        log.warning("Moins de segments éligibles que demandé, on prend tout.")
    idx = rng.sample(eligible, min(args.n, len(eligible)))

    seed_rs = RuleSet.load(SEED_RULES) if SEED_RULES.exists() else RuleSet()

    out = Path(args.out or CFG.diagnostic_raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["segment_id", "dataset_index", "audio_ref",
                    "whisper_ht", "pre_correction", "corrected", "notes"])
        for k, i in enumerate(idx):
            row = ds[i]
            raw = normalize_surface(row[col])
            audio_ref = ""
            av = row.get(CFG.asr_audio_column) if isinstance(row, dict) else None
            if isinstance(av, dict):
                audio_ref = av.get("path") or ""
            pre = seed_rs.apply(raw) if seed_rs.rules else ""
            w.writerow([f"diag_{k:04d}", i, audio_ref, raw, pre, "", ""])

    log.info("Feuille de correction -> %s", out)
    print(
        "\nÀ FAIRE MAINTENANT (travail humain, non automatisable) :\n"
        f"  1. Ouvrir {out}\n"
        "  2. Pour chaque ligne, écouter l'audio si disponible et écrire la version\n"
        "     gcf correcte dans la colonne `corrected` (partir de `pre_correction`).\n"
        "  3. Laisser `corrected` vide pour ignorer une ligne (audio inaudible, hors langue).\n"
        "  4. Noter dans `notes` tout cas ambigu : ce sont eux qui feront les règles contextuelles.\n"
        f"  5. Enregistrer sous {CFG.diagnostic_done} puis lancer `mine`.\n"
    )


# ---------------------------------------------------------------------------
# mine
# ---------------------------------------------------------------------------

def _load_corrected(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    skipped = 0
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            src = normalize_surface(row.get("whisper_ht", ""))
            tgt = normalize_surface(row.get("corrected", ""))
            if not src or not tgt:
                skipped += 1
                continue
            pairs.append((src, tgt))
    log.info("%d paires annotées chargées (%d lignes ignorées)", len(pairs), skipped)
    return pairs


def cmd_mine(args) -> None:
    path = Path(args.input or CFG.diagnostic_done)
    if not path.exists():
        raise SystemExit(
            f"{path} introuvable. Lancer d'abord `sample`, corriger la feuille, "
            f"puis l'enregistrer sous ce nom."
        )
    pairs = _load_corrected(path)

    identical = sum(1 for s, t in pairs
                    if normalize_surface(s, lower=True) == normalize_surface(t, lower=True))
    log.info("Segments identiques après correction : %d", identical)
    if len(pairs) < 50:
        log.warning(
            "Seulement %d paires annotées. Le cahier des charges demande un minimum "
            "de matière pour que les fréquences soient interprétables ; en dessous de ~150 "
            "paires les seuils de support sont peu fiables.", len(pairs)
        )

    rs, residuals = mine_rules(
        pairs,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        max_ngram=args.max_ngram,
        align_method=args.align,
    )
    log.info("%d règles retenues sur %d substitutions candidates",
             len(rs.rules), rs.meta["n_candidate_substitutions"])

    if args.merge_seed and SEED_RULES.exists():
        seed_rs = RuleSet.load(SEED_RULES)
        # Les amorces confirmées par les données sont déjà dans `rs` avec un support
        # réel ; on n'ajoute que celles jamais observées, marquées comme non validées.
        observed = {r.src for r in rs.rules}
        unconfirmed = [r for r in seed_rs.rules if r.src not in observed]
        for r in unconfirmed:
            r.status = "seed_unconfirmed"
        rs = merge_rulesets(rs, RuleSet(unconfirmed), prefer="mined")
        log.info("%d amorces non confirmées par les données conservées (statut seed_unconfirmed)",
                 len(unconfirmed))

    cov = rs.coverage(pairs)
    rs.meta.update({
        "identical_segments": identical,
        "coverage_on_annotated": cov,
        "gerec2_violations_in_targets": _violation_report([t for _, t in pairs]),
    })
    rs.save(args.out or CFG.rules_path)

    # Résidus : regroupés et triés, c'est le §5 du livrable de phase 1.
    res_counter = Counter((r["src"], r["tgt"], r["op"]) for r in residuals)
    write_json(CFG.residuals_path, {
        "n_residual_blocks": len(residuals),
        "explanation": (
            "Divergences ht/gcf observées mais NON converties en règle : support ou "
            "confiance sous le seuil, insertion pure (rien à substituer), ou "
            "n-gram trop long. À traiter soit par extension des règles contextuelles, "
            "soit — le plus souvent — par le correcteur neuronal de la Phase 3."
        ),
        "top_unexplained": [
            {"src": s, "tgt": t, "op": op, "count": c}
            for (s, t, op), c in res_counter.most_common(200)
        ],
        "samples": residuals[:500],
    })

    print(f"\nCouverture des règles seules sur les paires annotées : "
          f"{cov['exact_match']}/{cov['n']} segments exactement reconstruits "
          f"({cov['exact_match_rate']:.1%}).")
    print("Le reste est précisément ce que le correcteur ByT5 doit apprendre.\n")


def _violation_report(texts: list[str]) -> dict:
    viol = Counter()
    for t in texts:
        for c in gerec2_violations(t):
            viol[c] += 1
    return dict(viol)


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def cmd_audit(args) -> None:
    rs = RuleSet.load(args.rules or CFG.rules_path)
    pairs = _load_corrected(Path(args.input or CFG.diagnostic_done))
    cov = rs.coverage(pairs)
    print(f"Règles : {len(rs.rules)} | Paires : {cov['n']} | "
          f"Reconstruction exacte : {cov['exact_match_rate']:.1%}")
    shown = 0
    for s, t in pairs:
        pred = rs.apply(s)
        if normalize_surface(pred, lower=True) != normalize_surface(t, lower=True):
            print(f"\n  ht    : {s}\n  règles: {pred}\n  cible : {t}")
            shown += 1
            if shown >= args.show:
                break


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1 — diagnostic des substitutions ht->gcf")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="extraire N segments à corriger")
    s.add_argument("--n", type=int, default=CFG.n_diagnostic_segments)
    s.add_argument("--seed", type=int, default=CFG.seed)
    s.add_argument("--min-tokens", type=int, default=5)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_sample)

    m = sub.add_parser("mine", help="fouiller les règles depuis la feuille corrigée")
    m.add_argument("--input", default=None)
    m.add_argument("--out", default=None)
    m.add_argument("--min-support", type=int, default=CFG.min_rule_support)
    m.add_argument("--min-confidence", type=float, default=CFG.min_rule_confidence)
    m.add_argument("--max-ngram", type=int, default=CFG.max_ngram)
    m.add_argument("--align", choices=["levenshtein", "simalign"], default="levenshtein")
    m.add_argument("--merge-seed", action="store_true", default=True)
    m.add_argument("--no-merge-seed", dest="merge_seed", action="store_false")
    m.set_defaults(func=cmd_mine)

    a = sub.add_parser("audit", help="mesurer la couverture des règles")
    a.add_argument("--rules", default=None)
    a.add_argument("--input", default=None)
    a.add_argument("--show", type=int, default=15)
    a.set_defaults(func=cmd_audit)
    return p


if __name__ == "__main__":
    _args = build_parser().parse_args()
    _args.func(_args)
