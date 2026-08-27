"""Métriques d'évaluation : WER, CER, taux de correction. Aucune dépendance externe."""
from __future__ import annotations

from .normalize import normalize_surface, tokenize


def edit_distance(a: list[str], b: list[str]) -> int:
    """Distance de Levenshtein, mémoire O(min(|a|,|b|))."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str, lower: bool = True) -> float:
    ref = tokenize(normalize_surface(reference, lower=lower))
    hyp = tokenize(normalize_surface(hypothesis, lower=lower))
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str, lower: bool = True) -> float:
    ref = list(normalize_surface(reference, lower=lower))
    hyp = list(normalize_surface(hypothesis, lower=lower))
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def corpus_wer(references: list[str], hypotheses: list[str], lower: bool = True) -> float:
    """WER agrégé au niveau corpus (somme des erreurs / somme des mots),
    et non moyenne des WER par phrase : les phrases courtes ne dominent pas."""
    errs = words = 0
    for r, h in zip(references, hypotheses):
        rt = tokenize(normalize_surface(r, lower=lower))
        ht = tokenize(normalize_surface(h, lower=lower))
        errs += edit_distance(rt, ht)
        words += len(rt)
    return errs / max(words, 1)


def corpus_cer(references: list[str], hypotheses: list[str], lower: bool = True) -> float:
    errs = chars = 0
    for r, h in zip(references, hypotheses):
        rc = list(normalize_surface(r, lower=lower))
        hc = list(normalize_surface(h, lower=lower))
        errs += edit_distance(rc, hc)
        chars += len(rc)
    return errs / max(chars, 1)


def evaluate(sources: list[str], predictions: list[str], references: list[str]) -> dict:
    """
    Trois chiffres qui doivent toujours être lus ensemble :
      - wer_before : distance source(bruitée) <-> référence, la ligne de base
      - wer_after  : distance prédiction <-> référence, le résultat
      - wer_reduction : gain relatif. Négatif = le correcteur dégrade.
    """
    before = corpus_wer(references, sources)
    after = corpus_wer(references, predictions)
    exact = sum(1 for p, r in zip(predictions, references)
                if normalize_surface(p, lower=True) == normalize_surface(r, lower=True))
    return {
        "n": len(references),
        "wer_before": round(before, 4),
        "wer_after": round(after, 4),
        "wer_reduction_rel": round((before - after) / before, 4) if before else 0.0,
        "cer_before": round(corpus_cer(references, sources), 4),
        "cer_after": round(corpus_cer(references, predictions), 4),
        "exact_match": exact,
        "exact_match_rate": round(exact / max(len(references), 1), 4),
    }
