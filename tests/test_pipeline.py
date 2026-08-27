"""
Tests de la logique pure : normalisation, alignement, fouille de règles,
métriques, LM de repli, découpage des splits.

Volontairement sans torch ni transformers : ces couches exigent un GPU et un
téléchargement de modèle, ce qui n'a pas sa place dans une CI. Ce qui est testé
ici est justement la partie où une régression silencieuse coûterait le plus cher
— une erreur d'alignement ne lève aucune exception, elle produit simplement des
règles fausses que personne ne remarque avant l'entraînement.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lm import BackoffCharLM
from src.metrics import cer, corpus_wer, edit_distance, evaluate, wer
from src.normalize import (
    enforce_gerec2,
    gerec2_violations,
    normalize_surface,
    strip_accents,
    tokenize,
)
from src.phase2_synthetic import _fingerprint, _split_of, add_asr_noise
from src.rules import (
    Rule,
    RuleSet,
    align_words,
    decompose_block,
    mine_rules,
)

PAIRS = [
    ("mwen ap manje diri a", "an ka manjé diri la"),
    ("li ap pale ak mwen", "i ka palé é mwen"),
    ("nou te gen anpil bagay", "nou té ni onlo biten"),
    ("yo ap vini nan kay la", "yo ka vini adan kaz la"),
    ("mwen te ale nan mache a", "an té alé adan maché la"),
]


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

def test_normalize_surface_preserve_accents():
    """La normalisation de surface ne doit jamais toucher aux accents créoles."""
    assert normalize_surface("  an   ka  manjé ") == "an ka manjé"
    assert "é" in normalize_surface("manjé")
    assert normalize_surface("mwen \u2019ka\u2019 alé").count("'") == 2


def test_normalize_removes_space_before_punct():
    assert normalize_surface("an ka alé , wi !") == "an ka alé, wi!"


def test_tokenize_keeps_apostrophe():
    assert tokenize("k'ay alé") == ["k'ay", "alé"]
    assert tokenize("alé, wi") == ["alé", ",", "wi"]


def test_strip_accents():
    assert strip_accents("manjé èvè ò") == "manje eve o"


@pytest.mark.parametrize("src,expected_absent", [
    ("çé une question", "ç"),
    ("qui coûte", "q"),
    ("un exemple", "x"),
])
def test_enforce_gerec2_removes_foreign_graphemes(src, expected_absent):
    assert expected_absent not in enforce_gerec2(src, mode="fix")


def test_gerec2_violations_detects_and_clean_text_passes():
    assert gerec2_violations("çela coûte") != []
    assert gerec2_violations("an ka manjé adan kaz la") == []


def test_enforce_gerec2_strict_raises_on_unfixable():
    """`fix` corrige ce qui est déterministe (ç, qu, ï...) ; `strict` signale ce
    qui ne l'est pas. Le « u » isolé n'existe pas en GEREC-2 hors du digraphe
    « ou », mais le corriger demanderait de deviner le mot : on le signale."""
    assert enforce_gerec2("naïve", mode="fix") == "naive"
    with pytest.raises(ValueError):
        enforce_gerec2("une kestion", mode="strict")


# ---------------------------------------------------------------------------
# alignement
# ---------------------------------------------------------------------------

def test_align_words_marks_identical_tokens_equal():
    blocks = align_words(tokenize("yo ap vini la"), tokenize("yo ka vini la"))
    ops = [(b.op, b.src, b.tgt) for b in blocks]
    assert ("equal", ["yo"], ["yo"]) in ops
    assert ("replace", ["ap"], ["ka"]) in ops


def test_decompose_block_splits_multiword_replacement():
    """Régression : sans décomposition, « mwen ap manje » -> « an ka manjé »
    reste un bloc unique et les trois règles réelles sont perdues."""
    pairs = decompose_block(["mwen", "ap", "manje"], ["an", "ka", "manjé"])
    assert (["mwen"], ["an"]) in pairs
    assert (["ap"], ["ka"]) in pairs
    assert (["manje"], ["manjé"]) in pairs


def test_decompose_block_handles_length_mismatch():
    pairs = decompose_block(["tap"], ["té", "ka"])
    assert sum(len(t) for _, t in pairs) == 2


# ---------------------------------------------------------------------------
# fouille de règles
# ---------------------------------------------------------------------------

@pytest.fixture
def mined():
    rs, residuals = mine_rules(PAIRS * 8, min_support=4, min_confidence=0.5, max_ngram=2)
    return rs, residuals


def test_mine_rules_finds_core_substitutions(mined):
    rs, _ = mined
    found = {(r.src, r.tgt) for r in rs.rules}
    for expected in [("ap", "ka"), ("te", "té"), ("nan", "adan"), ("li", "i")]:
        assert expected in found, f"règle attendue manquante : {expected}"


def test_mined_rules_have_plausible_confidence(mined):
    rs, _ = mined
    for r in rs.rules:
        assert 0.0 < r.confidence <= 1.0
        assert r.support <= r.src_total


def test_ruleset_roundtrip(tmp_path, mined):
    rs, _ = mined
    p = tmp_path / "rules.json"
    rs.save(p)
    back = RuleSet.load(p)
    assert len(back.rules) == len(rs.rules)
    assert back.apply("mwen ap manje") == rs.apply("mwen ap manje")


def test_apply_is_deterministic_at_prob_one(mined):
    rs, _ = mined
    out = {rs.apply("li ap vini nan kay la") for _ in range(10)}
    assert len(out) == 1


def test_longer_ngram_wins_over_unigram():
    rs = RuleSet([
        Rule(src="ap", tgt="ka", support=10, src_total=10, confidence=1.0, ngram=1),
        Rule(src="t ap", tgt="té ka", support=8, src_total=8, confidence=1.0, ngram=2),
    ])
    assert rs.apply("li t ap vini") == "li té ka vini"


def test_context_constraint_is_respected():
    rs = RuleSet([
        Rule(src="mwen", tgt="an", support=5, src_total=5, confidence=1.0,
             ngram=1, right_context="ka"),
    ])
    assert rs.apply("mwen ka alé") == "an ka alé"
    assert rs.apply("kaz an mwen") == "kaz an mwen"  # pas de contexte -> intact


def test_case_is_preserved():
    rs = RuleSet([Rule(src="ap", tgt="ka", support=5, src_total=5, confidence=1.0, ngram=1)])
    assert rs.apply("Ap vini") == "Ka vini"


def test_inverse_then_forward_recovers_most_of_the_sentence(mined):
    rs, _ = mined
    rng = random.Random(0)
    gcf = "an ka manjé adan kaz la"
    pseudo = rs.apply_inverse(gcf, prob=1.0, rng=rng)
    assert pseudo != gcf, "la corruption inverse n'a rien changé"
    assert wer(gcf, rs.apply(pseudo)) < wer(gcf, pseudo)


def test_coverage_reports_reasonable_rate(mined):
    rs, _ = mined
    cov = rs.coverage(PAIRS)
    assert 0.0 <= cov["exact_match_rate"] <= 1.0
    assert cov["n"] == len(PAIRS)


def test_residuals_are_recorded():
    _, residuals = mine_rules(PAIRS, min_support=100, min_confidence=0.99, max_ngram=2)
    assert residuals, "avec un seuil inatteignable, tout doit finir en résidu"


# ---------------------------------------------------------------------------
# métriques
# ---------------------------------------------------------------------------

def test_edit_distance_basics():
    assert edit_distance(list("kaz"), list("kaz")) == 0
    assert edit_distance(list("kay"), list("kaz")) == 1


def test_wer_identical_is_zero_and_bounds():
    assert wer("an ka manjé", "an ka manjé") == 0.0
    assert wer("an ka manjé", "") == 1.0
    assert cer("an", "an") == 0.0


def test_wer_ignores_case_by_default():
    assert wer("An Ka Manjé", "an ka manjé") == 0.0


def test_corpus_wer_is_length_weighted():
    """Une phrase longue doit peser plus qu'une courte : c'est la différence
    avec une moyenne de WER par phrase."""
    refs = ["a b c d e f g h", "x"]
    hyps = ["a b c d e f g h", "y"]
    assert corpus_wer(refs, hyps) == pytest.approx(1 / 9)


def test_evaluate_detects_improvement():
    m = evaluate(["mwen ap manje"], ["an ka manjé"], ["an ka manjé"])
    assert m["wer_after"] == 0.0
    assert m["wer_reduction_rel"] == 1.0
    assert m["exact_match"] == 1


def test_evaluate_flags_degradation():
    m = evaluate(["an ka manjé"], ["zzz zzz zzz"], ["an ka manjé"])
    assert m["wer_after"] > m["wer_before"]


# ---------------------------------------------------------------------------
# LM de repli
# ---------------------------------------------------------------------------

def test_charlm_prefers_seen_language():
    lm = BackoffCharLM(order=4).fit([t for _, t in PAIRS] * 5)
    assert lm.perplexity("an ka manjé diri la") < lm.perplexity("mwen ap manje diri a")


def test_charlm_roundtrip(tmp_path):
    lm = BackoffCharLM(order=3).fit([t for _, t in PAIRS])
    p = tmp_path / "lm.pkl"
    lm.save(p)
    back = BackoffCharLM.load(p)
    assert back.perplexity("an ka manjé") == pytest.approx(lm.perplexity("an ka manjé"))


# ---------------------------------------------------------------------------
# phase 2
# ---------------------------------------------------------------------------

def test_fingerprint_is_insensitive_to_accents_and_punctuation():
    assert _fingerprint("An ka manjé, wi!") == _fingerprint("an ka manje wi")


def test_split_assignment_is_stable_and_covers_all_buckets():
    fps = [_fingerprint(f"fraz nimewo {i}") for i in range(400)]
    splits = [_split_of(f, (0.8, 0.1, 0.1)) for f in fps]
    assert set(splits) == {"train", "val", "test"}
    assert splits == [_split_of(f, (0.8, 0.1, 0.1)) for f in fps]  # déterministe
    assert 0.7 < splits.count("train") / len(splits) < 0.9


def test_no_leakage_between_splits():
    fps = {_fingerprint(f"fraz {i}") for i in range(500)}
    buckets = {s: set() for s in ("train", "val", "test")}
    for f in fps:
        buckets[_split_of(f, (0.8, 0.1, 0.1))].add(f)
    assert not (buckets["train"] & buckets["test"])
    assert not (buckets["train"] & buckets["val"])


def test_asr_noise_perturbs_but_stays_readable():
    rng = random.Random(3)
    src = "an ka manjé adan kaz la èvè fanmi an mwen"
    outs = [add_asr_noise(src, 0.3, rng) for _ in range(20)]
    assert any(o != src for o in outs), "le bruit ne fait jamais rien"
    assert all(wer(src, o) < 0.8 for o in outs), "le bruit détruit la phrase"


def test_asr_noise_is_identity_at_zero():
    assert add_asr_noise("an ka manjé", 0.0, random.Random(0)) == "an ka manjé"
