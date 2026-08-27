"""
Modèle de langue gcf entraîné sur PawolKreyol-gfc.

Deux implémentations derrière la même interface :

  KenLMModel     — si kenlm + les binaires lmplz sont disponibles. C'est l'option
                   à privilégier (n-grammes 5, Kneser-Ney modifié).
  BackoffCharLM  — repli pur-Python : n-grammes de caractères avec stupid backoff.
                   Moins bon, mais il tourne partout et suffit à trier des segments
                   par plausibilité, ce qui est l'usage réel en Phase 4.

Le repli existe parce que compiler KenLM échoue souvent sur les machines de
travail ; il ne doit pas bloquer le pipeline. La perplexité renvoyée n'est PAS
comparable entre les deux modèles : le seuil `--max-perplexity` doit être
recalibré (voir `calibrate_threshold`) si l'on bascule de l'un à l'autre.
"""
from __future__ import annotations

import math
import pickle
import subprocess
from collections import defaultdict
from pathlib import Path

from .io_utils import log
from .normalize import normalize_surface, tokenize


class LMBase:
    def perplexity(self, text: str) -> float:
        raise NotImplementedError

    def log10_prob(self, text: str) -> float:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# KenLM
# ---------------------------------------------------------------------------

class KenLMModel(LMBase):
    def __init__(self, path: str | Path):
        import kenlm
        self.model = kenlm.Model(str(path))
        self.path = str(path)

    def log10_prob(self, text: str) -> float:
        return self.model.score(normalize_surface(text, lower=True), bos=True, eos=True)

    def perplexity(self, text: str) -> float:
        toks = tokenize(normalize_surface(text, lower=True))
        n = len(toks) + 1  # +1 pour </s>
        if n <= 1:
            return float("inf")
        return 10 ** (-self.log10_prob(text) / n)


def train_kenlm(corpus_txt: Path, arpa_out: Path, order: int = 5,
                binary_out: Path | None = None) -> bool:
    """Appelle lmplz/build_binary. Retourne False si les binaires sont absents."""
    try:
        subprocess.run(["lmplz", "--help"], capture_output=True, check=False)
    except FileNotFoundError:
        log.warning("lmplz introuvable — installer kenlm ou utiliser le repli caractère.")
        return False

    arpa_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["lmplz", "-o", str(order), "--discount_fallback",
           "--text", str(corpus_txt), "--arpa", str(arpa_out)]
    log.info("KenLM : %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        log.error("lmplz a échoué :\n%s", res.stderr[-2000:])
        return False
    if binary_out:
        subprocess.run(["build_binary", str(arpa_out), str(binary_out)],
                       capture_output=True, check=False)
    return True


# ---------------------------------------------------------------------------
# Repli : n-grammes de caractères, stupid backoff
# ---------------------------------------------------------------------------

class BackoffCharLM(LMBase):
    def __init__(self, order: int = 5, alpha: float = 0.4):
        self.order = order
        self.alpha = alpha
        self.counts: list[dict[str, int]] = [defaultdict(int) for _ in range(order + 1)]
        self.vocab: set[str] = set()
        self.total = 0

    def fit(self, sentences) -> BackoffCharLM:
        for s in sentences:
            s = "^" + normalize_surface(s, lower=True) + "$"
            self.vocab.update(s)
            for n in range(1, self.order + 1):
                for i in range(len(s) - n + 1):
                    self.counts[n][s[i:i + n]] += 1
            self.total += len(s)
        log.info("BackoffCharLM entraîné : %d caractères, |V|=%d", self.total, len(self.vocab))
        return self

    def _prob(self, ctx: str, ch: str) -> float:
        for k in range(len(ctx), -1, -1):
            c = ctx[len(ctx) - k:] if k else ""
            num = self.counts[k + 1].get(c + ch, 0)
            den = self.counts[k].get(c, 0) if k else self.total
            if num and den:
                return (self.alpha ** (len(ctx) - k)) * num / den
        return 1.0 / max(self.total, 1)

    def log10_prob(self, text: str) -> float:
        s = "^" + normalize_surface(text, lower=True) + "$"
        lp = 0.0
        for i in range(1, len(s)):
            ctx = s[max(0, i - self.order + 1):i]
            lp += math.log10(max(self._prob(ctx, s[i]), 1e-12))
        return lp

    def perplexity(self, text: str) -> float:
        s = "^" + normalize_surface(text, lower=True) + "$"
        n = len(s) - 1
        if n <= 0:
            return float("inf")
        return 10 ** (-self.log10_prob(text) / n)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"order": self.order, "alpha": self.alpha,
                         "counts": [dict(c) for c in self.counts],
                         "vocab": self.vocab, "total": self.total}, fh)
        log.info("BackoffCharLM -> %s", path)

    @classmethod
    def load(cls, path: Path) -> BackoffCharLM:
        with Path(path).open("rb") as fh:
            d = pickle.load(fh)
        m = cls(order=d["order"], alpha=d["alpha"])
        m.counts = [defaultdict(int, c) for c in d["counts"]]
        m.vocab, m.total = d["vocab"], d["total"]
        return m


# ---------------------------------------------------------------------------
# Chargement automatique + calibration
# ---------------------------------------------------------------------------

def load_lm(cfg) -> LMBase:
    for p in (cfg.kenlm_binary, cfg.kenlm_arpa):
        if Path(p).exists():
            try:
                log.info("LM : KenLM (%s)", p)
                return KenLMModel(p)
            except ImportError:
                log.warning("Module kenlm absent malgré le fichier %s", p)
            except Exception as exc:  # noqa: BLE001
                log.warning("Chargement KenLM impossible (%s)", exc)
    fallback = Path(cfg.kenlm_arpa).with_suffix(".charlm.pkl")
    if fallback.exists():
        log.info("LM : repli caractère (%s)", fallback)
        return BackoffCharLM.load(fallback)
    raise FileNotFoundError(
        "Aucun modèle de langue. Lancer : python -m src.build_lm"
    )


def calibrate_threshold(lm: LMBase, clean_sentences: list[str], quantile: float = 0.95) -> float:
    """
    Fixe le seuil de perplexité empiriquement plutôt qu'à la main : on mesure la
    perplexité du LM sur du gcf propre et on garde le quantile demandé. Un seuil
    codé en dur (100) n'a pas le même sens selon l'ordre du LM et le tokenizer.
    """
    vals = sorted(lm.perplexity(s) for s in clean_sentences if s.strip())
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("inf")
    idx = min(int(quantile * len(vals)), len(vals) - 1)
    thr = vals[idx]
    log.info("Seuil de perplexité calibré (q%.2f sur %d phrases propres) : %.1f",
             quantile, len(vals), thr)
    return thr
