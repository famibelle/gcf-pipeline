"""
Cœur linguistique du pipeline :
  - alignement mot-à-mot entre une transcription Whisper-ht et sa correction gcf
  - fouille des substitutions systématiques -> RuleSet
  - application directe (ht -> gcf) et inverse (gcf -> pseudo-ht)

L'alignement par défaut est un alignement monotone par distance d'édition
(difflib), suffisant et robuste ici : ht et gcf sont des langues sœurs, l'ordre
des mots est quasi identique. simalign (embeddings multilingues) est disponible
en option pour les segments où l'ordre diverge (réordonnancements, insertions
lourdes) — cf. `align_words(method="simalign")`.
"""
from __future__ import annotations

import difflib
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Literal

from .normalize import normalize_surface, tokenize

AlignOp = Literal["equal", "replace", "delete", "insert"]


# ---------------------------------------------------------------------------
# Alignement
# ---------------------------------------------------------------------------

@dataclass
class AlignedBlock:
    op: AlignOp
    src: list[str]
    tgt: list[str]
    src_span: tuple[int, int]
    tgt_span: tuple[int, int]


def align_words(
    src_tokens: list[str],
    tgt_tokens: list[str],
    method: str = "levenshtein",
) -> list[AlignedBlock]:
    """
    Retourne la liste des blocs alignés. method ∈ {"levenshtein", "simalign"}.
    "levenshtein" utilise difflib.SequenceMatcher sur les tokens minusculés,
    ce qui donne exactement les opcodes equal/replace/delete/insert.
    """
    if method == "simalign":
        return _align_simalign(src_tokens, tgt_tokens)

    a = [t.lower() for t in src_tokens]
    b = [t.lower() for t in tgt_tokens]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    blocks: list[AlignedBlock] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        blocks.append(
            AlignedBlock(
                op=op,  # type: ignore[arg-type]
                src=src_tokens[i1:i2],
                tgt=tgt_tokens[j1:j2],
                src_span=(i1, i2),
                tgt_span=(j1, j2),
            )
        )
    return blocks


_SA = None  # aligneur simalign, instancié à la première utilisation (coûteux)


def _align_simalign(src_tokens: list[str], tgt_tokens: list[str]) -> list[AlignedBlock]:
    """Alignement par embeddings (optionnel). Nécessite `pip install simalign`."""
    try:
        from simalign import SentenceAligner
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "simalign non installé. `pip install simalign` ou method='levenshtein'."
        ) from exc

    global _SA
    if _SA is None:
        _SA = SentenceAligner(model="bert", token_type="bpe", matching_methods="i")

    pairs = _SA.get_word_aligns(src_tokens, tgt_tokens)["itermax"]
    blocks: list[AlignedBlock] = []
    for i, j in sorted(pairs):
        s, t = src_tokens[i], tgt_tokens[j]
        blocks.append(
            AlignedBlock(
                op="equal" if s.lower() == t.lower() else "replace",
                src=[s], tgt=[t], src_span=(i, i + 1), tgt_span=(j, j + 1),
            )
        )
    return blocks


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=a.lower(), b=b.lower(), autojunk=False).ratio()


def decompose_block(
    src: list[str], tgt: list[str], max_ngram: int = 3, gap_cost: float = 0.95
) -> list[tuple[list[str], list[str]]]:
    """
    Décompose un bloc de remplacement multi-mots en sous-substitutions.

    Indispensable : difflib renvoie « mwen ap manje » -> « an ka manjé » comme UN
    bloc de 3 mots. Sans décomposition, on perd les trois règles réelles
    (mwen->an, ap->ka, manje->manjé) et il ne reste presque rien à fouiller.

    Programmation dynamique monotone autorisant les appariements 1:1, 1:2, 2:1,
    plus insertions et suppressions. Le coût d'un appariement est 1 - similarité
    caractère, ce qui rapproche naturellement `manje`/`manjé` et sépare
    `mwen`/`an` de `ap`/`ka`.
    """
    n, m = len(src), len(tgt)
    if n == 0 or m == 0:
        return [(src, tgt)] if (src or tgt) else []

    INF = float("inf")
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    back: dict[tuple[int, int], tuple[int, int]] = {}
    dp[0][0] = 0.0

    # Pas de mouvement 2:2 : il masquerait les substitutions unitaires en les
    # absorbant dans un appariement de groupe (ex. « mwen ap » -> « an ka »
    # au lieu de mwen->an ET ap->ka). Le bloc entier reste proposé séparément
    # par mine_rules, donc les règles de séquence ne sont pas perdues.
    moves = [(1, 1), (1, 2), (2, 1), (1, 0), (0, 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] == INF:
                continue
            for di, dj in moves:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                if di == 0 or dj == 0:
                    cost = gap_cost * max(di, dj)
                else:
                    s = " ".join(src[i:ni])
                    t = " ".join(tgt[j:nj])
                    cost = 1.0 - _similarity(s, t)
                    if (di, dj) != (1, 1):
                        cost += 0.30          # forte préférence pour l'appariement 1:1
                if dp[i][j] + cost < dp[ni][nj]:
                    dp[ni][nj] = dp[i][j] + cost
                    back[(ni, nj)] = (i, j)

    # remontée du chemin
    path: list[tuple[list[str], list[str]]] = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj = back.get((i, j), (max(i - 1, 0), max(j - 1, 0)))
        path.append((src[pi:i], tgt[pj:j]))
        i, j = pi, pj
    path.reverse()
    return [(s, t) for s, t in path
            if (s or t) and len(s) <= max_ngram and len(t) <= max_ngram]


# ---------------------------------------------------------------------------
# Règles
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    src: str                      # séquence source (ht), tokens séparés par espace
    tgt: str                      # séquence cible (gcf)
    kind: str = "word"            # "word" | "regex" | "char"
    support: int = 0              # nb d'observations src->tgt
    src_total: int = 0            # nb d'occurrences de src côté ht
    confidence: float = 0.0       # support / src_total
    ngram: int = 1
    left_context: str | None = None    # contrainte optionnelle (token précédent)
    right_context: str | None = None
    examples: list[str] = field(default_factory=list)
    status: str = "mined"         # "mined" | "seed" | "manual" | "rejected"
    note: str = ""

    def key(self) -> tuple[str, str]:
        return (self.src, self.tgt)


class RuleSet:
    """Ensemble ordonné de règles, applicable dans les deux sens."""

    _warned_inverse_regex = False

    def __init__(self, rules: Iterable[Rule] | None = None, meta: dict | None = None):
        self.rules: list[Rule] = list(rules or [])
        self.meta: dict = meta or {}
        self._reindex()

    # -- indexation --------------------------------------------------------
    def _reindex(self) -> None:
        # tri : n-grams longs d'abord, puis confiance décroissante
        self.rules.sort(key=lambda r: (-r.ngram, -r.confidence, -r.support))
        self._fwd: dict[tuple[str, ...], list[Rule]] = defaultdict(list)
        self._bwd: dict[tuple[str, ...], list[Rule]] = defaultdict(list)
        self._regex: list[Rule] = []
        for r in self.rules:
            if r.status == "rejected":
                continue
            if r.kind == "regex":
                self._regex.append(r)
                continue
            self._fwd[tuple(r.src.split())].append(r)
            self._bwd[tuple(r.tgt.split())].append(r)
        self.max_ngram = max([r.ngram for r in self.rules], default=1)

    # -- (dé)sérialisation --------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": "1.0",
            "direction": "ht->gcf",
            "meta": self.meta,
            "n_rules": len(self.rules),
            "rules": [asdict(r) for r in self.rules],
        }

    @classmethod
    def from_dict(cls, d: dict) -> RuleSet:
        return cls([Rule(**r) for r in d.get("rules", [])], meta=d.get("meta", {}))

    def save(self, path) -> None:
        from .io_utils import write_json
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path) -> RuleSet:
        from .io_utils import read_json
        return cls.from_dict(read_json(path))

    # -- application ht -> gcf ---------------------------------------------
    def apply(self, text: str, prob: float = 1.0, rng: random.Random | None = None) -> str:
        return self._apply_directional(text, self._fwd, forward=True, prob=prob, rng=rng)

    # -- application inverse gcf -> pseudo-ht ------------------------------
    def apply_inverse(self, text: str, prob: float = 1.0, rng: random.Random | None = None) -> str:
        return self._apply_directional(text, self._bwd, forward=False, prob=prob, rng=rng)

    def _apply_directional(self, text, index, forward: bool, prob: float, rng) -> str:
        rng = rng or random
        tokens = tokenize(text)
        out: list[str] = []
        i = 0
        while i < len(tokens):
            matched = False
            for n in range(min(self.max_ngram, len(tokens) - i), 0, -1):
                key = tuple(t.lower() for t in tokens[i:i + n])
                cands = index.get(key)
                if not cands:
                    continue
                cands = [r for r in cands if self._context_ok(r, tokens, i, n, forward)]
                if not cands:
                    continue
                if prob < 1.0 and rng.random() > prob:
                    break  # règle applicable mais volontairement non appliquée
                rule = self._pick(cands, rng, forward)
                repl = (rule.tgt if forward else rule.src).split()
                repl = [_match_case(tokens[i], w) for w in repl[:1]] + repl[1:]
                out.extend(repl)
                i += n
                matched = True
                break
            if not matched:
                out.append(tokens[i])
                i += 1
        result = " ".join(out)
        # Les règles regex ne s'appliquent qu'en sens direct. Une substitution
        # regex est en général non inversible : `(\w{2,})e\b -> \1é` n'a pas
        # d'inverse bien défini, et permuter motif et remplacement produit soit
        # une erreur (`bad escape \s`), soit une réécriture silencieusement
        # fausse. La corruption e/é du sens inverse est couverte par le bruit
        # de `phase2_synthetic.add_asr_noise` (chute d'accents).
        if forward:
            for r in self._regex:
                if prob >= 1.0 or rng.random() <= prob:
                    result = re.sub(r.src, r.tgt, result)
        elif self._regex and not RuleSet._warned_inverse_regex:
            RuleSet._warned_inverse_regex = True
            from .io_utils import log
            log.warning("%d règle(s) regex ignorée(s) en sens inverse (non inversibles).",
                        len(self._regex))
        return normalize_surface(_fix_spacing(result))

    @staticmethod
    def _context_ok(rule: Rule, tokens: list[str], i: int, n: int, forward: bool) -> bool:
        if rule.left_context is not None:
            left = tokens[i - 1].lower() if i > 0 else "<s>"
            if left != rule.left_context:
                return False
        if rule.right_context is not None:
            right = tokens[i + n].lower() if i + n < len(tokens) else "</s>"
            if right != rule.right_context:
                return False
        return True

    @staticmethod
    def _pick(cands: list[Rule], rng, forward: bool) -> Rule:
        """Sens direct : règle la plus confiante. Sens inverse : tirage pondéré
        par le support, pour reproduire la variabilité réelle de Whisper."""
        if forward:
            return max(cands, key=lambda r: (r.confidence, r.support))
        weights = [max(r.support, 1) for r in cands]
        return rng.choices(cands, weights=weights, k=1)[0]

    # -- diagnostic ---------------------------------------------------------
    def coverage(self, pairs: list[tuple[str, str]]) -> dict:
        """% de segments intégralement expliqués par les règles."""
        exact = sum(1 for s, t in pairs
                    if normalize_surface(self.apply(s), lower=True)
                    == normalize_surface(t, lower=True))
        return {"n": len(pairs), "exact_match": exact,
                "exact_match_rate": round(exact / max(len(pairs), 1), 4)}


def _match_case(model: str, word: str) -> str:
    if model.isupper() and len(model) > 1:
        return word.upper()
    if model[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?…])")
_SPACE_APOSTROPHE = re.compile(r"\s*'\s*")


def _fix_spacing(text: str) -> str:
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_APOSTROPHE.sub("'", text)
    return text


# ---------------------------------------------------------------------------
# Fouille de règles à partir de paires corrigées
# ---------------------------------------------------------------------------

def mine_rules(
    pairs: list[tuple[str, str]],
    min_support: int = 5,
    min_confidence: float = 0.6,
    max_ngram: int = 3,
    align_method: str = "levenshtein",
) -> tuple[RuleSet, list[dict]]:
    """
    pairs : liste de (whisper_ht, corrigé_gcf).
    Retourne (RuleSet, résidus) où les résidus sont les blocs divergents non
    retenus comme règles (support ou confiance insuffisants) — la matière
    première du §5 de la phase 1.
    """
    sub_counts: Counter[tuple[str, str]] = Counter()
    src_totals: Counter[str] = Counter()
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    raw_blocks: list[dict] = []

    for seg_id, (src_text, tgt_text) in enumerate(pairs):
        s_tok, t_tok = tokenize(src_text), tokenize(tgt_text)

        # dénominateur : toutes les occurrences de n-grams source
        for n in range(1, max_ngram + 1):
            for i in range(len(s_tok) - n + 1):
                src_totals[" ".join(w.lower() for w in s_tok[i:i + n])] += 1

        for blk in align_words(s_tok, t_tok, method=align_method):
            if blk.op == "equal":
                continue
            context = " ".join(s_tok[max(0, blk.src_span[0] - 2):blk.src_span[1] + 2])

            # candidats : le bloc entier (s'il tient dans max_ngram) ET ses
            # sous-appariements. Les deux granularités coexistent ; à
            # l'application, la règle la plus longue l'emporte.
            candidates: list[tuple[list[str], list[str]]] = []
            if len(blk.src) <= max_ngram and len(blk.tgt) <= max_ngram:
                candidates.append((blk.src, blk.tgt))
            if blk.op == "replace" and (len(blk.src) > 1 or len(blk.tgt) > 1):
                candidates.extend(decompose_block(blk.src, blk.tgt, max_ngram))

            seen_here: set[tuple[str, str]] = set()
            for cs, ct in candidates:
                s = " ".join(w.lower() for w in cs)
                t = " ".join(w.lower() for w in ct)
                if (s, t) in seen_here:
                    continue
                seen_here.add((s, t))
                raw_blocks.append({
                    "segment_id": seg_id, "op": blk.op, "src": s, "tgt": t,
                    "src_context": context,
                })
                if not s:  # insertion pure : rien à substituer
                    continue
                sub_counts[(s, t)] += 1
                if len(examples[(s, t)]) < 3:
                    examples[(s, t)].append(src_text[:120])

    rules: list[Rule] = []
    kept_keys: set[tuple[str, str]] = set()
    for (s, t), cnt in sub_counts.most_common():
        total = max(src_totals.get(s, cnt), cnt)
        conf = cnt / total
        if cnt >= min_support and conf >= min_confidence:
            rules.append(Rule(
                src=s, tgt=t, kind="word", support=cnt, src_total=total,
                confidence=round(conf, 4), ngram=len(s.split()),
                examples=examples[(s, t)], status="mined",
            ))
            kept_keys.add((s, t))

    residuals = [
        {**b, "reason": "sous_le_seuil" if (b["src"], b["tgt"]) in sub_counts else "insertion",
         "support": sub_counts.get((b["src"], b["tgt"]), 0)}
        for b in raw_blocks if (b["src"], b["tgt"]) not in kept_keys
    ]

    meta = {
        "n_pairs": len(pairs),
        "min_support": min_support,
        "min_confidence": min_confidence,
        "max_ngram": max_ngram,
        "align_method": align_method,
        "n_candidate_substitutions": len(sub_counts),
        "n_rules_kept": len(rules),
        "n_residual_blocks": len(residuals),
    }
    return RuleSet(rules, meta=meta), residuals


def merge_rulesets(mined: RuleSet, seed: RuleSet, prefer: str = "mined") -> RuleSet:
    """Fusionne règles fouillées et règles amorces ; `prefer` tranche les conflits."""
    by_key: dict[tuple[str, str], Rule] = {}
    first, second = (seed, mined) if prefer == "mined" else (mined, seed)
    for rs in (first, second):
        for r in rs.rules:
            by_key[r.key()] = r
    meta = {**seed.meta, **mined.meta, "merged": True, "prefer": prefer}
    return RuleSet(by_key.values(), meta=meta)
