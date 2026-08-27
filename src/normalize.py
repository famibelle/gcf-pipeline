"""
Normalisation orthographique et tokenisation.

Deux niveaux distincts, à ne jamais confondre :

1. `normalize_surface()`  -> nettoyage neutre (unicode, ponctuation, espaces).
   Appliqué partout, des deux côtés (ht et gcf). Ne change aucune graphie.

2. `enforce_gerec2()`     -> contraintes de l'alphabet GEREC-2 du gcf.
   Appliqué UNIQUEMENT côté cible gcf, en post-traitement. C'est un garde-fou,
   pas un traducteur : il retire les graphèmes hors alphabet, il ne devine rien.

L'alphabet GEREC-2 retenu pour le gcf :
    a b ch d e é è f g h i j k l m n ng o ò p r s t v w y z
    digraphes vocaliques : ou, an, on, en (nasales)
Les lettres c, q, u (seule), x et les accents circonflexes n'appartiennent pas
au système. `enforce_gerec2` les signale (mode strict) ou les corrige (mode fix).
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Niveau 1 : surface
# ---------------------------------------------------------------------------

_QUOTES = {
    "\u2018": "'", "\u2019": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00a0": " ", "\u202f": " ", "\u2009": " ",
    "\u2026": "...",
}

_WS = re.compile(r"\s+")
_PUNCT_SPACE = re.compile(r"\s+([,.;:!?%])")
_MULTI_PUNCT = re.compile(r"([,.;:!?])\1+")


def normalize_surface(text: str, lower: bool = False) -> str:
    """Nettoyage neutre, sans effet sur la graphie créole."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    for src, tgt in _QUOTES.items():
        text = text.replace(src, tgt)
    text = _MULTI_PUNCT.sub(r"\1", text)
    text = _PUNCT_SPACE.sub(r"\1", text)
    text = _WS.sub(" ", text).strip()
    if lower:
        text = text.lower()
    return text


# ---------------------------------------------------------------------------
# Niveau 2 : GEREC-2
# ---------------------------------------------------------------------------

GEREC2_LETTERS = set("abdefghijklmnoprstvwyz") | {"é", "è", "ò", "à"}
OUT_OF_ALPHABET = set("cqux") | {"ê", "â", "î", "ô", "û", "ë", "ï", "ü", "ç"}

# Corrections déterministes sûres (graphème -> graphème), hors contexte lexical.
_GEREC2_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ç"), "s"),
    (re.compile(r"[êë]"), "è"),
    (re.compile(r"[âà](?!\b)"), "a"),
    (re.compile(r"[îï]"), "i"),
    (re.compile(r"[ôö]"), "ò"),
    (re.compile(r"[ûü]"), "u"),
    (re.compile(r"qu"), "k"),
    (re.compile(r"ph"), "f"),
    (re.compile(r"c([kh])", re.IGNORECASE), r"\1"),      # cch -> ch
    (re.compile(r"c(?=[eiéè])"), "s"),
    (re.compile(r"c(?=[aouò])"), "k"),
    (re.compile(r"q(?!u)"), "k"),
    (re.compile(r"x"), "ks"),
]


def enforce_gerec2(text: str, mode: str = "fix") -> str:
    """
    mode='fix'    : applique les corrections graphémiques déterministes.
    mode='strict' : identique, mais lève une erreur s'il reste un caractère hors alphabet.
    mode='check'  : ne modifie rien, retourne le texte tel quel.
    """
    if mode == "check":
        return text
    out = text
    for pat, rep in _GEREC2_FIXES:
        out = pat.sub(rep, out)
    if mode == "strict":
        residual = {c for c in out.lower() if c.isalpha() and c in OUT_OF_ALPHABET}
        if residual:
            raise ValueError(f"Caractères hors GEREC-2 restants {sorted(residual)} dans: {out!r}")
    return out


def gerec2_violations(text: str) -> list[str]:
    """Liste les caractères hors alphabet GEREC-2 présents dans le texte."""
    return sorted({c for c in text.lower() if c.isalpha() and c in OUT_OF_ALPHABET})


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[\w'’\-]+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenisation mot + ponctuation, conserve les apostrophes internes (ex: k'ay)."""
    return _TOKEN.findall(normalize_surface(text))


def detokenize(tokens: list[str]) -> str:
    out = " ".join(tokens)
    return normalize_surface(out)


def strip_accents(text: str) -> str:
    """Utile pour simuler la perte d'accents typique de Whisper-ht."""
    nfd = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in nfd if not unicodedata.combining(c)))
