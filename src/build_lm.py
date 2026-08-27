"""
Construit le modèle de langue gcf à partir de PawolKreyol-gfc.
Tente KenLM ; bascule sur le repli caractère si les binaires manquent.

Usage :
    python -m src.build_lm --order 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import CFG
from .io_utils import log
from .lm import BackoffCharLM, train_kenlm
from .phase2_synthetic import iter_gcf_sentences


def main(args) -> None:
    cfg = CFG
    corpus = Path(args.corpus or cfg.corpus_txt)
    corpus.parent.mkdir(parents=True, exist_ok=True)

    if args.rebuild_corpus or not corpus.exists():
        n = 0
        with corpus.open("w", encoding="utf-8") as fh:
            for sent in iter_gcf_sentences(cfg, limit=args.limit):
                fh.write(sent.lower() + "\n")
                n += 1
        log.info("Corpus texte : %d phrases -> %s", n, corpus)
    else:
        log.info("Corpus existant réutilisé : %s", corpus)

    if not args.force_fallback and train_kenlm(corpus, cfg.kenlm_arpa,
                                               order=args.order,
                                               binary_out=cfg.kenlm_binary):
        log.info("KenLM prêt : %s", cfg.kenlm_arpa)
        return

    log.info("Bascule sur le modèle caractère (repli).")
    sents = corpus.read_text(encoding="utf-8").splitlines()
    lm = BackoffCharLM(order=args.order).fit(sents)
    lm.save(Path(cfg.kenlm_arpa).with_suffix(".charlm.pkl"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Construction du LM gcf")
    p.add_argument("--order", type=int, default=CFG.kenlm_order)
    p.add_argument("--corpus", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--rebuild-corpus", action="store_true")
    p.add_argument("--force-fallback", action="store_true")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
