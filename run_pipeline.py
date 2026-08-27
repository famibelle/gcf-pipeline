#!/usr/bin/env python3
"""
Orchestrateur : enchaîne les phases dans l'ordre, en s'arrêtant là où une
intervention humaine est requise (Phase 1) plutôt qu'en la contournant.

    python run_pipeline.py --from 2 --to 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import CFG
from src.io_utils import log


def phase1(_):
    from src.phase1_diagnostic import build_parser, cmd_mine, cmd_sample
    if not Path(CFG.diagnostic_done).exists():
        cmd_sample(build_parser().parse_args(["sample", "--n", str(CFG.n_diagnostic_segments)]))
        log.error("ARRÊT : la correction manuelle de la Phase 1 n'est pas automatisable. "
                  "Remplir la feuille puis relancer.")
        sys.exit(2)
    cmd_mine(build_parser().parse_args(["mine"]))


def phase2(_):
    from src.phase2_synthetic import build_parser, generate
    generate(build_parser().parse_args([]))


def phase_lm(_):
    from src.build_lm import build_parser, main
    main(build_parser().parse_args([]))


def phase3(_):
    from src.phase3_train import build_parser, train
    train(build_parser().parse_args([]))


def phase4(_):
    from src.phase4_filter_decode import build_parser, run
    run(build_parser().parse_args(["--calibrate", "--audit-sample", "200"]))


def phase_report(_):
    from src.report import main
    main(argparse.Namespace(out=None))


STEPS = [(1, "diagnostic", phase1), (2, "paires synthétiques", phase2),
         (2.5, "modèle de langue", phase_lm), (3, "correcteur", phase3),
         (4, "filtrage", phase4), (4.5, "rapport", phase_report)]


def main():
    p = argparse.ArgumentParser(description="Pipeline ht -> gcf")
    p.add_argument("--from", dest="start", type=float, default=1)
    p.add_argument("--to", dest="end", type=float, default=4.5)
    args = p.parse_args()
    for num, name, fn in STEPS:
        if args.start <= num <= args.end:
            log.info("=== Phase %s : %s ===", num, name)
            fn(args)
    log.info("Terminé. La Phase 5 (pseudo-labeling) se lance à la demande : "
             "python -m src.phase5_pseudolabel --help")


if __name__ == "__main__":
    main()
