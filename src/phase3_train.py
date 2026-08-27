"""
PHASE 3 — FINE-TUNING DU CORRECTEUR

ByT5-small par défaut. Le choix est motivé : le correcteur doit gérer des
différences sous-lexicales (e/é, chute de consonne, agglutination) qu'un
tokenizer sous-mot entraîné sur d'autres langues fragmente mal. ByT5 travaille
sur les octets, ce qui convient à une langue peu dotée avec une orthographe
récente. `--model facebook/mbart-large-50` reste accepté pour comparaison.

Une note sur la perte : le cahier des charges mentionne « MAE ou label
smoothing ». La MAE n'a pas de sens pour une sortie discrète de génération —
on entraîne donc en entropie croisée avec label smoothing (défaut 0.1), qui est
l'intention derrière la demande : éviter la sur-confiance sur un corpus
synthétique dont les cibles ne sont pas parfaites.

Usage :
    python -m src.phase3_train --epochs 4 --batch-size 8
    python -m src.phase3_train --eval-only --checkpoint artifacts/gcf_corrector
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import CFG
from .io_utils import log, write_json
from .metrics import evaluate


def load_pairs_csv(path: Path) -> tuple[list[str], list[str]]:
    import csv
    src, tgt = [], []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            src.append(row["source_pseudo_ht"])
            tgt.append(row["target_gcf"])
    return src, tgt


PREFIX = "korije gcf: "   # préfixe de tâche, aide le modèle à ne pas paraphraser


def build_dataset(path: Path, tokenizer, cfg, limit: int | None = None):
    from datasets import Dataset

    src, tgt = load_pairs_csv(path)
    if limit:
        src, tgt = src[:limit], tgt[:limit]
    ds = Dataset.from_dict({"source": src, "target": tgt})

    def _tok(batch):
        model_in = tokenizer(
            [PREFIX + s for s in batch["source"]],
            max_length=cfg.max_source_len, truncation=True,
        )
        labels = tokenizer(
            batch["target"], max_length=cfg.max_target_len, truncation=True,
        )
        model_in["labels"] = labels["input_ids"]
        return model_in

    return ds.map(_tok, batched=True, remove_columns=["source", "target"],
                  desc=f"tokenisation {path.name}")


def train(args) -> dict:
    import numpy as np
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    cfg = CFG
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    train_ds = build_dataset(Path(args.train or cfg.train_pairs), tokenizer, cfg, args.limit)
    val_ds = build_dataset(Path(args.val or cfg.val_pairs), tokenizer, cfg, args.limit)
    log.info("train=%d | val=%d", len(train_ds), len(val_ds))

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dec_p = tokenizer.batch_decode(preds, skip_special_tokens=True)
        dec_l = tokenizer.batch_decode(labels, skip_special_tokens=True)
        from .metrics import corpus_cer, corpus_wer
        return {"wer": corpus_wer(dec_l, dec_p), "cer": corpus_cer(dec_l, dec_p)}

    targs = Seq2SeqTrainingArguments(
        output_dir=str(cfg.model_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=cfg.warmup_ratio,
        label_smoothing_factor=args.label_smoothing,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=cfg.max_target_len,
        generation_num_beams=args.beams,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to=[],
        seed=cfg.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    if not args.eval_only:
        trainer.train()
        trainer.save_model(str(cfg.model_dir))
        tokenizer.save_pretrained(str(cfg.model_dir))
        # Checkpoint léger demandé au format .pt, en plus du dossier HF.
        import torch as _t
        _t.save({
            "model_state_dict": trainer.model.state_dict(),
            "base_model": args.model,
            "prefix": PREFIX,
            "config": {k: str(v) for k, v in vars(cfg).items() if not k.startswith("_")},
        }, cfg.model_ckpt)
        log.info("Checkpoint -> %s", cfg.model_ckpt)

    metrics = test_report(trainer, tokenizer, Path(args.test or cfg.test_pairs),
                          n_examples=args.show)
    write_json(cfg.model_dir / "phase3_metrics.json", metrics)
    return metrics


def test_report(trainer, tokenizer, test_path: Path, n_examples: int = 12) -> dict:
    """WER avant/après sur test_pairs + exemples qualitatifs."""
    src, ref = load_pairs_csv(test_path)
    preds = generate_batch(trainer.model, tokenizer, src, batch_size=32)
    m = evaluate(src, preds, ref)

    examples = []
    for s, p, r in zip(src, preds, ref):
        if p != r:
            examples.append({"source_pseudo_ht": s, "prediction": p, "reference_gcf": r})
        if len(examples) >= n_examples:
            break

    print("\n=== PHASE 3 — test_pairs ===")
    print(f"  WER avant correction : {m['wer_before']:.4f}")
    print(f"  WER après correction : {m['wer_after']:.4f}")
    print(f"  Réduction relative   : {m['wer_reduction_rel']:.1%}")
    print(f"  Segments exacts      : {m['exact_match']}/{m['n']} ({m['exact_match_rate']:.1%})")
    print("\n--- Exemples d'erreurs résiduelles ---")
    for ex in examples[:n_examples]:
        print(f"  ht  : {ex['source_pseudo_ht']}")
        print(f"  pred: {ex['prediction']}")
        print(f"  ref : {ex['reference_gcf']}\n")

    if m["wer_after"] >= m["wer_before"]:
        print("ATTENTION : le correcteur ne réduit pas le WER. Causes usuelles : "
              "règles trop peu nombreuses (Phase 1), corruption déterministe "
              "(baisser --corruption-prob), ou sur-apprentissage.\n")

    return {**m, "examples": examples}


def generate_batch(model, tokenizer, texts: list[str], batch_size: int = 32,
                   beams: int = 4, max_len: int | None = None) -> list[str]:
    import torch

    device = next(model.parameters()).device
    model.eval()
    out: list[str] = []
    max_len = max_len or CFG.max_target_len
    for i in range(0, len(texts), batch_size):
        chunk = [PREFIX + t for t in texts[i:i + batch_size]]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=CFG.max_source_len).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=max_len, num_beams=beams,
                                 early_stopping=True)
        out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return out


def load_corrector(path: str | Path | None = None):
    """Recharge (modèle, tokenizer) depuis le dossier HF ou le .pt."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    path = Path(path or CFG.model_dir)
    if path.is_dir():
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
    else:
        ckpt = torch.load(path, map_location="cpu")
        tok = AutoTokenizer.from_pretrained(ckpt["base_model"])
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt["base_model"])
        model.load_state_dict(ckpt["model_state_dict"])
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tok


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 3 — fine-tuning du correcteur")
    p.add_argument("--model", default=CFG.base_model)
    p.add_argument("--train", default=None)
    p.add_argument("--val", default=None)
    p.add_argument("--test", default=None)
    p.add_argument("--epochs", type=float, default=CFG.epochs)
    p.add_argument("--batch-size", type=int, default=CFG.batch_size)
    p.add_argument("--grad-accum", type=int, default=CFG.grad_accum)
    p.add_argument("--lr", type=float, default=CFG.lr)
    p.add_argument("--label-smoothing", type=float, default=CFG.label_smoothing)
    p.add_argument("--patience", type=int, default=CFG.early_stopping_patience)
    p.add_argument("--beams", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="tronquer les datasets (debug)")
    p.add_argument("--fp16", action="store_true", default=CFG.fp16)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--show", type=int, default=12)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
