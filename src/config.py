"""
Configuration centrale du pipeline ht -> gcf.
Toutes les phases importent depuis ici. Surchargeable par variables d'env.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("GCF_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
for _d in (DATA, ARTIFACTS):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    # --- Sources HuggingFace -------------------------------------------------
    hf_asr_dataset: str = "POTOMITAN/potomitan-gcf-transcription"
    hf_text_dataset: str = "POTOMITAN/PawolKreyol-gfc"
    hf_asr_split: str = "train"
    hf_text_split: str = "train"
    hf_token: str | None = os.environ.get("HF_TOKEN")

    # Noms de colonnes (auto-détectés si None, voir io_utils.detect_text_column)
    asr_text_column: str | None = None
    asr_audio_column: str | None = "audio"
    text_column: str | None = None

    # --- Phase 1 : diagnostic ------------------------------------------------
    n_diagnostic_segments: int = 300
    seed: int = 2026
    min_rule_support: int = 5          # occurrences minimales d'une substitution
    min_rule_confidence: float = 0.60  # p(tgt | src) minimale
    max_ngram: int = 3                 # règles jusqu'à 3 mots -> 3 mots

    # --- Phase 2 : paires synthétiques --------------------------------------
    n_synthetic_pairs: int = 80_000
    corruption_prob: float = 0.85      # p d'appliquer une règle inverse applicable
    noise_prob: float = 0.06           # bruit typographique additionnel (Whisper-like)
    min_chars: int = 12
    max_chars: int = 300
    split_ratios: tuple = (0.8, 0.1, 0.1)

    # --- Phase 3 : correcteur -----------------------------------------------
    base_model: str = "google/byt5-small"
    max_source_len: int = 256          # ByT5 = octets, pas tokens
    max_target_len: int = 256
    batch_size: int = 8
    grad_accum: int = 4
    lr: float = 3e-4
    epochs: int = 4
    label_smoothing: float = 0.1
    warmup_ratio: float = 0.06
    early_stopping_patience: int = 2
    fp16: bool = True

    # --- Phase 4 : filtrage --------------------------------------------------
    min_avg_logprob: float = -1.0
    max_compression_ratio: float = 2.0
    max_perplexity: float = 100.0
    kenlm_order: int = 5
    high_conf_threshold: float = 0.80  # score composite [0,1]

    # --- Phase 5 : pseudo-labeling ------------------------------------------
    nbest: int = 10
    pseudo_agreement_threshold: float = 0.85
    pseudo_epochs: int = 2

    # --- Livrables -----------------------------------------------------------
    rules_path: Path = ARTIFACTS / "substitution_rules_ht2gcf.json"
    residuals_path: Path = ARTIFACTS / "residual_cases.json"
    diagnostic_raw: Path = DATA / "diagnostic_to_correct.csv"
    diagnostic_done: Path = DATA / "diagnostic_corrected.csv"
    train_pairs: Path = DATA / "train_pairs.csv"
    val_pairs: Path = DATA / "val_pairs.csv"
    test_pairs: Path = DATA / "test_pairs.csv"
    model_dir: Path = ARTIFACTS / "gcf_corrector"
    model_ckpt: Path = ARTIFACTS / "gcf_corrector_bytesm.pt"
    kenlm_arpa: Path = ARTIFACTS / "pawolkreyol.arpa"
    kenlm_binary: Path = ARTIFACTS / "pawolkreyol.bin"
    corpus_txt: Path = DATA / "pawolkreyol_clean.txt"
    dataset_corrected: Path = ARTIFACTS / "dataset_corrected.jsonl"
    dataset_high_conf: Path = ARTIFACTS / "dataset_high_confidence.jsonl"
    report_path: Path = ARTIFACTS / "rapport.md"
    metrics_path: Path = ARTIFACTS / "metrics.json"

    extra: dict = field(default_factory=dict)


CFG = Config()
