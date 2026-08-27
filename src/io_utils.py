"""I/O : chargement des corpus HuggingFace, détection de colonnes, jsonl/csv."""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gcf")

TEXT_COLUMN_CANDIDATES = [
    "text", "transcription", "sentence", "transcript", "raw_text",
    "whisper_text", "prediction", "target", "content", "line", "texte",
]
AUDIO_COLUMN_CANDIDATES = ["audio", "audio_filepath", "path", "file", "wav"]


def detect_text_column(dataset, explicit: str | None = None) -> str:
    """Retourne le nom de la colonne texte, en privilégiant un nom explicite."""
    cols = list(dataset.column_names)
    if explicit:
        if explicit not in cols:
            raise KeyError(f"Colonne '{explicit}' absente. Disponibles : {cols}")
        return explicit
    for cand in TEXT_COLUMN_CANDIDATES:
        if cand in cols:
            return cand
    # dernier recours : première colonne de type string non-audio
    for c in cols:
        if c not in AUDIO_COLUMN_CANDIDATES:
            return c
    raise KeyError(f"Aucune colonne texte détectée parmi {cols}")


def load_hf(name: str, split: str, token: str | None = None, streaming: bool = False):
    """Charge un dataset HuggingFace, avec message d'erreur explicite si absent."""
    from datasets import load_dataset

    log.info("Chargement de %s [%s]%s", name, split, " (streaming)" if streaming else "")
    try:
        return load_dataset(name, split=split, token=token, streaming=streaming)
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de charger '{name}' (split={split}). "
            f"Vérifier le nom du dépôt, l'accès (HF_TOKEN) et la config. Cause : {exc}"
        ) from exc


def load_hf_audio_no_decode(name: str, split: str, token: str | None = None):
    """
    Charge le dataset ASR SANS décoder l'audio : on ne veut que le texte et les
    métadonnées pour les phases 1 et 4. Évite de télécharger/décoder les wav.
    """
    from datasets import Audio

    ds = load_hf(name, split, token=token)
    for col in AUDIO_COLUMN_CANDIDATES:
        if col in ds.column_names:
            try:
                ds = ds.cast_column(col, Audio(decode=False))
            except Exception as exc:  # noqa: BLE001
                log.debug("cast_column(%s) impossible, colonne laissée telle quelle : %s",
                          col, exc)
            break
    return ds


# ---------------------------------------------------------------------------
# jsonl / json
# ---------------------------------------------------------------------------

def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    log.info("Écrit %d lignes -> %s", n, path)
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Écrit -> %s", path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
