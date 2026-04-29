"""Phase-1 data preparation for the text_transformer experiment.

Reads YAML config, materializes either raw bytes (uint8 numpy) or ByteBPE token
ids via :mod:`datasets.text_pipeline`, and writes a run directory under
``outputs/text_transformer/<stem>_bytes/`` or ``.../<stem>_v{vocab}/``.
``train.residual`` is reserved for later batching; not used here.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from datasets.text_pipeline import (
    BYTES_NPY,
    META_JSON,
    TOKENS_NPY,
    TOKENIZER_JSON,
    encode_bytes,
    encode_tokens,
)

OUT_SEG = ("outputs", "text_transformer")
MANIFEST_JSON = "manifest.json"
RESOLVED_CONFIG_YAML = "resolved_config.yaml"
DATA_SUBDIR = "data"


def _is_byte_mode(vocab_size: Any) -> bool:
    return vocab_size is None or vocab_size == 256


def _require_int_vocab_for_tokens(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"data.vocab_size must be an integer for token mode, got {v!r}")
    if v <= 0:
        raise ValueError(f"data.vocab_size must be positive, got {v}")
    return v


def _resolve_data_path(root: Path, path_value: str | Any) -> Path:
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root / path_value).resolve()


def run_dir_for_config(cfg: dict[str, Any], root: Path) -> Path:
    """``outputs/text_transformer/<stem>_bytes`` or ``<stem>_v{n}``."""
    data_cfg = cfg["data"]
    data_path = _resolve_data_path(root, data_cfg["path"])
    stem = data_path.stem
    vs = data_cfg.get("vocab_size")
    if _is_byte_mode(vs):
        suffix = "bytes"
    else:
        suffix = f"v{int(vs)}"
    return root.joinpath(*OUT_SEG, f"{stem}_{suffix}")


def prepare_data(
    root: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Load config, write ``data/`` artifacts and manifest under the run directory.

    Parameters
    ----------
    root
        Repository root (directory containing ``configs/``, ``data/``, ...).
    config_path
        Path to the experiment YAML (absolute or relative to *cwd*).

    Returns
    -------
    dict
        ``run_dir``, ``manifest_path``, ``resolved_config_path``, ``mode``,
        ``sequence_npy``, ``data_dir``, and ``manifest`` (parsed dict).
    """
    config_path = Path(config_path).resolve()
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp = cfg.get("experiment")
    if exp is not None and exp != "text_transformer":
        raise ValueError(f"Expected experiment: text_transformer, got {exp!r}")

    root = root.resolve()
    data_cfg = cfg["data"]
    source_path = _resolve_data_path(root, data_cfg["path"])
    if not source_path.is_file():
        raise FileNotFoundError(f"data file not found: {source_path}")

    run_dir = run_dir_for_config(cfg, root)
    data_dir = run_dir / DATA_SUBDIR
    data_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    vs = data_cfg.get("vocab_size")
    if _is_byte_mode(vs):
        mode = "bytes"
        encode_bytes(source_path, output_dir=data_dir)
        seq_path = data_dir / BYTES_NPY
        arr = np.load(seq_path)
        vocab_requested = None
        vocab_actual = 256
        seq_file = BYTES_NPY
        tokenizer_rel = None
    else:
        mode = "tokens"
        v = _require_int_vocab_for_tokens(vs)
        encode_tokens(
            source_path,
            v,
            output_dir=data_dir,
            train_sample_size=data_cfg.get("train_sample_size"),
            train_num_segments=int(data_cfg.get("train_num_segments", 40)),
            encode_chunk_bytes=int(data_cfg.get("encode_chunk_bytes", 5_000_000)),
            encode_batch_segments=int(data_cfg.get("encode_batch_segments", 4)),
        )
        seq_path = data_dir / TOKENS_NPY
        arr = np.load(seq_path)
        vocab_requested = v
        meta_path = data_dir / META_JSON
        with open(meta_path, encoding="utf-8") as mf:
            inner = json.load(mf)
        vocab_actual = int(inner["vocab_size_actual"])
        seq_file = TOKENS_NPY
        tokenizer_rel = TOKENIZER_JSON

    manifest: dict[str, Any] = {
        "experiment": "text_transformer",
        "name": cfg.get("name"),
        "config_path": str(config_path),
        "mode": mode,
        "source_path": str(source_path),
        "run_dir": str(run_dir),
        "data_dir": DATA_SUBDIR,
        "sequence_npy": f"{DATA_SUBDIR}/{seq_file}",
        "sequence_length": int(arr.shape[0]),
        "dtype": str(arr.dtype),
        "vocab_size_requested": vocab_requested,
        "vocab_size_actual": vocab_actual,
    }
    if tokenizer_rel is not None:
        manifest["tokenizer_json"] = f"{DATA_SUBDIR}/{tokenizer_rel}"

    manifest_path = run_dir / MANIFEST_JSON
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    resolved = copy.deepcopy(cfg)
    resolved["data"] = dict(resolved["data"])
    resolved["data"]["path"] = str(source_path)
    resolved["_config_source"] = str(config_path)
    resolved_path = run_dir / RESOLVED_CONFIG_YAML
    with open(resolved_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False, allow_unicode=True)

    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "resolved_config_path": resolved_path,
        "manifest": manifest,
        "mode": mode,
        "sequence_npy": seq_path,
        "data_dir": data_dir,
    }
