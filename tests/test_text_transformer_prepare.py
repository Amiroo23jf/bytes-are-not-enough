"""Tests for text_transformer phase-1 prepare (bytes vs token numpy materialization)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets.text_pipeline import BYTES_NPY, TOKENS_NPY
from experiments.text_transformer.prepare import prepare_data, run_dir_for_config


def _minimal_cfg_bytes(rel_path: str) -> str:
    return f"""
experiment: text_transformer
name: test
data:
  path: {rel_path}
  vocab_size: null
train:
  batch_size: 2
  residual: null
model:
  n_layers: 1
  d_embd: 16
  d_ffn: 32
  n_heads: 2
  init_std: 0.02
"""


def _minimal_cfg_tokens(rel_path: str, vocab: int) -> str:
    return f"""
experiment: text_transformer
name: test
data:
  path: {rel_path}
  vocab_size: {vocab}
  train_num_segments: 4
  encode_chunk_bytes: 65536
  encode_batch_segments: 2
train:
  batch_size: 2
  residual: null
model:
  n_layers: 1
  d_embd: 16
  d_ffn: 32
  n_heads: 2
  init_std: 0.02
"""


def test_prepare_bytes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    corpus = root / "data" / "tiny.bin"
    corpus.write_bytes((b"ab " * 500) + b"\x00\xff")

    cfg_path = root / "cfg.yaml"
    cfg_path.write_text(_minimal_cfg_bytes("data/tiny.bin"), encoding="utf-8")

    out = prepare_data(root, cfg_path)
    assert out["mode"] == "bytes"
    seq = out["sequence_npy"]
    assert seq.name == BYTES_NPY
    arr = np.load(seq)
    assert arr.dtype == np.uint8
    assert arr.tobytes() == corpus.read_bytes()

    man_path = out["manifest_path"]
    man = json.loads(man_path.read_text(encoding="utf-8"))
    assert man["mode"] == "bytes"
    assert man["vocab_size_actual"] == 256
    assert man["sequence_length"] == len(arr)
    assert (out["run_dir"] / "resolved_config.yaml").is_file()


def test_prepare_tokens(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    corpus = root / "data" / "tiny.bin"
    corpus.write_bytes((b"ab " * 2000) + b"\x00\xff")

    cfg_path = root / "cfg.yaml"
    cfg_path.write_text(_minimal_cfg_tokens("data/tiny.bin", 512), encoding="utf-8")

    out = prepare_data(root, cfg_path)
    assert out["mode"] == "tokens"
    assert out["sequence_npy"].name == TOKENS_NPY
    arr = np.load(out["sequence_npy"])
    assert arr.ndim == 1
    assert np.issubdtype(arr.dtype, np.integer)

    man = out["manifest"]
    assert man["mode"] == "tokens"
    assert man["vocab_size_requested"] == 512
    assert man["tokenizer_json"] == "data/tokenizer.json"


def test_run_dir_for_config_bytes_vs_tokens(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    cfg_b = {"data": {"path": "data/x.bin", "vocab_size": None}}
    cfg_t = {"data": {"path": "data/x.bin", "vocab_size": 8000}}
    assert run_dir_for_config(cfg_b, root).name == "x_bytes"
    assert run_dir_for_config(cfg_t, root).name == "x_v8000"


def test_byte_mode_explicit_256(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    corpus = root / "data" / "tiny.bin"
    corpus.write_bytes(b"a")

    cfg_path = root / "cfg.yaml"
    cfg_path.write_text(
        """
experiment: text_transformer
name: test
data:
  path: data/tiny.bin
  vocab_size: 256
train:
  batch_size: 2
  residual: null
model:
  n_layers: 1
  d_embd: 16
  d_ffn: 32
  n_heads: 2
  init_std: 0.02
""".lstrip(),
        encoding="utf-8",
    )
    out = prepare_data(root, cfg_path)
    assert out["mode"] == "bytes"
