"""End-to-end smoke: bytes and ByteBPE token pipelines round-trip to source bytes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets.text_pipeline import (
    BYTES_NPY,
    TOKENS_NPY,
    TOKENIZER_JSON,
    decode_bytes_file,
    decode_tokens_files,
    encode_bytes,
    encode_tokens,
)


@pytest.fixture
def sample_bytes() -> bytes:
    # Repetitive content so ByteLevel BPE min_frequency=2 sees statistics on small files.
    return (b"ab " * 2000) + b"\x00\xff"


def test_bytes_roundtrip(tmp_path: Path, sample_bytes: bytes) -> None:
    src = tmp_path / "sample.bin"
    src.write_bytes(sample_bytes)

    out = encode_bytes(src, output_dir=tmp_path / "out_bytes")
    assert (out / BYTES_NPY).is_file()

    decoded = tmp_path / "roundtrip.bin"
    decode_bytes_file(out / BYTES_NPY, decoded)
    assert decoded.read_bytes() == sample_bytes


def test_tokens_roundtrip(tmp_path: Path, sample_bytes: bytes) -> None:
    src = tmp_path / "sample.bin"
    src.write_bytes(sample_bytes)

    out = encode_tokens(
        src,
        vocab_size=512,
        output_dir=tmp_path / "out_tok",
        train_num_segments=4,
        encode_chunk_bytes=65_536,
        encode_batch_segments=2,
    )
    assert (out / TOKENS_NPY).is_file()
    assert (out / TOKENIZER_JSON).is_file()

    decoded = tmp_path / "roundtrip_tok.bin"
    decode_tokens_files(out / TOKENS_NPY, out / TOKENIZER_JSON, decoded)
    assert decoded.read_bytes() == sample_bytes
