"""Text file loaders: raw bytes or ByteLevel-BPE + frequency remap.

Encode artifacts default to ``outputs/datasets/<input_stem>/``. This module
performs all persistence; :mod:`tokenizer.bytebpe` only trains/encodes in memory.

Run the CLI with: ``python -m datasets.text_pipeline``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from tokenizer.bytebpe import decode_bytes, tokenize_data

BYTES_NPY = "bytes_uint8.npy"
TOKENS_NPY = "tokens_int64.npy"
TOKENIZER_JSON = "tokenizer.json"
UNIGRAM_NPY = "unigram_probs.npy"
REMAP_NPY = "remap_old_to_new.npy"
SORTED_OLD_IDS_NPY = "sorted_old_ids.npy"
META_JSON = "meta.json"
DEFAULT_OUTPUT_ROOT = Path("outputs/datasets")


def output_dir_for_input(
    input_path: Path | str,
    root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """``root / input_path.stem``.

    Two different files with the same basename will target the same directory;
    pass a distinct ``output_dir`` when calling encode if that matters.
    """
    p = Path(input_path)
    return Path(root) / p.stem


def _write_meta(path: Path, meta: dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def encode_bytes(
    input_path: Path | str,
    output_dir: Path | str | None = None,
    *,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Read *input_path*, save uint8 numpy array and metadata.

    Returns path to the directory containing artifacts.
    """
    input_path = Path(input_path).resolve()
    out = Path(output_dir) if output_dir is not None else output_dir_for_input(
        input_path, root=output_root
    )
    out.mkdir(parents=True, exist_ok=True)

    raw = input_path.read_bytes()
    arr = np.frombuffer(raw, dtype=np.uint8).copy()
    np.save(out / BYTES_NPY, arr)

    _write_meta(
        out / META_JSON,
        {
            "mode": "bytes",
            "source_path": str(input_path),
            "num_bytes": len(raw),
            "array_file": BYTES_NPY,
            "dtype": "uint8",
        },
    )
    return out


def decode_bytes_file(npy_path: Path | str, output_path: Path | str) -> None:
    """Load ``bytes_uint8.npy`` and write raw bytes to *output_path*."""
    npy_path = Path(npy_path)
    arr = np.load(npy_path)
    Path(output_path).write_bytes(arr.astype(np.uint8).tobytes())


def encode_tokens(
    input_path: Path | str,
    vocab_size: int,
    output_dir: Path | str | None = None,
    *,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    train_sample_size: int | None = None,
    train_num_segments: int = 40,
    encode_chunk_bytes: int = 5_000_000,
    encode_batch_segments: int = 4,
) -> Path:
    """Train ByteBPE, remap by frequency, save numpy + tokenizer + remap maps."""
    input_path = Path(input_path).resolve()
    out = Path(output_dir) if output_dir is not None else output_dir_for_input(
        input_path, root=output_root
    )
    out.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    (
        data_arr,
        tokenizer,
        unigram_probs,
        old_to_new,
        sorted_old_ids,
    ) = tokenize_data(
        str(input_path),
        vocab_size=vocab_size,
        train_sample_size=train_sample_size,
        train_num_segments=train_num_segments,
        encode_chunk_bytes=min(encode_chunk_bytes, max(1, input_path.stat().st_size)),
        encode_batch_segments=encode_batch_segments,
        base_dir=None,
    )

    np.save(out / TOKENS_NPY, data_arr)
    np.save(out / UNIGRAM_NPY, unigram_probs)
    np.save(out / REMAP_NPY, old_to_new)
    np.save(out / SORTED_OLD_IDS_NPY, sorted_old_ids)

    tokenizer_path = out / TOKENIZER_JSON
    tokenizer.save(str(tokenizer_path))

    _write_meta(
        out / META_JSON,
        {
            "mode": "tokens",
            "source_path": str(input_path),
            "vocab_size_requested": vocab_size,
            "vocab_size_actual": int(tokenizer.get_vocab_size()),
            "num_tokens": int(data_arr.shape[0]),
            "tokens_file": TOKENS_NPY,
            "tokenizer_file": TOKENIZER_JSON,
            "unigram_probs_file": UNIGRAM_NPY,
            "remap_old_to_new_file": REMAP_NPY,
            "sorted_old_ids_file": SORTED_OLD_IDS_NPY,
        },
    )

    return out


def decode_tokens_files(
    tokens_npy: Path | str,
    tokenizer_json: Path | str,
    output_path: Path | str,
) -> None:
    """Load token IDs and tokenizer; write decoded bytes to *output_path*."""
    from tokenizers import Tokenizer

    ids = np.load(Path(tokens_npy))
    tok = Tokenizer.from_file(str(Path(tokenizer_json)))
    data = decode_bytes(tok, ids)
    Path(output_path).write_bytes(data)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Text dataset: bytes or ByteBPE pipeline.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bytes", help="Read file → uint8 numpy under outputs/datasets/<stem>/")
    b.add_argument("--input", type=Path, required=True)
    b.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"default: {DEFAULT_OUTPUT_ROOT}/<input_stem>",
    )

    t = sub.add_parser("tokens", help="Train ByteBPE + save token numpy + tokenizer")
    t.add_argument("--input", type=Path, required=True)
    t.add_argument("--vocab-size", type=int, required=True)
    t.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"default: {DEFAULT_OUTPUT_ROOT}/<input_stem>",
    )
    t.add_argument("--train-sample-size", type=int, default=None)
    t.add_argument("--train-num-segments", type=int, default=40)
    t.add_argument("--encode-chunk-bytes", type=int, default=5_000_000)
    t.add_argument("--encode-batch-segments", type=int, default=4)

    db = sub.add_parser("decode-bytes", help="bytes_uint8.npy → raw file")
    db.add_argument("--npy", type=Path, required=True, help="Path to bytes_uint8.npy")
    db.add_argument("--output", type=Path, required=True)

    dt = sub.add_parser("decode-tokens", help="tokens + tokenizer JSON → raw bytes")
    dt.add_argument("--tokens", type=Path, required=True, help="tokens_int64.npy")
    dt.add_argument("--tokenizer", type=Path, required=True, help="tokenizer.json")
    dt.add_argument("--output", type=Path, required=True)

    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    if args.cmd == "bytes":
        out = encode_bytes(args.input, output_dir=args.output_dir)
        print(out)
    elif args.cmd == "tokens":
        out = encode_tokens(
            args.input,
            args.vocab_size,
            output_dir=args.output_dir,
            train_sample_size=args.train_sample_size,
            train_num_segments=args.train_num_segments,
            encode_chunk_bytes=args.encode_chunk_bytes,
            encode_batch_segments=args.encode_batch_segments,
        )
        print(out)
    elif args.cmd == "decode-bytes":
        decode_bytes_file(args.npy, args.output)
    elif args.cmd == "decode-tokens":
        decode_tokens_files(args.tokens, args.tokenizer, args.output)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
