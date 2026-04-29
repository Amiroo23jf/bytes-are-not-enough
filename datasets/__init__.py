"""Modality-specific dataset loaders for compression experiments.

Import pipeline helpers from ``datasets.text_pipeline`` (or ``from datasets import
<name>`` — resolved lazily so ``python -m datasets.text_pipeline`` does not
preload this submodule).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BYTES_NPY",
    "DEFAULT_OUTPUT_ROOT",
    "META_JSON",
    "REMAP_NPY",
    "SORTED_OLD_IDS_NPY",
    "TOKENIZER_JSON",
    "TOKENS_NPY",
    "UNIGRAM_NPY",
    "decode_bytes_file",
    "decode_tokens_files",
    "encode_bytes",
    "encode_tokens",
    "output_dir_for_input",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        tp = importlib.import_module(f"{__name__}.text_pipeline")
        return getattr(tp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
