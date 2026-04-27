"""Vanilla-transformer compressor (our own work; no upstream/ dir).

Wraps `models.transformer_lm.TransformerLM` with arithmetic coding to act as
a lossless compressor under the `Compressor` interface.
"""

from .compressor import TransformerCompressor

__all__ = ["TransformerCompressor"]
