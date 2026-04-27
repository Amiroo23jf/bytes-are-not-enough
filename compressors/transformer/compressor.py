"""Vanilla transformer compressor: TransformerLM + arithmetic coding.

The model lives in `models.transformer_lm` so it can be shared with the
fragmentation_transformer and tokenization_transformer experiments. This
file only adds the arithmetic-coding wrapper that turns next-token
distributions into a lossless byte stream.

Stub: replace `_NOT_IMPLEMENTED` once the AC implementation lands.
"""

from compressors import Compressor, register_compressor

_NOT_IMPLEMENTED = (
    "Transformer compressor is not implemented yet. The model definition "
    "lives in models.transformer_lm; this wrapper still needs an arithmetic "
    "coder + bits-per-symbol bookkeeping."
)


class TransformerCompressor(Compressor):
    name = "transformer"

    def __init__(self, representation=None, model_kwargs=None, **kwargs):
        self.representation = representation
        self.model_kwargs = dict(model_kwargs or {})
        self._kwargs = kwargs

    def compress(self, data):
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def decompress(self, blob):
        raise NotImplementedError(_NOT_IMPLEMENTED)


register_compressor("transformer", TransformerCompressor)
