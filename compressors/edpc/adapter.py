"""Adapter: vendored EDPC codec -> Compressor interface.

This file is the *only* place in compression_benchmarks that mixes our
representation pipeline with the upstream EDPC code under `upstream/`. Every
modification to the upstream code must be logged in
`compressors/edpc/upstream/MODIFICATIONS.md`.

Stub: replace `_NOT_VENDORED_YET` once `upstream/` is populated.
"""

from compressors import Compressor, register_compressor

_NOT_VENDORED_YET = (
    "EDPC upstream code has not been vendored yet. See "
    "compressors/edpc/upstream/README.md for the vendoring procedure."
)


class EDPCCompressor(Compressor):
    name = "edpc"

    def __init__(self, representation=None, **kwargs):
        # `representation` is an optional `representations.Pipeline` instance;
        # when provided, raw bytes are decoded / tokenized before being fed
        # to the underlying EDPC codec.
        self.representation = representation
        self._kwargs = kwargs

    def compress(self, data):
        raise NotImplementedError(_NOT_VENDORED_YET)

    def decompress(self, blob):
        raise NotImplementedError(_NOT_VENDORED_YET)


register_compressor("edpc", EDPCCompressor)
