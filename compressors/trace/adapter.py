"""Adapter: vendored TRACE codec -> Compressor interface.

This file is the *only* place in compression_benchmarks that mixes our
representation pipeline with the upstream TRACE code under `upstream/`. Every
modification to the upstream code must be logged in
`compressors/trace/upstream/MODIFICATIONS.md`.

Stub: replace `_NOT_VENDORED_YET` once `upstream/` is populated.
"""

from compressors import Compressor, register_compressor

_NOT_VENDORED_YET = (
    "TRACE upstream code has not been vendored yet. See "
    "compressors/trace/upstream/README.md for the vendoring procedure."
)


class TRACECompressor(Compressor):
    name = "trace"

    def __init__(self, representation=None, **kwargs):
        self.representation = representation
        self._kwargs = kwargs

    def compress(self, data):
        raise NotImplementedError(_NOT_VENDORED_YET)

    def decompress(self, blob):
        raise NotImplementedError(_NOT_VENDORED_YET)


register_compressor("trace", TRACECompressor)
