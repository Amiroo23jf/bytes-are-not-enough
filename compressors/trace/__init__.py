"""TRACE compressor sub-package.

Layout:
    adapter.py      -- our wrapper around the vendored TRACE codec.
    upstream/       -- vendored TRACE source (see upstream/README.md).

Not yet wired up: importing `TRACECompressor` raises NotImplementedError until
the upstream code is vendored.
"""

from .adapter import TRACECompressor

__all__ = ["TRACECompressor"]
