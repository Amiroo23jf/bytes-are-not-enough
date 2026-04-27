"""EDPC compressor sub-package.

Layout:
    adapter.py      -- our wrapper around the vendored EDPC codec.
    upstream/       -- vendored EDPC source (see upstream/README.md).

Not yet wired up: importing `EDPCCompressor` raises NotImplementedError until
the upstream code is vendored.
"""

from .adapter import EDPCCompressor

__all__ = ["EDPCCompressor"]
