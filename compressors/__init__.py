"""Compressor base class + registry.

The `Compressor` ABC is the single API surface compression_benchmarks talks
to, regardless of whether the underlying codec is EDPC, TRACE, our own
transformer compressor, or a classical baseline (gzip, etc.).

External codecs (EDPC, TRACE) are vendored as sub-packages under this
directory; see THIRD_PARTY_NOTICES.md and each compressor's
`upstream/README.md` for vendoring metadata.

Each concrete compressor lives in its own sub-package
(`compressors.<name>.adapter`) and is responsible for:
  1. Adapting the upstream codec's API to the methods below.
  2. Optionally accepting a `representation` (a
     `representations.Pipeline` instance) so the same compressor can run
     either on raw bytes or on a decoded / tokenized stream.
"""

from abc import ABC, abstractmethod


class Compressor(ABC):
    """Minimal interface every compressor in the benchmark implements."""

    name = "abstract"

    @abstractmethod
    def compress(self, data):
        """Encode `data` (bytes-like) to a compressed byte string."""

    @abstractmethod
    def decompress(self, blob):
        """Inverse of `compress`. `blob` is what `compress` returned."""

    def bits_per_symbol(self, data):
        """Compression rate in bits per source symbol.

        Default implementation: len(compress(data)) * 8 / len(data). Override
        if "symbol" means something other than "source byte" for this codec
        (e.g. when a representation pipeline is in use).
        """
        if len(data) == 0:
            return 0.0
        return len(self.compress(data)) * 8.0 / len(data)


_REGISTRY = {}


def register_compressor(name, cls):
    """Register a compressor class under `name`. Overwrites if already present."""
    _REGISTRY[name] = cls


def get_compressor(name, **kwargs):
    """Instantiate a compressor by name. Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown compressor {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


__all__ = ["Compressor", "register_compressor", "get_compressor"]
