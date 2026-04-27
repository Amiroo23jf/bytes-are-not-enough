"""Modality-specific dataset loaders for the compression_benchmarks experiment.

Each loader returns raw bytes (or a path to them) plus light metadata. The
representation pipeline (`representations.Pipeline`) is then responsible for
turning those bytes into the token stream the compressor consumes.

Not yet implemented; loaders for text, audio, and image will be added when
the compression_benchmarks experiment is wired up.
"""

__all__ = []
