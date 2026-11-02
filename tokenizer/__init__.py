"""Tokenizer registry.

A tokenizer maps a sequence of source symbols to a sequence of token ids over
a (possibly different) vocabulary. The protocol is intentionally minimal so
that new tokenizers can be plugged in by adding a single module + one
registry entry.

Protocol expected of each tokenizer:
    .vocab_size : int                       # size of the output alphabet
    .encode(seq: Sequence[int]) -> np.ndarray  # source ids -> token ids
    .decode(tokens: Sequence[int]) -> np.ndarray  # token ids -> source ids
    .name : str                             # for logging / config tagging

Use `get_tokenizer(name, **kwargs)` to instantiate by name.
"""

from .bit_fragment import BitFragmentTokenizer
from .bpe import BPETokenizer

_REGISTRY = {
    "bit_fragment": BitFragmentTokenizer,
    "bpe": BPETokenizer,
}


def get_tokenizer(name, **kwargs):
    """Instantiate a tokenizer by name.

    Raises KeyError if the name is unknown. Future tokenizers (LZW, ...) are
    registered by importing them here and adding to `_REGISTRY`.

    Note: `BPETokenizer` requires an already-trained state, so callers
    should typically use `BPETokenizer.train(sequence, vocab_size, ...)` or
    `BPETokenizer.load(path)` rather than `get_tokenizer("bpe", ...)`.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown tokenizer {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def register_tokenizer(name, cls):
    """Register a tokenizer class under `name`. Overwrites if already present."""
    _REGISTRY[name] = cls


__all__ = [
    "BitFragmentTokenizer",
    "BPETokenizer",
    "get_tokenizer",
    "register_tokenizer",
]
