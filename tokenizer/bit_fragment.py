"""Bit-fragmentation tokenizer.

Wraps the existing `core.markov.fragment` operator (V_y-ary symbol -> M binary
tokens, MSB first) in the tokenizer protocol so that the fragmentation
experiments and the future tokenization_transformer experiment can swap in
arbitrary tokenizers via a single interface.

This is the trivial / "no real compression" tokenizer; it exists primarily as
a baseline and to validate the tokenizer plumbing end-to-end before BPE / LZW
arrive.
"""

import numpy as np

from core.markov import fragment


class BitFragmentTokenizer:
    """Map V_y = 2**M -ary symbols to binary tokens (MSB first)."""

    name = "bit_fragment"

    def __init__(self, M):
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        self.M = int(M)
        self.vocab_size = 2

    @property
    def source_vocab_size(self):
        return 2 ** self.M

    def encode(self, seq):
        """Encode a sequence of source symbols (in [0, 2**M)) to bits."""
        return fragment(np.asarray(seq, dtype=np.int64), self.M)

    def decode(self, tokens):
        """Inverse of `encode`. Tokens length must be a multiple of M."""
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.size % self.M != 0:
            raise ValueError(
                f"len(tokens)={tokens.size} is not a multiple of M={self.M}"
            )
        bits = tokens.reshape(-1, self.M)
        shifts = np.arange(self.M - 1, -1, -1, dtype=np.int64)
        return (bits * (1 << shifts)).sum(axis=1).astype(np.int64)
