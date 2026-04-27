"""Vectorized Laplace-smoothed n-gram cross-entropy estimator.

Used as the workhorse predictor for the fragmentation_ngram experiment, and
as a baseline that other experiments (tokenization_transformer,
compression_benchmarks) can call without depending on any experiment package.

All cross-entropies are returned in bits per token (log base 2).
"""

import numpy as np


def context_ints(seq, context_len, vocab_size):
    """Integer-encode every length-`context_len` window in `seq` (big-endian).

    Returns an int64 array of length (n - context_len + 1) where entry i is
        sum_{j=0}^{K-1} seq[i + j] * vocab_size^(K-1-j).

    For context_len == 0 returns a length-(n+1) zero vector (every window
    collapses to the single empty context).
    """
    n = len(seq)
    if context_len == 0:
        return np.zeros(n + 1, dtype=np.int64)
    ctx_ints = np.zeros(n - context_len + 1, dtype=np.int64)
    for j in range(context_len):
        ctx_ints = ctx_ints * vocab_size + seq[j : n - context_len + 1 + j]
    return ctx_ints


def ngram_cross_entropy_bits(seq, context_len, vocab_size, smoothing_alpha):
    """Per-token cross-entropy (bits/token) of a Laplace-smoothed n-gram.

    Trains and evaluates on the same sequence (matches the Theorem 3.1 spec).
    Fully vectorized via rolling-hash context encoding and dense count arrays.

    Parameters
    ----------
    seq : array-like of int
        The token sequence.
    context_len : int
        Number of context tokens conditioned on (the "K" of an n-gram with
        n = K+1).
    vocab_size : int
        Alphabet size; sets the count-array width.
    smoothing_alpha : float
        Add-alpha (Laplace) smoothing constant per (context, next-token) cell.

    Returns
    -------
    float
        Average cross-entropy in bits per predicted token.
    """
    seq = np.asarray(seq, dtype=np.int64)
    n = len(seq)
    if n <= context_len:
        return 0.0

    n_ctxs = vocab_size ** context_len if context_len > 0 else 1
    counts = np.zeros((n_ctxs, vocab_size), dtype=np.float64)

    if context_len == 0:
        ctx_ints = np.zeros(n, dtype=np.int64)
    else:
        ctx_ints = context_ints(seq, context_len, vocab_size)[: n - context_len]
    next_syms = seq[context_len:]

    np.add.at(counts, (ctx_ints, next_syms), 1.0)

    row_sums = counts.sum(axis=1)
    denoms = row_sums[ctx_ints] + smoothing_alpha * vocab_size
    nums = counts[ctx_ints, next_syms] + smoothing_alpha
    p_pred = nums / denoms
    loss = -np.log2(np.maximum(p_pred, 1e-15)).sum()
    return float(loss / (n - context_len))
