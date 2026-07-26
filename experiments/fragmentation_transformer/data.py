"""Streaming data setup for the Theorem 1 transformer experiment.

Generates one long Y sequence from the existing Markov-K source, fragments it
to X via the existing MSB-first bijection, and splits both into B contiguous
non-overlapping streams that are stacked along the batch dimension.
"""

import numpy as np

from core.markov import (
    fragment,
    generate_markov_sequence,
    sample_transition_matrix,
)


# Vocabulary size of the plain bit-fragmentation alphabet {0, 1}. The
# boundary token (when enabled) is the next free id above this base.
V_X_BASE = 2


def build_streams(M, K, alpha, n_sequence, seed, boundary=False):
    """Sample a Markov-K source and prepare the stream tensors.

    Parameters
    ----------
    boundary : bool, optional
        If True, append a boundary token (id `V_X_BASE`) after every M-bit
        block of X, marking the true block boundary so a finite-context
        predictor can always recover its phase. Default False reproduces the
        plain fragmentation exactly (byte-for-byte identical X).

    Returns a dict with:
        P                -- (V_y**K, V_y) row-stochastic transition matrix (numpy).
        Y                -- (n_Y,) int64 numpy array.
        X                -- (n_X,) int64 numpy array.
        V_y              -- int, alphabet size 2**M.
        V_x              -- int, X vocabulary size (V_X_BASE, or +1 if boundary).
        block_len_x      -- int, X tokens per Y symbol (M, or M+1 if boundary).
        boundary         -- bool, whether boundary marking is enabled.
        n_Y              -- int, length of Y.
        n_X              -- int, length of X.
    """
    V_y = 2 ** M
    rng = np.random.default_rng(seed)
    P = sample_transition_matrix(V_y, K, rng, alpha=alpha)
    Y = generate_markov_sequence(P, V_y, K, n_sequence, rng)
    boundary_token = V_X_BASE if boundary else None
    X = fragment(Y, M, boundary_token=boundary_token)

    block_len_x = M + (1 if boundary else 0)
    V_x = V_X_BASE + (1 if boundary else 0)

    return {
        "P": P,
        "Y": Y,
        "X": X,
        "V_y": V_y,
        "V_x": V_x,
        "block_len_x": block_len_x,
        "boundary": boundary,
        "n_Y": int(n_sequence),
        "n_X": int(n_sequence * block_len_x),
    }
