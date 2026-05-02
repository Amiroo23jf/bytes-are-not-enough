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


def build_streams(M, K, alpha, n_sequence, seed):
    """Sample a Markov-K source and prepare the stream tensors.

    Returns a dict with:
        P                -- (V_y**K, V_y) row-stochastic transition matrix (numpy).
        Y                -- (n_Y,) int64 numpy array.
        X                -- (n_X,) int64 numpy array.
        V_y              -- int, alphabet size 2**M.
        n_Y              -- int, length of Y.
        n_X              -- int, length of X.
    """
    V_y = 2 ** M
    rng = np.random.default_rng(seed)
    P = sample_transition_matrix(V_y, K, rng, alpha=alpha)
    Y = generate_markov_sequence(P, V_y, K, n_sequence, rng)
    X = fragment(Y, M)

    return {
        "P": P,
        "Y": Y,
        "X": X,
        "V_y": V_y,
        "n_Y": int(n_sequence),
        "n_X": int(n_sequence * M),
    }
