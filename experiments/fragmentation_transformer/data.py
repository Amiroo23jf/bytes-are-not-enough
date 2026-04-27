"""Streaming data setup for the Theorem 1 transformer experiment.

Generates one long Y sequence from the existing Markov-K source, fragments it
to X via the existing MSB-first bijection, and splits both into B contiguous
non-overlapping streams that are stacked along the batch dimension.
"""

import numpy as np
import torch

from core.markov import (
    fragment,
    generate_markov_sequence,
    sample_transition_matrix,
)


def build_streams(M, K, W, alpha, n_sequence, batch_size, seed, device):
    """Sample a Markov-K source and prepare the stream tensors.

    Returns a dict with:
        P                -- (V_y**K, V_y) row-stochastic transition matrix (numpy).
        Y_streams        -- (B, stream_len_Y) int64 tensor on `device`.
        X_streams        -- (B, stream_len_X) int64 tensor on `device`.
        V_y              -- int, alphabet size 2**M.
        stream_len_Y     -- int, number of Y-symbols per stream.
        stream_len_X     -- int, number of X-bits per stream (= stream_len_Y * M).
        S_Y              -- int, total Y-model iterations per seed (= stream_len_Y - W).
        S_X              -- int, total X-model iterations per seed (= stream_len_X - M*W).
    """
    V_y = 2 ** M
    rng = np.random.default_rng(seed)
    P = sample_transition_matrix(V_y, K, rng, alpha=alpha)
    Y = generate_markov_sequence(P, V_y, K, n_sequence, rng)
    X = fragment(Y, M)

    stream_len_Y = n_sequence // batch_size
    Y_used = stream_len_Y * batch_size
    Y_streams_np = Y[:Y_used].reshape(batch_size, stream_len_Y).astype(np.int64)

    stream_len_X = stream_len_Y * M
    X_used = stream_len_X * batch_size
    X_streams_np = X[:X_used].reshape(batch_size, stream_len_X).astype(np.int64)

    Y_streams = torch.from_numpy(Y_streams_np).to(device)
    X_streams = torch.from_numpy(X_streams_np).to(device)

    S_Y = stream_len_Y - W
    S_X = stream_len_X - M * W

    return {
        "P": P,
        "Y_streams": Y_streams,
        "X_streams": X_streams,
        "V_y": V_y,
        "stream_len_Y": stream_len_Y,
        "stream_len_X": stream_len_X,
        "S_Y": int(S_Y),
        "S_X": int(S_X),
    }
