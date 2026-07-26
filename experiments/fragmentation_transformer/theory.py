"""Theoretical asymptotes L_Y* and L_X* for the Theorem 1 experiment.

Thin wrapper around the existing pipeline in `core/theory.py`. Both quantities
are in bits per Y-symbol.

L_Y* = H(Y_{t+1} | Y_{t-W+1}..Y_t).
    When W >= K this equals the entropy rate, computed analytically from the
    transition matrix via `H_Y_theoretical`.
    TODO: for W < K this should be computed from the W-step conditional
    distribution of the Markov chain; not needed for the Theorem 1 main
    configurations (A, B, C all have W = K).

L_X* = phase-averaged H(b | K*M-bit ctx) * M, with K here replaced by W (the
    user-chosen window). Computed by:
        1. Sampling a long Y with the same transition matrix.
        2. Estimating the (W+1)-tuple marginal of Y.
        3. Building the joint table p(theta, ctx, b) at window W.
        4. L_X* = L_X_stat * M.
"""

import numpy as np

from core.markov import generate_markov_sequence
from core.theory import (
    H_Y_theoretical,
    build_joint_table,
    compute_losses_from_table,
    estimate_marginal_Y,
)


def compute_asymptotes(P, V_y, K, W, M, n_marginal, rng_seed, boundary=False):
    """Compute L_Y* and L_X* in bits per Y-symbol for a given transition matrix.

    `boundary`: if True, the joint table is built over (M+1)-token blocks
    (M content bits + a boundary token), matching `build_streams(boundary=True)`.
    """
    if W < K:
        raise NotImplementedError(
            "L_Y* computation for W < K is not yet implemented. "
            "Theorem 1 main configs all use W >= K."
        )
    L_Y_star = H_Y_theoretical(P, V_y, K)

    boundary_token = 2 if boundary else None
    vocab_size = 3 if boundary else 2
    block_len = M + (1 if boundary else 0)

    rng = np.random.default_rng(rng_seed)
    Y_long = generate_markov_sequence(P, V_y, K, n_marginal, rng)
    marginal = estimate_marginal_Y(Y_long, W, V_y)
    table = build_joint_table(marginal, V_y, W, M, boundary_token=boundary_token)
    losses = compute_losses_from_table(table, M, vocab_size=vocab_size)
    L_X_star = losses["L_X_stat"] * block_len
    L_X_orc = losses["L_X_orc"] * block_len
    return {
        "L_Y_star": float(L_Y_star),
        "L_X_star": float(L_X_star),
        "L_X_orc_star": float(L_X_orc),
        "phase_MI_per_Y": float(losses["MI"] * block_len),
    }
