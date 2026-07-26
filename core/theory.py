"""Theoretical quantity computation for the fragmentation penalty.

All entropy and mutual information values are in bits (log base 2).

The key object is the joint distribution p(theta, ctx, b) over
    theta: uniform phase in {0, ..., M - 1}
    ctx:   K*M-bit context visible to a bit-level predictor
    b:     next bit to predict.

From this table we derive:
    L_X_stat -- loss of a phase-blind (stationary) binary predictor
    L_X_orc  -- loss of an oracle that knows theta
    MI = L_X_stat - L_X_orc  -- the fragmentation penalty (per X token)
"""

from collections import defaultdict

import numpy as np

from .markov import stationary_dist

_EPS = 1e-15


def H_Y_theoretical(P, V_y, k):
    """Conditional entropy H(Y_i | Y_{i-k}^{i-1}) in bits per Y symbol.

    Computed from the true transition matrix and its stationary distribution:
        H = sum_s pi(s) * H(P[s, :]).
    """
    pi = stationary_dist(P, V_y, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.log2(np.maximum(P, _EPS))
    row_entropy = -np.sum(P * logP, axis=1)
    return float(np.sum(pi * row_entropy))


def estimate_marginal_Y(Y, K, V_y):
    """Empirical joint distribution over (K+1)-symbol tuples from sequence Y.

    Returns a dict mapping tuple -> probability, containing only tuples with
    positive count. Uses a rolling-hash integer encoding of each length-(K+1)
    window + np.bincount for fully vectorized counting.
    """
    Y = np.asarray(Y, dtype=np.int64)
    n = len(Y)
    win = K + 1
    if n < win:
        return {}

    # Integer-encode every length-win window via Horner on base V_y.
    window_ints = np.zeros(n - win + 1, dtype=np.int64)
    for j in range(win):
        window_ints = window_ints * V_y + Y[j : n - win + 1 + j]

    n_tuples = V_y ** win
    bins = np.bincount(window_ints, minlength=n_tuples)
    total = bins.sum()
    inv = 1.0 / total

    result = {}
    nz = np.flatnonzero(bins)
    for u in nz.tolist():
        tup = [0] * win
        x = u
        for j in range(win - 1, -1, -1):
            tup[j] = x % V_y
            x //= V_y
        result[tuple(tup)] = bins[u] * inv
    return result


def _symbol_to_tokens(y, M, boundary_token=None):
    """Return the per-symbol token tuple for y: M bits (MSB first), plus
    `boundary_token` appended if given."""
    bits = tuple((y >> (M - 1 - j)) & 1 for j in range(M))
    if boundary_token is None:
        return bits
    return bits + (boundary_token,)


def build_joint_table(marginal_Y, V_y, K, M, boundary_token=None):
    """Build the joint table p(theta, ctx, b) from the marginal over (K+1)-tuples.

    Each Y symbol expands to a block of `block_len` tokens: M bits (MSB
    first), plus one trailing `boundary_token` if given (block_len = M + 1);
    otherwise block_len = M.

    For each (K+1)-tuple (y_0, ..., y_K) with probability p:
      1. Flatten to (K+1)*block_len tokens.
      2. For each theta in {0, ..., block_len-1}:
           ctx = tokens[theta : theta + K*block_len]
           b   = tokens[theta + K*block_len]
           table[(theta, ctx, b)] += p * (1/block_len)

    The returned table sums to 1 over all (theta, ctx, b) entries.
    """
    block_len = M + (1 if boundary_token is not None else 0)
    table = defaultdict(float)
    w = 1.0 / block_len
    for tup, p in marginal_Y.items():
        tokens = []
        for y in tup:
            tokens.extend(_symbol_to_tokens(int(y), M, boundary_token))
        K_block = K * block_len
        for theta in range(block_len):
            ctx = tuple(tokens[theta : theta + K_block])
            b = tokens[theta + K_block]
            table[(theta, ctx, b)] += p * w
    return dict(table)


def compute_losses_from_table(table, M, vocab_size=2):
    """Compute L_X_stat, L_X_orc, and MI from the joint table p(theta, ctx, b).

    Stationary predictor (phase-blind):
        p(ctx, b) = sum_theta p(theta, ctx, b)
        p(ctx)    = sum_b p(ctx, b)
        H(b|ctx)  = -sum_b p(b|ctx) log2 p(b|ctx)
        L_X_stat  = sum_ctx p(ctx) H(b|ctx)

    Oracle predictor (phase known):
        p(theta, ctx)   = sum_b p(theta, ctx, b)
        H(b|theta, ctx) = -sum_b p(b|theta, ctx) log2 p(b|theta, ctx)
        L_X_orc         = sum_{theta, ctx} p(theta, ctx) H(b|theta, ctx)

    `M` is unused (kept for call-site compatibility with existing callers);
    `vocab_size` sets the range of next-token values `b` ranges over (2 for
    plain binary fragmentation, 3 when a boundary token is included).

    Returns dict with keys L_X_stat, L_X_orc, MI (all in bits per X token).
    """
    # Aggregate over theta for the stationary predictor.
    p_ctx_b = defaultdict(float)           # (ctx, b) -> prob
    p_theta_ctx_b = defaultdict(float)     # already the table itself
    for (theta, ctx, b), p in table.items():
        p_ctx_b[(ctx, b)] += p
        p_theta_ctx_b[(theta, ctx, b)] += p

    p_ctx = defaultdict(float)
    for (ctx, b), p in p_ctx_b.items():
        p_ctx[ctx] += p

    L_X_stat = 0.0
    for ctx, pc in p_ctx.items():
        if pc < _EPS:
            continue
        H_bc = 0.0
        for b in range(vocab_size):
            pcb = p_ctx_b.get((ctx, b), 0.0)
            if pcb <= 0.0:
                continue
            pb_given = pcb / pc
            H_bc -= pb_given * np.log2(max(pb_given, _EPS))
        L_X_stat += pc * H_bc

    p_theta_ctx = defaultdict(float)
    for (theta, ctx, b), p in p_theta_ctx_b.items():
        p_theta_ctx[(theta, ctx)] += p

    L_X_orc = 0.0
    for (theta, ctx), ptc in p_theta_ctx.items():
        if ptc < _EPS:
            continue
        H_btc = 0.0
        for b in range(vocab_size):
            ptcb = p_theta_ctx_b.get((theta, ctx, b), 0.0)
            if ptcb <= 0.0:
                continue
            pb_given = ptcb / ptc
            H_btc -= pb_given * np.log2(max(pb_given, _EPS))
        L_X_orc += ptc * H_btc

    MI = L_X_stat - L_X_orc
    return {
        "L_X_stat": float(L_X_stat),
        "L_X_orc": float(L_X_orc),
        "MI": float(MI),
    }
