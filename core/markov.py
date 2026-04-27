"""Markov source utilities.

Provides sampling and analysis routines for k-th order Markov chains over an
alphabet of size V_y, and a bit-level fragmentation operator that maps each
V_y-ary symbol (V_y = 2^M) to M binary tokens.

State encoding convention (big-endian):
    state s encodes k-tuple (y_0, ..., y_{k-1}) as
        s = sum_{j=0}^{k-1} y_j * V_y^(k-1-j)
"""

import numpy as np


def stationary_dist(P, V_y, k):
    """Compute the stationary distribution over k-tuple states via power iteration.

    Parameters
    ----------
    P : np.ndarray of shape (V_y**k, V_y)
        Row-stochastic transition matrix. Row s gives p(y | state s).
    V_y : int
        Alphabet size of Y.
    k : int
        Markov order.

    Returns
    -------
    pi : np.ndarray of shape (V_y**k,)
        Stationary distribution over the V_y**k k-tuple states.
    """
    n_states = V_y ** k
    # Induced transition matrix T over k-tuple states:
    # from state s = (y_0, ..., y_{k-1}), emitting y_k moves to
    # state s' = (y_1, ..., y_k). With big-endian encoding:
    #   s' = (s mod V_y^(k-1)) * V_y + y_k
    # and the transition probability is P[s, y_k].
    T = np.zeros((n_states, n_states), dtype=np.float64)
    if k == 0:
        # Degenerate case: no state, stationary is the row itself.
        return P.reshape(-1).copy()
    tail_mod = V_y ** (k - 1)
    for s in range(n_states):
        base = (s % tail_mod) * V_y
        for y in range(V_y):
            s_next = base + y
            T[s, s_next] = P[s, y]

    pi = np.full(n_states, 1.0 / n_states, dtype=np.float64)
    for _ in range(50000):
        pi_new = pi @ T
        pi_new /= pi_new.sum()
        if np.max(np.abs(pi_new - pi)) < 1e-12:
            pi = pi_new
            break
        pi = pi_new
    return pi


def sample_transition_matrix(V_y, k, rng, alpha=0.5):
    """Sample a random row-stochastic transition matrix.

    Each row is drawn independently from Dirichlet(alpha * ones(V_y)).
    """
    n_states = V_y ** k
    return rng.dirichlet(alpha * np.ones(V_y), size=n_states).astype(np.float64)


def generate_markov_sequence(P, V_y, k, n, rng):
    """Generate a length-n sequence from the k-th order Markov chain P.

    Starts from a state sampled from the stationary distribution of P.
    """
    pi = stationary_dist(P, V_y, k)
    Y = np.empty(n, dtype=np.int32)

    if k == 0:
        # Memoryless source: every symbol drawn i.i.d. from P[0].
        probs = P[0]
        Y[:] = rng.choice(V_y, size=n, p=probs)
        return Y

    # Decode initial state index into k symbols (big-endian).
    s0 = rng.choice(len(pi), p=pi)
    init = np.empty(k, dtype=np.int32)
    tmp = s0
    for j in range(k - 1, -1, -1):
        init[j] = tmp % V_y
        tmp //= V_y

    if n <= k:
        Y[:] = init[:n]
        return Y

    Y[:k] = init
    state = int(s0)
    tail_mod = V_y ** (k - 1)
    # Draw all needed uniforms once for speed; decide transitions via cumsum.
    u = rng.random(n - k)
    cum = np.cumsum(P, axis=1)
    for i in range(k, n):
        y = int(np.searchsorted(cum[state], u[i - k]))
        if y >= V_y:
            y = V_y - 1
        Y[i] = y
        state = (state % tail_mod) * V_y + y
    return Y


def fragment(Y, M):
    """Fragment each V_y-ary symbol into M binary tokens (MSB first).

    bit j of symbol y = (y >> (M - 1 - j)) & 1.

    Parameters
    ----------
    Y : np.ndarray of shape (n,), int32
        Symbols in {0, ..., 2**M - 1}.
    M : int
        Bits per symbol.

    Returns
    -------
    X : np.ndarray of shape (n * M,), int32
        Binary tokens in {0, 1}.
    """
    Y = np.asarray(Y, dtype=np.int64)
    shifts = np.arange(M - 1, -1, -1, dtype=np.int64)
    bits = ((Y[:, None] >> shifts[None, :]) & 1).astype(np.int32)
    return bits.reshape(-1)
