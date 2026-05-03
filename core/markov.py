"""Markov source utilities.

Provides sampling and analysis routines for k-th order Markov chains over an
alphabet of size V_y, and a bit-level fragmentation operator that maps each
V_y-ary symbol (V_y = 2^M) to M binary tokens.

State encoding convention (big-endian):
    state s encodes k-tuple (y_0, ..., y_{k-1}) as
        s = sum_{j=0}^{k-1} y_j * V_y^(k-1-j)
"""

import numpy as np

try:
    import numba
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


# ---------------------------------------------------------------------------
# Numba-accelerated inner loops (compiled on first call, cached to disk)
# ---------------------------------------------------------------------------

if _HAVE_NUMBA:
    @numba.jit(nopython=True, cache=True)
    def _markov_loop(cum, u, Y, k, state, tail_mod, V_y):
        """Fill Y[k:] for a general V_y-ary Markov chain."""
        n = len(Y)
        for i in range(k, n):
            row = cum[state]
            u_i = u[i - k]
            y = V_y - 1
            for j in range(V_y - 1):
                if u_i <= row[j]:
                    y = j
                    break
            Y[i] = y
            state = (state % tail_mod) * V_y + y

    @numba.jit(nopython=True, cache=True)
    def _markov_loop_binary(p0, u, Y, k, state, tail_mod):
        """Fill Y[k:] for a binary Markov chain (avoids searchsorted)."""
        n = len(Y)
        for i in range(k, n):
            bit = 1 if u[i - k] > p0[state] else 0
            Y[i] = bit
            state = (state % tail_mod) * 2 + bit


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
    if k == 0:
        return P.reshape(-1).copy()

    n_states = V_y ** k
    tail_mod = V_y ** (k - 1)

    # Efficient O(V_y * n_states) per-iteration algorithm.
    #
    # Each successor state s' = (s % tail_mod) * V_y + y_k has exactly V_y
    # predecessors: s = y0 * tail_mod + (s' // V_y) for y0 in {0..V_y-1},
    # with transition probability P[s, s' % V_y].
    #
    # So pi'[s'] = sum_{y0} pi[y0 * tail_mod + s'//V_y] * P[y0*tail_mod + s'//V_y, s'%V_y]
    #
    # This avoids building the dense (n_states, n_states) T matrix.
    s_prime = np.arange(n_states, dtype=np.int64)
    y_k = s_prime % V_y          # emitted symbol that leads into s'
    suffix = s_prime // V_y      # = s % tail_mod (shared suffix of predecessors)

    # preds[y0, s'] = y0 * tail_mod + suffix[s']  -- predecessor states
    preds = (
        np.arange(V_y, dtype=np.int64)[:, None] * tail_mod + suffix[None, :]
    )  # (V_y, n_states)
    trans = P[preds, y_k[None, :]]  # (V_y, n_states)  transition probs

    pi = np.full(n_states, 1.0 / n_states, dtype=np.float64)
    for _ in range(50000):
        pi_new = (pi[preds] * trans).sum(axis=0)
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
    Uses Numba JIT if available, otherwise falls back to a pure-Python loop.
    """
    pi = stationary_dist(P, V_y, k)
    Y = np.empty(n, dtype=np.int32)

    if k == 0:
        Y[:] = rng.choice(V_y, size=n, p=P[0])
        return Y

    # Decode initial state into k symbols (big-endian).
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
    u = rng.random(n - k)

    if _HAVE_NUMBA:
        if V_y == 2:
            _markov_loop_binary(
                P[:, 0].astype(np.float64),
                u.astype(np.float64),
                Y, k, state, tail_mod,
            )
        else:
            cum = np.cumsum(P, axis=1).astype(np.float64)
            _markov_loop(cum, u.astype(np.float64), Y, k, state, tail_mod, V_y)
    else:
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
