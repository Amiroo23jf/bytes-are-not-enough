"""Tests for the optional boundary-token fragmentation mode.

Covers the two guarantees the rebuttal experiment depends on:
  - flag OFF: output is byte-for-byte identical to the pre-existing
    fragmentation (no boundary token, vocab_size 2, block length M).
  - flag ON: a boundary token (id 2, the next free symbol above {0, 1}) is
    inserted exactly once every M+1 positions, marking the true block
    boundary, and the vocabulary/context bookkeeping (V_x, block_len_x, n_X)
    is updated consistently.
"""

from __future__ import annotations

import numpy as np

from core.markov import fragment, generate_markov_sequence, sample_transition_matrix
from core.theory import build_joint_table, compute_losses_from_table, estimate_marginal_Y
from experiments.fragmentation_transformer.data import build_streams
from experiments.fragmentation_transformer.theory import compute_asymptotes


def _sample_Y(M, K, n, seed):
    V_y = 2 ** M
    rng = np.random.default_rng(seed)
    P = sample_transition_matrix(V_y, K, rng, alpha=0.5)
    Y = generate_markov_sequence(P, V_y, K, n, rng)
    return P, Y, V_y


def test_fragment_boundary_off_is_unchanged():
    _, Y, _ = _sample_Y(M=3, K=2, n=500, seed=0)
    baseline = fragment(Y, 3)
    assert np.array_equal(fragment(Y, 3, boundary_token=None), baseline)
    # default arg matches explicit None
    assert np.array_equal(fragment(Y, 3), baseline)


def test_fragment_boundary_on_marks_every_block():
    M = 3
    _, Y, _ = _sample_Y(M=M, K=2, n=500, seed=0)
    X = fragment(Y, M, boundary_token=2)

    assert X.shape[0] == Y.shape[0] * (M + 1)
    blocks = X.reshape(-1, M + 1)
    # Boundary token appears exactly once every M+1 positions, at block end.
    assert np.all(blocks[:, M] == 2)
    # No boundary token leaks into the content positions.
    assert np.all(blocks[:, :M] != 2)
    # Content bits are untouched by the insertion.
    assert np.array_equal(blocks[:, :M].reshape(-1), fragment(Y, M))


def test_build_streams_off_matches_baseline_fragment():
    M, K, alpha, n_sequence, seed = 3, 2, 0.5, 2000, 0
    streams = build_streams(M, K, alpha, n_sequence, seed)  # boundary defaults False

    assert streams["boundary"] is False
    assert streams["V_x"] == 2
    assert streams["block_len_x"] == M
    assert streams["n_X"] == n_sequence * M
    # Byte-for-byte identical to calling the pre-existing fragment() directly
    # on the same Y.
    assert np.array_equal(streams["X"], fragment(streams["Y"], M))


def test_build_streams_boundary_on():
    M, K, alpha, n_sequence, seed = 3, 2, 0.5, 2000, 0
    streams = build_streams(M, K, alpha, n_sequence, seed, boundary=True)

    assert streams["boundary"] is True
    assert streams["V_x"] == 3
    assert streams["block_len_x"] == M + 1
    assert streams["n_X"] == n_sequence * (M + 1)

    X = streams["X"]
    blocks = X.reshape(-1, M + 1)
    assert np.all(blocks[:, M] == 2)
    assert np.array_equal(blocks[:, :M].reshape(-1), fragment(streams["Y"], M))


def test_build_streams_same_seed_same_Y_regardless_of_boundary_flag():
    # Turning boundary marking on/off must not perturb the underlying source Y.
    M, K, alpha, n_sequence, seed = 3, 2, 0.5, 2000, 0
    s_off = build_streams(M, K, alpha, n_sequence, seed, boundary=False)
    s_on = build_streams(M, K, alpha, n_sequence, seed, boundary=True)
    assert np.array_equal(s_off["Y"], s_on["Y"])


def test_theory_phase_MI_vanishes_with_boundary_marking():
    # The core empirical claim of the rebuttal: once the true boundary is
    # marked, a finite-context predictor can always recover its phase, so the
    # phase-mutual-information term in the Theorem 1 decomposition is exactly
    # zero (not just small) -- it's a property of the joint table itself, not
    # a finite-sample effect.
    M, K, W = 3, 2, 3
    rng = np.random.default_rng(0)
    P = sample_transition_matrix(2 ** M, K, rng, alpha=0.5)
    Y = generate_markov_sequence(P, 2 ** M, K, 50_000, rng)
    marginal = estimate_marginal_Y(Y, W, 2 ** M)

    table_off = build_joint_table(marginal, 2 ** M, W, M)
    losses_off = compute_losses_from_table(table_off, M)
    assert losses_off["MI"] > 0  # phase ambiguity present without marking

    table_on = build_joint_table(marginal, 2 ** M, W, M, boundary_token=2)
    losses_on = compute_losses_from_table(table_on, M, vocab_size=3)
    assert losses_on["MI"] == 0.0  # ambiguity eliminated exactly


def test_compute_asymptotes_boundary_flag_defaults_to_off_baseline():
    M, K, W = 3, 2, 3
    rng = np.random.default_rng(0)
    P = sample_transition_matrix(2 ** M, K, rng, alpha=0.5)

    a_default = compute_asymptotes(P=P, V_y=2 ** M, K=K, W=W, M=M,
                                    n_marginal=20_000, rng_seed=1)
    a_explicit_off = compute_asymptotes(P=P, V_y=2 ** M, K=K, W=W, M=M,
                                         n_marginal=20_000, rng_seed=1,
                                         boundary=False)
    assert a_default == a_explicit_off

    a_on = compute_asymptotes(P=P, V_y=2 ** M, K=K, W=W, M=M,
                               n_marginal=20_000, rng_seed=1, boundary=True)
    assert a_on["phase_MI_per_Y"] == 0.0
