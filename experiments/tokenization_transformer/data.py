"""Data preparation for the Markov tokenization-transformer experiment.

For a given seed, generates one binary Markov-K sequence Y and tokenizes it
for each V in V_list. V=2 means no tokenization (Y is used directly). All
tokenized sequences and their metadata are returned so they can be shared
across all (W, V) training runs within the same seed.
"""

import time

import numpy as np

from core.markov import generate_markov_sequence, sample_transition_matrix, stationary_dist
from tokenizer.bpe import BPETokenizer


def build_sequences(K, V_list, alpha, n_sequence, seed, tok_cfg):
    """Generate one Markov-K binary sequence and tokenize for each V.

    Parameters
    ----------
    K : int
        Markov order. Source alphabet is always binary (V_y = 2).
    V_list : list[int]
        Vocabulary sizes. V=2 means no tokenization.
    alpha : float
        Dirichlet concentration for sampling the transition matrix.
    n_sequence : int
        Length of the binary source sequence Y (in bits).
    seed : int
        Single source of randomness: controls sequence generation and all
        BPE training (chunk sampling, tie-breaks).
    tok_cfg : dict
        Tokenizer settings with keys: encoder, train_bits, chunk_size.

    Returns
    -------
    dict with:
        P      -- (2**K, 2) row-stochastic transition matrix (numpy).
        Y      -- (n_sequence,) uint8, raw binary source sequence.
        per_V  -- dict[int -> dict] mapping each requested V to:
                      Z          (n_Z,) int64, token sequence (= Y if V==2)
                      ratio      float, len(Y) / len(Z)
                      token_bits (actual_V,) int64, bit-length per token id
                      actual_V   int, realized vocabulary size (== V in practice)
                      tokenizer  BPETokenizer instance, or None if V==2
    """
    rng = np.random.default_rng(seed)
    P = sample_transition_matrix(2, K, rng, alpha=alpha)
    print(f"  [seed={seed}] generating sequence (n={n_sequence:,}, K={K}) ...", flush=True)
    t0 = time.time()
    Y = generate_markov_sequence(P, 2, K, n_sequence, rng).astype(np.uint8)
    print(f"  [seed={seed}] sequence done ({time.time() - t0:.1f}s)", flush=True)
    pi = stationary_dist(P, 2, K)
    H = -float(np.sum(pi[:, None] * np.where(P > 0, P * np.log2(np.where(P > 0, P, 1.0)), 0.0)))
    print(f"  [seed={seed}] theoretical entropy rate: {H:.6f} bits/symbol", flush=True)

    encoder = tok_cfg.get("encoder", "greedy")
    train_bits = tok_cfg.get("train_bits", None)
    chunk_size = int(tok_cfg.get("chunk_size", 8192))

    per_V = {}
    for V in V_list:
        if V == 2:
            print(f"  [seed={seed}] V=2: no tokenization (ratio=1.0)", flush=True)
            per_V[V] = {
                "Z": Y.astype(np.int64),
                "ratio": 1.0,
                "token_bits": np.array([1, 1], dtype=np.int64),
                "actual_V": 2,
                "tokenizer": None,
            }
        else:
            print(f"  [seed={seed}] V={V}: training BPE tokenizer ...", flush=True)
            t1 = time.time()
            tok = BPETokenizer.train(
                Y,
                vocab_size=int(V),
                encoder=encoder,
                seed=int(seed),
                train_bits=None if train_bits is None else int(train_bits),
                chunk_size=chunk_size,
            )
            print(f"  [seed={seed}] V={V}: encoding sequence ...", flush=True)
            Z = tok.encode(Y).astype(np.int64)
            ratio = float(Y.size) / float(Z.size)
            print(
                f"  [seed={seed}] V={V}: done ({time.time() - t1:.1f}s)  "
                f"len(Z)={Z.size:,}  ratio={ratio:.4f}",
                flush=True,
            )
            per_V[V] = {
                "Z": Z,
                "ratio": ratio,
                "token_bits": tok.token_bit_lengths().astype(np.int64),
                "actual_V": int(tok.vocab_size),
                "tokenizer": tok,
            }

    return {"P": P, "Y": Y, "entropy_rate": H, "per_V": per_V}
