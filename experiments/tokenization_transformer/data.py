"""Streaming data setup for the Theorem 3 tokenization-transformer experiment.

Generates one long binary Markov-K source Y (V_y = 2), trains a BPE
tokenizer on Y, encodes Y to a token sequence Z, then splits both Y and Z
into B contiguous non-overlapping streams stacked along the batch axis.

The two stream tensors are consumed by:
    - the Y-LM (vocab=2, context=W bits)
    - the Z-LM (vocab=V, context=W tokens)

The length ratio `mean_token_bits = len(Y) / len(Z)` is what makes the
Z-LM's per-token loss comparable to the Y-LM's per-bit loss after dividing.
"""

import numpy as np
import torch

from core.markov import generate_markov_sequence, sample_transition_matrix
from tokenizer.bpe import BPETokenizer


def build_streams(
    K,
    W,
    V,
    alpha,
    n_sequence,
    batch_size,
    seed,
    device,
    encoder="greedy",
    train_bits=None,
    chunk_size=8192,
):
    """Sample Y, train BPE, encode to Z, and reshape both into B streams.

    Parameters
    ----------
    K : int
        Order of the binary Markov source.
    W : int
        Transformer context length in tokens (same for Y- and Z-LMs).
    V : int
        BPE vocabulary size (>= 2).
    alpha : float
        Dirichlet concentration for sampling the transition matrix.
    n_sequence : int
        Length of the Y sequence (in bits).
    batch_size : int
        B; number of parallel streams stacked along dim 0.
    seed : int
        Source-of-randomness for the entire build (transition matrix,
        sequence, and tokenizer reproducibility).
    device : torch.device
        Where the stream tensors are placed.
    encoder : {"greedy", "priority"}
        Which encoder the trained `BPETokenizer` should use for `encode`.
    train_bits : Optional[int]
        If given, fit the BPE merge table on `~train_bits` randomly
        placed chunks of `chunk_size` bits drawn from Y, instead of
        scanning the full Y. The full Y is still encoded into Z.
    chunk_size : int
        Length of each BPE-training chunk in bits.

    Returns
    -------
    dict with:
        P                -- (2**K, 2) transition matrix (numpy).
        Y_streams        -- (B, stream_len_Y) int64 tensor on `device`.
        Z_streams        -- (B, stream_len_Z) int64 tensor on `device`.
        V                -- int, BPE vocab size (== tokenizer.vocab_size).
        V_y              -- int, source alphabet (always 2).
        mean_token_bits  -- float, len(Y) / len(Z).
        token_bits       -- (V,) int64; bit-length of each token id.
        tokenizer        -- the trained BPETokenizer instance.
        S_Y              -- int, total Y-LM iterations per seed (= stream_len_Y - W).
        S_Z              -- int, total Z-LM iterations per seed (= stream_len_Z - W).
        stream_len_Y     -- int.
        stream_len_Z     -- int.
        n_Y              -- int, len(Y) actually used (after batch trim).
        n_Z              -- int, len(Z) (full encoded sequence).
    """
    V_y = 2
    rng = np.random.default_rng(seed)

    P = sample_transition_matrix(V_y, K, rng, alpha=alpha)
    Y = generate_markov_sequence(P, V_y, K, n_sequence, rng).astype(np.uint8)

    tokenizer = BPETokenizer.train(
        Y,
        vocab_size=int(V),
        encoder=encoder,
        seed=int(seed),
        train_bits=None if train_bits is None else int(train_bits),
        chunk_size=int(chunk_size),
    )
    Z = tokenizer.encode(Y)
    token_bits = tokenizer.token_bit_lengths()

    if Z.size == 0:
        raise RuntimeError("BPE encoding produced an empty sequence.")
    mean_token_bits = float(Y.size) / float(Z.size)

    stream_len_Y = n_sequence // batch_size
    Y_used = stream_len_Y * batch_size
    Y_streams_np = Y[:Y_used].reshape(batch_size, stream_len_Y).astype(np.int64)

    stream_len_Z = Z.size // batch_size
    Z_used = stream_len_Z * batch_size
    Z_streams_np = Z[:Z_used].reshape(batch_size, stream_len_Z).astype(np.int64)

    Y_streams = torch.from_numpy(Y_streams_np).to(device)
    Z_streams = torch.from_numpy(Z_streams_np).to(device)

    S_Y = stream_len_Y - W
    S_Z = stream_len_Z - W
    if S_Y <= 0:
        raise ValueError(
            f"stream_len_Y={stream_len_Y} <= W={W}; increase n_sequence or "
            f"decrease batch_size/W."
        )
    if S_Z <= 0:
        raise ValueError(
            f"stream_len_Z={stream_len_Z} <= W={W}; the BPE compression ratio "
            f"shrank Z below batch_size*(W+1). Increase n_sequence or shrink V/W."
        )

    return {
        "P": P,
        "Y_streams": Y_streams,
        "Z_streams": Z_streams,
        "V": int(tokenizer.vocab_size),
        "V_y": V_y,
        "mean_token_bits": mean_token_bits,
        "token_bits": token_bits.astype(np.int64),
        "tokenizer": tokenizer,
        "S_Y": int(S_Y),
        "S_Z": int(S_Z),
        "stream_len_Y": int(stream_len_Y),
        "stream_len_Z": int(stream_len_Z),
        "n_Y": int(Y_used),
        "n_Z": int(Z.size),
    }
