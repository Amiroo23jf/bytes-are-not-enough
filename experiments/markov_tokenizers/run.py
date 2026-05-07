"""Span-distribution analysis for Markov-source BPE tokenizers.

For each V, generates a Markov-K binary sequence, trains a BPE tokenizer
(or uses Y directly for V=2), then computes P(span(Z^w) = w_s) for each
window size w in W_list.  span is measured in source bits.

Artifacts
---------
  outputs/markov_tokenizers/results/V{V}.npz
      pmf_W{w}, cdf_W{w}, ws_W{w}, mean_span_W{w}  for each w in W
      vocab_size, ratio, name, W_list
"""

import os
import time

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from experiments.tokenization_transformer.data import build_sequences       # noqa: E402
from experiments.wikitext_tokenizers.analysis import compute_span_distribution  # noqa: E402

OUT_ROOT    = os.path.join(ROOT, "outputs", "markov_tokenizers")
RESULTS_DIR = os.path.join(OUT_ROOT, "results")


def run(cfg, verbose=True):
    K          = int(cfg["K"])
    V_list     = [int(v) for v in cfg["V"]]
    W_list     = [int(w) for w in cfg["W"]]
    alpha      = float(cfg["alpha"])
    n_sequence = int(cfg["n_sequence"])
    seed       = int(cfg["seed"])
    tok_cfg    = cfg.get("tokenizer", {})

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if verbose:
        print(f"Generating Markov sequence (K={K}, n={n_sequence:,}, seed={seed}) ...",
              flush=True)
    data = build_sequences(K, V_list, alpha, n_sequence, seed, tok_cfg)

    for V in V_list:
        info       = data["per_V"][V]
        Z          = info["Z"]
        token_bits = info["token_bits"]   # (vocab_size,) int64: source bits per token id
        ratio      = info["ratio"]
        actual_V   = info["actual_V"]

        # Per-position source-bit lengths: how many bits does each token cover.
        char_lens = token_bits[Z].astype(np.int32)

        result = {
            "vocab_size": np.int32(actual_V),
            "ratio":      np.float32(ratio),
            "name":       f"V{V}",
            "W_list":     np.array(W_list, dtype=np.int32),
        }

        for W in W_list:
            if verbose:
                print(f"  V={actual_V:2d}  W={W:2d}: computing span distribution ...",
                      flush=True)
            t0 = time.time()
            pmf, cdf, ws_values, mean_span = compute_span_distribution(char_lens, W)
            if verbose:
                print(f"           done ({time.time() - t0:.1f}s)  "
                      f"mean_span={mean_span:.2f} bits", flush=True)
            result[f"pmf_W{W}"]       = pmf
            result[f"cdf_W{W}"]       = cdf
            result[f"ws_W{W}"]        = ws_values
            result[f"mean_span_W{W}"] = np.float32(mean_span)

        path = os.path.join(RESULTS_DIR, f"V{V}.npz")
        np.savez_compressed(path, **result)
        if verbose:
            print(f"  Saved {path}", flush=True)
