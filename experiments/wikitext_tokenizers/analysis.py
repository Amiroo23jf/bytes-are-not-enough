"""Span-distribution analysis for the WikiText-103 tokenizer experiment.

For a token sequence with known per-token character lengths, computes:
    P(span(Z^w) = ws)
i.e. the distribution over the number of source characters covered by a
sliding window of w consecutive tokens.
"""

import numpy as np


def compute_span_distribution(char_lens, w):
    """Compute P(span = ws) for a window of w tokens.

    Parameters
    ----------
    char_lens : np.ndarray (n,) int32
        Per-position character coverage: how many source characters token i spans.
    w : int
        Window size in tokens.

    Returns
    -------
    pmf      : np.ndarray float32  -- P(span = ws_values[i])
    cdf      : np.ndarray float32  -- cumulative sum of pmf
    ws_values: np.ndarray int32    -- the ws axis (min_span .. max_span)
    mean_span: float
    """
    n = len(char_lens)
    if n < w:
        raise ValueError(f"Sequence length {n} < window w={w}")

    # Fast path: all tokens have identical length (e.g. character tokenizer).
    # Every window spans exactly cval * w characters — distribution is a delta.
    cmin, cmax = int(char_lens.min()), int(char_lens.max())
    if cmin == cmax:
        val = cmin * w
        return (
            np.array([1.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([val], dtype=np.int32),
            float(val),
        )

    # Prefix sums in int32 (safe: total_chars ≤ ~542M < 2^31 for WikiText-103).
    cs = np.empty(n + 1, dtype=np.int32)
    cs[0] = 0
    np.cumsum(char_lens, out=cs[1:])

    # Chunked histogram: never materialise the full spans array (~n × 4 bytes).
    # Each chunk is 20M elements = 80 MB at a time.
    _CHUNK = 20_000_000
    upper = cmax * w  # tight upper bound on any window span
    counts = np.zeros(upper + 1, dtype=np.int64)
    for start in range(0, n - w + 1, _CHUNK):
        end = min(start + _CHUNK, n - w + 1)
        chunk = cs[start + w : end + w] - cs[start:end]  # int32, ~80 MB
        counts += np.bincount(chunk, minlength=upper + 1)

    del cs  # free prefix-sum array before building output

    nonzero = np.flatnonzero(counts)
    ws_values = nonzero.astype(np.int32)
    pmf = (counts[nonzero] / counts[nonzero].sum()).astype(np.float32)
    cdf = np.cumsum(pmf).astype(np.float32)
    mean_span = float((ws_values.astype(np.float64) * pmf).sum())

    return pmf, cdf, ws_values, mean_span
