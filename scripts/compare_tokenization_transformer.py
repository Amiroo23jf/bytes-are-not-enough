"""Compare two tokenization_transformer configs from saved NPZ logs.

Produces two figures comparing:
  - W=4,  V=20  (blue,   #378ADD)
  - W=12, V=2   (orange, #D85A30)

Output
------
  outputs/tokenization_transformer/figures/compare_W4V20_vs_W12V2_steps.{ext}
  outputs/tokenization_transformer/figures/compare_W4V20_vs_W12V2_bits.{ext}

Usage
-----
  python scripts/compare_tokenization_transformer.py          # PDF
  python scripts/compare_tokenization_transformer.py --png
  python scripts/compare_tokenization_transformer.py --both
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT_ROOT = os.path.join(ROOT, "outputs", "tokenization_transformer")
LOG_DIR  = os.path.join(OUT_ROOT, "logs")
FIG_DIR  = os.path.join(OUT_ROOT, "figures")
SUMMARY  = os.path.join(OUT_ROOT, "summary.json")

CONFIGS = [
    dict(V=20, W=4,  color="#378ADD", label=r"$W=4,\ V=20$"),
    dict(V=2,  W=12, color="#D85A30", label=r"$W=12,\ V=2$"),
]

SMOOTH     = 500
HALF_COL_W = 4.875
HALF_COL_H = 3.3

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         8,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "lines.linewidth":   1.5,
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})


def _smooth(arr, k):
    if k <= 1:
        return arr.astype(np.float32), 0
    cs = np.concatenate([[0.0], np.cumsum(arr, dtype=np.float64)])
    return ((cs[k:] - cs[:-k]) / k).astype(np.float32), k - 1


def _si(x, _pos=None):
    if abs(x) >= 1e6:
        v = x / 1e6
        return f"{v:.0f}M" if v % 1 == 0 else f"{v:.1f}M"
    if abs(x) >= 1e3:
        v = x / 1e3
        return f"{v:.0f}K" if v % 1 == 0 else f"{v:.1f}K"
    return f"{x:.0f}"


def _save(fig, path, fmt, dpi=300):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi if fmt == "png" else None)
    plt.close(fig)
    print(f"  Saved {path}")


def load_entropy_rate():
    if not os.path.isfile(SUMMARY):
        return None
    with open(SUMMARY) as f:
        return float(json.load(f)[0]["entropy_rate_mean"])


def load_config(V, W):
    path = os.path.join(LOG_DIR, f"markov_V{V}_W{W}_seed0.npz")
    if not os.path.isfile(path):
        sys.exit(f"Log not found: {path}")
    d = np.load(path)
    loss = d["loss_per_iter"].astype(np.float64)
    bits = np.cumsum(d["bits_seen_per_iter"].astype(np.float64))
    return loss, bits


def plot_steps(H, fmt):
    fig, ax = plt.subplots(figsize=(HALF_COL_W, HALF_COL_H))
    for cfg in CONFIGS:
        loss, _ = load_config(cfg["V"], cfg["W"])
        smoothed, off = _smooth(loss, SMOOTH)
        ax.plot(np.arange(off + 1, len(loss) + 1), smoothed,
                color=cfg["color"], label=cfg["label"])
    if H is not None:
        ax.axhline(H, color="black", linewidth=0.8, linestyle="--", label=r"$H$")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss (bits per source symbol)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_si))
    ax.grid(linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.85, edgecolor="none", handlelength=1.4)
    fig.tight_layout(pad=0.4)
    _save(fig, os.path.join(FIG_DIR, f"compare_W4V20_vs_W12V2_steps.{fmt}"), fmt)


def plot_bits(H, fmt):
    fig, ax = plt.subplots(figsize=(HALF_COL_W, HALF_COL_H))

    # Truncate to the shorter run.
    all_bits = []
    for cfg in CONFIGS:
        _, bits = load_config(cfg["V"], cfg["W"])
        all_bits.append(bits)
    bits_max = min(b[-1] for b in all_bits)

    for cfg, bits in zip(CONFIGS, all_bits):
        loss, _ = load_config(cfg["V"], cfg["W"])
        smoothed, off = _smooth(loss, SMOOTH)
        bits_sm = bits[off:]
        mask = bits_sm <= bits_max * 1.001
        ax.plot(bits_sm[mask], smoothed[mask],
                color=cfg["color"], label=cfg["label"])
    if H is not None:
        ax.axhline(H, color="black", linewidth=0.8, linestyle="--", label=r"$H$")
    ax.set_xlabel("Source Symbols Processed")
    ax.set_ylabel("Loss (bits per source symbol)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_si))
    ax.grid(linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.85, edgecolor="none", handlelength=1.4)
    fig.tight_layout(pad=0.4)
    _save(fig, os.path.join(FIG_DIR, f"compare_W4V20_vs_W12V2_bits.{fmt}"), fmt)


def main():
    both = "--both" in sys.argv
    fmt  = "png" if "--png" in sys.argv else "pdf"
    fmts = ["pdf", "png"] if both else [fmt]

    H = load_entropy_rate()

    for fmt in fmts:
        plot_steps(H, fmt)
        plot_bits(H, fmt)

    print("Done.")


if __name__ == "__main__":
    main()
