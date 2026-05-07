"""Plot span-distribution slack curves for the Markov tokenizer experiment.

For each window size W, produces one figure showing the scaled CDF
  CDF(w_s) × log2(V) / ratio
versus the source-bit span w_s, with one curve per V.

Colors are drawn from the same plasma palette used in
plot_tokenization_transformer.py, keyed by each V's rank in the sorted V list.

Output
------
  outputs/markov_tokenizers/figures/slack_W{w}.{ext}

Usage
-----
  python scripts/plot_markov_tokenizers.py           # PDF (default)
  python scripts/plot_markov_tokenizers.py --png
  python scripts/plot_markov_tokenizers.py --both
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RESULTS_DIR = os.path.join(ROOT, "outputs", "markov_tokenizers", "results")
FIG_DIR     = os.path.join(ROOT, "outputs", "markov_tokenizers", "figures")

HALF_COL_W = 4.875
HALF_COL_H = 3.3

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         8,
    "axes.labelsize":    8,
    "axes.titlesize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   6.5,
    "lines.linewidth":   1.5,
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})


def _palette(n, vmax=0.75):
    cmap = matplotlib.colormaps["plasma"]
    return [cmap(vmax * i / max(n - 1, 1)) for i in range(n)]


def _save(fig, path, fmt, dpi=300):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi if fmt == "png" else None)
    plt.close(fig)
    print(f"  Saved {path}")


def load_all():
    """Return list of (V, npz_data) sorted by V."""
    entries = []
    for fname in os.listdir(RESULTS_DIR):
        if not fname.startswith("V") or not fname.endswith(".npz"):
            continue
        V = int(fname[1:-4])
        entries.append((V, np.load(os.path.join(RESULTS_DIR, fname), allow_pickle=True)))
    return sorted(entries, key=lambda x: x[0])


def plot_slack_for_W(W, entries, colors, fmt):
    fig, ax = plt.subplots(figsize=(HALF_COL_W, HALF_COL_H))

    all_ws = []
    for V, d in entries:
        key_ws = f"ws_W{W}"
        if key_ws in d.files:
            all_ws.append(d[key_ws])

    if not all_ws:
        plt.close(fig)
        return

    x_min = max(1, W * 0.7)
    # Clip each curve at its own 95th-percentile span, then extend flat to
    # x_max.  This removes the heavy tail while making all curves visually
    # constant at the right edge of the plot.
    p95_map = {}   # V -> (x_95, idx_95)
    for V, d in entries:
        key_ws  = f"ws_W{W}"
        key_cdf = f"cdf_W{W}"
        if key_ws not in d.files or key_cdf not in d.files:
            continue
        ws_v  = d[key_ws].astype(np.float64)
        cdf_v = d[key_cdf].astype(np.float64)
        idx = min(np.searchsorted(cdf_v, 0.95), len(ws_v) - 1)
        p95_map[V] = (float(ws_v[idx]), idx)
    x_max = max(x for x, _ in p95_map.values()) * 1.05 if p95_map else float(max(ws[-1] for ws in all_ws))

    for color, (V, d) in zip(colors, entries):
        key_ws  = f"ws_W{W}"
        key_cdf = f"cdf_W{W}"
        if key_ws not in d.files or key_cdf not in d.files:
            continue
        ws    = d[key_ws].astype(np.float64)
        cdf   = d[key_cdf].astype(np.float64)
        ratio = float(d["ratio"])
        scale = np.log2(float(d["vocab_size"])) / ratio
        label = f"$V={V}$"

        # Delta distribution (V=2): step at exactly W source bits.
        if len(ws) == 1:
            ws_plot  = np.array([0.0, ws[0], ws[0], x_max])
            sc_plot  = np.array([0.0, 0.0,   scale,  scale])
            ax.plot(ws_plot, sc_plot, color=color, label=label,
                    linewidth=1.8, zorder=3)
        else:
            _, idx95 = p95_map[V]
            ws_clip = ws[:idx95 + 1]
            sc_clip = cdf[:idx95 + 1] * scale
            ax.plot(np.append(ws_clip, x_max),
                    np.append(sc_clip, sc_clip[-1]),
                    color=color, label=label)

    ax.set_xlim(x_min, x_max)
    ax.axvline(W, color="black", linewidth=0.9, linestyle="--")
    ax.text(W * 1.02, 0.04, f"$w={W}$",
            ha="left", va="bottom", fontsize=6.5,
            transform=ax.get_xaxis_transform(),
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.75))

    ax.set_xlabel(r"Span $w_s$ (source bits)")
    ax.set_ylabel(
        r"$\dfrac{\varepsilon(w_s) \cdot \log_2 V}{\mathbb{E}[|Z|/|Y|]^{-1}}$",
        labelpad=6,
    )
    ax.grid(linewidth=0.4, alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.85, edgecolor="none",
              ncol=2, columnspacing=0.8, handlelength=1.4)

    fig.tight_layout(pad=0.4)
    _save(fig, os.path.join(FIG_DIR, f"slack_W{W}.{fmt}"), fmt)


def main():
    if not os.path.isdir(RESULTS_DIR):
        sys.exit(f"Results not found: {RESULTS_DIR}\nRun the experiment first.")

    both = "--both" in sys.argv
    fmt  = "png" if "--png" in sys.argv else "pdf"
    fmts = ["pdf", "png"] if both else [fmt]

    entries = load_all()
    if not entries:
        sys.exit(f"No V*.npz files found in {RESULTS_DIR}")

    colors = _palette(len(entries))

    sample_d = entries[0][1]
    W_list = [int(w) for w in sample_d["W_list"]]
    V_list = [V for V, _ in entries]
    print(f"V values: {V_list}")
    print(f"W values: {W_list}")

    for fmt in fmts:
        print(f"--- {fmt} ---")
        for W in W_list:
            print(f"  Generating slack_W{W} ...")
            plot_slack_for_W(W, entries, colors, fmt)

    print("Done.")


if __name__ == "__main__":
    main()
