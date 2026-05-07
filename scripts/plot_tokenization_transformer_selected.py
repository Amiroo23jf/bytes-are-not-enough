"""Plot loss vs iterations for three selected (V, W) pairs in one figure.

Colors are drawn from the same plasma palette used in plot_tokenization_transformer.py,
keyed by each V's rank in the full sorted V list from summary.json.

Pairs:
  - V=20, W=4
  - V=8,  W=6
  - V=2,  W=12

Output
------
  outputs/tokenization_transformer/figures/selected_pairs_steps.{ext}
  outputs/tokenization_transformer/figures/selected_pairs_bits.{ext}

Usage
-----
  python scripts/plot_tokenization_transformer_selected.py          # PDF
  python scripts/plot_tokenization_transformer_selected.py --png
  python scripts/plot_tokenization_transformer_selected.py --both
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

PAIRS = [
    dict(V=20, W=4,  label=r"$V=20,\ W=4$"),
    dict(V=8,  W=6,  label=r"$V=8,\  W=6$"),
    dict(V=2,  W=12, label=r"$V=2,\  W=12$"),
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


def _palette(n, vmax=0.75):
    cmap = matplotlib.colormaps["plasma"]
    return [cmap(vmax * i / max(n - 1, 1)) for i in range(n)]


def load_summary():
    if not os.path.isfile(SUMMARY):
        sys.exit(f"Summary not found: {SUMMARY}\nRun the experiment first.")
    with open(SUMMARY) as f:
        entries = json.load(f)
    H = float(entries[0]["entropy_rate_mean"])
    V_vals = sorted(set(e["V"] for e in entries))
    colors = _palette(len(V_vals))
    color_map = {V: c for V, c in zip(V_vals, colors)}
    return H, color_map


def load_config(V, W):
    """Return (mean_loss, std_loss, mean_bits) averaged over all available seeds."""
    loss_runs, bits_runs = [], []
    for fname in sorted(os.listdir(LOG_DIR)):
        if not (fname.startswith(f"markov_V{V}_W{W}_seed") and fname.endswith(".npz")):
            continue
        d = np.load(os.path.join(LOG_DIR, fname))
        loss_runs.append(d["loss_per_iter"].astype(np.float64))
        bits_runs.append(np.cumsum(d["bits_seen_per_iter"].astype(np.float64)))
    if not loss_runs:
        sys.exit(f"No logs found for V={V} W={W} in {LOG_DIR}")
    loss_arr = np.stack(loss_runs)
    return np.mean(loss_arr, axis=0), np.std(loss_arr, axis=0), np.mean(bits_runs, axis=0)


def plot_steps(H, color_map, fmt):
    fig, ax = plt.subplots(figsize=(HALF_COL_W, HALF_COL_H))
    for cfg in PAIRS:
        loss, std, _ = load_config(cfg["V"], cfg["W"])
        mean_sm, off = _smooth(loss, SMOOTH)
        std_sm,  _   = _smooth(std,  SMOOTH)
        steps = np.arange(off + 1, len(loss) + 1)
        color = color_map[cfg["V"]]
        ax.fill_between(steps, mean_sm - 0.5*std_sm, mean_sm + 0.5*std_sm,
                        color=color, alpha=0.25, linewidth=0)
        ax.plot(steps, mean_sm, color=color, label=cfg["label"])
    if H is not None:
        ax.axhline(H, color="black", linewidth=0.8, linestyle="--", label=r"$H$")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss (bits per source symbol)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_si))
    ax.grid(linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.85, edgecolor="none", handlelength=1.4)
    fig.tight_layout(pad=0.4)
    _save(fig, os.path.join(FIG_DIR, f"selected_pairs_steps.{fmt}"), fmt)


def plot_bits(H, color_map, fmt):
    fig, ax = plt.subplots(figsize=(HALF_COL_W, HALF_COL_H))

    all_data = [load_config(cfg["V"], cfg["W"]) for cfg in PAIRS]
    bits_max = min(d[2][-1] for d in all_data)

    for cfg, (loss, std, bits) in zip(PAIRS, all_data):
        mean_sm, off = _smooth(loss, SMOOTH)
        std_sm,  _   = _smooth(std,  SMOOTH)
        bits_sm = bits[off:]
        mask = bits_sm <= bits_max * 1.001
        color = color_map[cfg["V"]]
        ax.fill_between(bits_sm[mask], (mean_sm - 0.5*std_sm)[mask], (mean_sm + 0.5*std_sm)[mask],
                        color=color, alpha=0.25, linewidth=0)
        ax.plot(bits_sm[mask], mean_sm[mask], color=color, label=cfg["label"])
    if H is not None:
        ax.axhline(H, color="black", linewidth=0.8, linestyle="--", label=r"$H$")
    ax.set_xlabel("Source Symbols Processed")
    ax.set_ylabel("Loss (bits per source symbol)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_si))
    ax.grid(linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.85, edgecolor="none", handlelength=1.4)
    fig.tight_layout(pad=0.4)
    _save(fig, os.path.join(FIG_DIR, f"selected_pairs_bits.{fmt}"), fmt)


def main():
    both = "--both" in sys.argv
    fmt  = "png" if "--png" in sys.argv else "pdf"
    fmts = ["pdf", "png"] if both else [fmt]

    H, color_map = load_summary()

    for fmt in fmts:
        plot_steps(H, color_map, fmt)
        plot_bits(H, color_map, fmt)

    print("Done.")


if __name__ == "__main__":
    main()
