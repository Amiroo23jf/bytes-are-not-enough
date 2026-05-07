"""Generate figures for the tokenization_transformer experiment.

Reads outputs/tokenization_transformer/summary.json and the per-seed NPZ logs,
then writes into outputs/tokenization_transformer/figures/:

  plateau_vs_W.{ext}
      Plateau loss vs window size W; one line per V.
      Half NeurIPS column (~3.25 × 2.2 in).

  curves_W{w}_steps.{ext}    (one per W)
      Loss vs gradient steps; all V trends, smoothed.

  curves_W{w}_bits.{ext}     (one per W)
      Loss vs source symbols processed; all V trends, smoothed.

Usage
-----
  python scripts/plot_tokenization_transformer.py          # PDF (default)
  python scripts/plot_tokenization_transformer.py --png    # PNG at 300 DPI
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

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

HALF_COL_W = 4.875  # 1.5× half NeurIPS column width (inches)
HALF_COL_H = 3.3   # 1.5× height

# Trailing moving-average window (gradient steps).
# Using a trailing average avoids all boundary artifacts: output starts at
# step SMOOTH and ends at the final step with no zero-padding distortion.
SMOOTH = 500

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
    """Sample n colors from plasma[0, vmax] — avoids the harsh yellow tail."""
    cmap = matplotlib.colormaps["plasma"]
    return [cmap(vmax * i / max(n - 1, 1)) for i in range(n)]


def _smooth(arr, k):
    """Trailing uniform moving average.

    smoothed[i] = mean(arr[i : i+k]), aligned to index i+k-1 of the
    original array (the last step in that window).  No zero-padding,
    so there are no boundary artifacts at either end.

    Returns (smoothed_array, offset) where offset = k-1 is the index in
    the original array corresponding to the first output element.
    """
    if k <= 1:
        return arr.astype(np.float32), 0
    cs = np.concatenate([[0.0], np.cumsum(arr, dtype=np.float64)])
    smoothed = ((cs[k:] - cs[:-k]) / k).astype(np.float32)
    return smoothed, k - 1   # first output is at original index k-1


def _si(x, _pos=None):
    if abs(x) >= 1e9:
        return f"{x/1e9:.1f}G"
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def _save(fig, path, fmt, dpi=300):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi if fmt == "png" else None)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary():
    with open(SUMMARY) as f:
        entries = json.load(f)
    V_vals = sorted(set(e["V"] for e in entries))
    W_vals = sorted(set(e["W"] for e in entries))
    H = float(entries[0]["entropy_rate_mean"])
    return entries, V_vals, W_vals, H


def load_curves(V, W):
    """Return (loss_1d, bits_cumsum_1d) averaged over seeds, or (None, None)."""
    loss_runs, bits_runs = [], []
    for fname in sorted(os.listdir(LOG_DIR)):
        if not (fname.startswith(f"markov_V{V}_W{W}_seed") and fname.endswith(".npz")):
            continue
        d = np.load(os.path.join(LOG_DIR, fname))
        loss_runs.append(d["loss_per_iter"].astype(np.float64))
        bits_runs.append(np.cumsum(d["bits_seen_per_iter"].astype(np.float64)))
    if not loss_runs:
        return None, None
    return np.mean(loss_runs, axis=0), np.mean(bits_runs, axis=0)


# ---------------------------------------------------------------------------
# Plot 1 — plateau loss vs W
# ---------------------------------------------------------------------------

_PLATEAU_W_EXCLUDE = {16}   # W values to omit from the plateau figure

def plot_plateau_vs_W(entries, V_vals, W_vals, H, fmt):
    fig, ax = plt.subplots(figsize=(HALF_COL_W, HALF_COL_H))
    colors = _palette(len(V_vals))
    lookup = {(e["V"], e["W"]): e for e in entries}
    plot_W_vals = [w for w in W_vals if w not in _PLATEAU_W_EXCLUDE]

    for color, V in zip(colors, V_vals):
        ws, ys = [], []
        for W in plot_W_vals:
            e = lookup.get((V, W))
            if e is not None:
                ws.append(W)
                ys.append(e["plateau_loss_mean"])
        ax.plot(ws, ys, marker="o", markersize=4, color=color, label=f"$V={V}$")

    ax.axhline(H, color="black", linewidth=0.9, linestyle="--",
               label=r"$H$ (entropy rate)")

    ax.set_xlabel("Window size $W$")
    ax.set_ylabel("Plateau loss (bits / source symbol)")
    ax.set_xticks(plot_W_vals)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.legend(loc="upper right", framealpha=0.85, edgecolor="none",
              ncol=2, columnspacing=0.8, handlelength=1.4)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.tight_layout(pad=0.4)
    _save(fig, os.path.join(FIG_DIR, f"plateau_vs_W.{fmt}"), fmt)


# ---------------------------------------------------------------------------
# Plot 2a — loss vs gradient steps (one figure per W)
# ---------------------------------------------------------------------------

def plot_curves_steps(W, V_vals, H, fmt):
    fig, ax = plt.subplots(figsize=(HALF_COL_W, HALF_COL_H))
    colors = _palette(len(V_vals))

    for color, V in zip(colors, V_vals):
        loss, _ = load_curves(V, W)
        if loss is None:
            continue
        smoothed, off = _smooth(loss, SMOOTH)
        # 1-indexed gradient steps; first output is at step off+1
        steps = np.arange(off + 1, len(loss) + 1)
        ax.plot(steps, smoothed, color=color, label=f"$V={V}$", alpha=0.9)

    ax.axhline(H, color="black", linewidth=0.8, linestyle="--", label=r"$H$")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss (bits per source symbol)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_si))
    ax.grid(linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.85, edgecolor="none",
              ncol=2, columnspacing=0.8, handlelength=1.4)

    fig.tight_layout(pad=0.4)
    _save(fig, os.path.join(FIG_DIR, f"curves_W{W}_steps.{fmt}"), fmt)


# ---------------------------------------------------------------------------
# Plot 2b — loss vs source symbols processed (one figure per W)
# ---------------------------------------------------------------------------

def plot_curves_bits(W, V_vals, H, fmt):
    fig, ax = plt.subplots(figsize=(HALF_COL_W, HALF_COL_H))
    colors = _palette(len(V_vals))

    # Collect all runs first to find the shared truncation point.
    curves = {}
    for V in V_vals:
        loss, bits = load_curves(V, W)
        if loss is not None:
            curves[V] = (loss, bits)

    if not curves:
        plt.close(fig)
        return

    max_bits = min(bits[-1] for _, bits in curves.values())

    for color, V in zip(colors, V_vals):
        if V not in curves:
            continue
        loss, bits = curves[V]
        smoothed, off = _smooth(loss, SMOOTH)
        # Align bits to the smoothed output (off = first valid index).
        bits_sm = bits[off:]
        mask = bits_sm <= max_bits * 1.001
        ax.plot(bits_sm[mask], smoothed[mask], color=color,
                label=f"$V={V}$", alpha=0.9)

    ax.axhline(H, color="black", linewidth=0.8, linestyle="--", label=r"$H$")
    ax.set_xlabel("Source Symbols Processed")
    ax.set_ylabel("Loss (bits per source symbol)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_si))
    ax.grid(linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.85, edgecolor="none",
              ncol=2, columnspacing=0.8, handlelength=1.4)

    fig.tight_layout(pad=0.4)
    _save(fig, os.path.join(FIG_DIR, f"curves_W{W}_bits.{fmt}"), fmt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isfile(SUMMARY):
        sys.exit(f"Summary not found: {SUMMARY}\nRun the experiment first.")

    both = "--both" in sys.argv
    fmt  = "png" if "--png" in sys.argv else "pdf"
    fmts = ["pdf", "png"] if both else [fmt]

    entries, V_vals, W_vals, H = load_summary()
    print(f"Loaded summary: V={V_vals}  W={W_vals}  H={H:.4f}  format={'pdf+png' if both else fmt}")

    for fmt in fmts:
        print(f"--- {fmt} ---")
        print("Generating plateau_vs_W ...")
        plot_plateau_vs_W(entries, V_vals, W_vals, H, fmt)

        for W in W_vals:
            print(f"Generating curves W={W} ...")
            plot_curves_steps(W, V_vals, H, fmt)
            plot_curves_bits(W, V_vals, H, fmt)

    print("Done.")


if __name__ == "__main__":
    main()
