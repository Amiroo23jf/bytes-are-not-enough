"""Plot span-distribution CDFs for the WikiText-103 tokenizer experiment.

For each window size W, produces one figure showing the CDF of the character
span covered by W consecutive tokens, with one curve per tokenizer.

Output
------
  outputs/wikitext_tokenizers/figures/cdf_W{w}.{ext}

Usage
-----
  python scripts/plot_wikitext_tokenizers.py           # PDF (default)
  python scripts/plot_wikitext_tokenizers.py --png     # PNG at 300 DPI
  python scripts/plot_wikitext_tokenizers.py --both    # PDF + PNG
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RESULTS_DIR = os.path.join(ROOT, "outputs", "wikitext_tokenizers", "results")
FIG_DIR     = os.path.join(ROOT, "outputs", "wikitext_tokenizers", "figures")

# ---------------------------------------------------------------------------
# Tokenizer display order and pretty labels
# ---------------------------------------------------------------------------

TOK_ORDER = [
    "char",
    "bpe_1k",
    "bpe_4k",
    "bpe_16k",
    "bpe_32k",
    "wordpiece_32k",
    "sentencepiece_32k",
    "bert",
    "gpt2",
]

TOK_LABEL = {
    "char":              "Char",
    "bpe_1k":            "BPE (1k)",
    "bpe_4k":            "BPE (4k)",
    "bpe_16k":           "BPE (16k)",
    "bpe_32k":           "BPE (32k)",
    "wordpiece_32k":     "WordPiece (32k)",
    "sentencepiece_32k": "SentencePiece (32k)",
    "bert":              "BERT",
    "gpt2":              "GPT-2",
}

# ---------------------------------------------------------------------------
# Style — matches tokenization_transformer figures
# ---------------------------------------------------------------------------

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


def _palette(n):
    cmap = matplotlib.colormaps["tab10"]
    return [cmap(i) for i in range(n)]


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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all():
    """Return dict: name -> npz data, for tokenizers present on disk."""
    data = {}
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".npz"):
            continue
        name = fname[:-4]
        data[name] = np.load(os.path.join(RESULTS_DIR, fname), allow_pickle=True)
    return data


# ---------------------------------------------------------------------------
# Plot — CDF per W
# ---------------------------------------------------------------------------

def plot_cdf_for_W(W, tok_data, fmt):
    # Select tokenizers present in disk, in display order.
    names = [n for n in TOK_ORDER if n in tok_data]
    colors = _palette(len(names))

    fig, ax = plt.subplots(figsize=(HALF_COL_W, HALF_COL_H))

    for color, name in zip(colors, names):
        d = tok_data[name]
        key_ws  = f"ws_W{W}"
        key_cdf = f"cdf_W{W}"
        if key_ws not in d.files or key_cdf not in d.files:
            continue
        ws  = d[key_ws].astype(np.float64)
        cdf = d[key_cdf].astype(np.float64)
        label = TOK_LABEL.get(name, name)

        # Char tokenizer is a delta: extend horizontally so the step is visible.
        if len(ws) == 1:
            ws_plot  = np.array([0.0, ws[0], ws[0]])
            cdf_plot = np.array([0.0, 0.0,   1.0])
            ax.plot(ws_plot, cdf_plot, color=color, label=label,
                    linewidth=1.8, zorder=3)
        else:
            ax.plot(ws, cdf, color=color, label=label)

    ax.set_xlabel("Span $w_s$ (characters)")
    ax.set_ylabel("CDF")
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_si))
    ax.grid(linewidth=0.4, alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.85, edgecolor="none",
              handlelength=1.4)

    fig.tight_layout(pad=0.4)
    _save(fig, os.path.join(FIG_DIR, f"cdf_W{W}.{fmt}"), fmt)


# ---------------------------------------------------------------------------
# Plot — scaled CDF per W:  CDF(w_s) × log2(vocab_size) / ratio
# ---------------------------------------------------------------------------

def plot_scaled_cdf_for_W(W, tok_data, fmt):
    names  = [n for n in TOK_ORDER if n in tok_data and n != "char"]
    colors = _palette(len(names))

    fig, ax = plt.subplots(figsize=(HALF_COL_W, HALF_COL_H))

    for color, name in zip(colors, names):
        d = tok_data[name]
        key_ws  = f"ws_W{W}"
        key_cdf = f"cdf_W{W}"
        if key_ws not in d.files or key_cdf not in d.files:
            continue
        ws    = d[key_ws].astype(np.float64)
        cdf   = d[key_cdf].astype(np.float64)
        scale = np.log2(float(d["vocab_size"])) / float(d["ratio"])
        label = TOK_LABEL.get(name, name)

        if len(ws) == 1:
            ws_plot    = np.array([0.0, ws[0], ws[0]])
            scaled_plot = np.array([0.0, 0.0, 1.0]) * scale
            ax.plot(ws_plot, scaled_plot, color=color, label=label,
                    linewidth=1.8, zorder=3)
        else:
            ax.plot(ws, cdf * scale, color=color, label=label)

    # x-axis range: start just below the minimum possible span (≈W),
    # end where the rightmost curve reaches near 1.
    all_ws = [tok_data[n][f"ws_W{W}"] for n in names
              if f"ws_W{W}" in tok_data[n].files]
    x_min = max(1, W * 0.7)
    x_max = max(float(ws[-1]) for ws in all_ws) * 1.05
    ax.set_xlim(x_min, x_max)

    ax.axvline(W, color="black", linewidth=0.9, linestyle="--")
    # x in data coords, y in axes fraction (0=bottom, 1=top)
    ax.text(W * 1.02, 0.04, f"$w={W}$",
            ha="left", va="bottom", fontsize=6.5,
            transform=ax.get_xaxis_transform(),
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.75))

    ax.set_xlabel("Span $w_s$ (characters)")
    ax.set_ylabel(
        r"$\dfrac{\varepsilon \cdot \log_2 |Z_1^{w_s}|}{E[|Z|]}$",
        labelpad=6,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_si))
    ax.grid(linewidth=0.4, alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.85, edgecolor="none",
              handlelength=1.4)

    fig.tight_layout(pad=0.4)
    _save(fig, os.path.join(FIG_DIR, f"cdf_scaled_W{W}.{fmt}"), fmt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isdir(RESULTS_DIR):
        sys.exit(f"Results not found: {RESULTS_DIR}\nRun the experiment first.")

    both = "--both" in sys.argv
    fmt  = "png" if "--png" in sys.argv else "pdf"
    fmts = ["pdf", "png"] if both else [fmt]

    tok_data = load_all()
    if not tok_data:
        sys.exit(f"No .npz files found in {RESULTS_DIR}")

    sample = next(iter(tok_data.values()))
    W_list = [int(w) for w in sample["W_list"]]
    print(f"Tokenizers: {[n for n in TOK_ORDER if n in tok_data]}")
    print(f"W values:   {W_list}")

    for fmt in fmts:
        print(f"--- {fmt} ---")
        for W in W_list:
            print(f"  Generating cdf_W{W} ...")
            plot_cdf_for_W(W, tok_data, fmt)
            print(f"  Generating cdf_scaled_W{W} ...")
            plot_scaled_cdf_for_W(W, tok_data, fmt)

    print("Done.")


if __name__ == "__main__":
    main()
