"""Plotting for the wikitext_span_transformer experiment.

Three figures, one per experiment plus a shared training-curve view:

  Exp 1 (fixed token window w, varying tokenizer): scatter of measured
  character span w_s vs. held-out test loss (bits/char), one point per
  tokenizer -- checks whether tokenizers with a larger span at the same
  token budget also show higher end-to-end loss.

  Exp 2 (fixed target character span w_s, per-tokenizer solved w): bar chart
  of held-out test loss (bits/char) per tokenizer, each using its own solved
  context length w -- checks whether losses converge once context is
  matched in character terms instead of token terms.

  Training curves (either experiment): smoothed training loss (bits/char)
  vs. cumulative characters seen -- a tokenizer-agnostic x-axis, since token
  counts aren't comparable across tokenizers with different granularities.
  Each tokenizer's curve is truncated to the x-range common to all of them
  (the shortest run's endpoint), so every curve ends at the same point.

  Slack curves (see scripts/plot_wikitext_span_slack_curves.py): scaled-CDF
  span curves, one per tokenizer, each using its own already-solved w --
  every curve should cross the same (target w_s, target slack) point by
  construction, visualizing how each tokenizer's span distribution differs
  in shape even though they were all calibrated to hit that one point.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         8,
    "axes.labelsize":    8,
    "axes.titlesize":    9,
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


def _palette(n, vmax=0.85):
    """Qualitative colors for up to 10 unordered categories (tokenizers).

    `plasma` (a sequential colormap, meant for ordered data) packs adjacent
    categories into visually similar shades when there are this many of
    them -- tab10 is built specifically so up to 10 categories stay
    distinguishable from each other.
    """
    if n <= 10:
        cmap = plt.colormaps["tab10"]
        return [cmap(i) for i in range(n)]
    cmap = plt.colormaps["plasma"]
    return [cmap(vmax * i / max(n - 1, 1)) for i in range(n)]


def _save(fig, save_dir, file_stem, dpi=300):
    os.makedirs(save_dir, exist_ok=True)
    pdf = os.path.join(save_dir, f"{file_stem}.pdf")
    png = os.path.join(save_dir, f"{file_stem}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return pdf, png


def _sliding_mean(x, w):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    csum = np.concatenate(([0.0], np.cumsum(x)))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i + 1 - w)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def plot_exp1(results, save_dir, file_stem="wikitext_span_exp1_fixed_w"):
    """results: list of dicts with name, w, w_s, test_loss_bits_per_char."""
    results = sorted(results, key=lambda r: r["w_s"])
    colors = _palette(len(results))

    fig, ax = plt.subplots(figsize=(4.875, 3.3))
    xs = [r["w_s"] for r in results]
    ys = [r["test_loss_bits_per_char"] for r in results]
    for color, r in zip(colors, results):
        ax.scatter(r["w_s"], r["test_loss_bits_per_char"], color=color, s=28, zorder=3,
                   label=r["name"])
    ax.plot(xs, ys, color="0.6", linewidth=0.8, linestyle="--", zorder=1)

    w = results[0]["w"] if results else "?"
    ax.set_xlabel(r"Measured span $w_s$ (chars) at fixed $w=%s$" % w)
    ax.set_ylabel("Held-out test loss (bits/char)")
    ax.set_title("Exp 1: fixed token window, varying span")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.85, edgecolor="none", ncol=2, handlelength=1.2)
    fig.tight_layout(pad=0.4)
    return _save(fig, save_dir, file_stem)


def plot_exp2(results, save_dir, file_stem="wikitext_span_exp2_fixed_w_s"):
    """results: list of dicts with name, w, w_s (target), test_loss_bits_per_char."""
    results = sorted(results, key=lambda r: r["w"])
    colors = _palette(len(results))

    fig, ax = plt.subplots(figsize=(4.875, 3.3))
    names = [r["name"] for r in results]
    ys = [r["test_loss_bits_per_char"] for r in results]
    x = np.arange(len(results))
    ax.bar(x, ys, color=colors, zorder=3)
    mean_y = float(np.mean(ys))
    ax.axhline(mean_y, color="black", linewidth=0.9, linestyle="--", zorder=2,
               label=f"mean = {mean_y:.3f}")

    for xi, r in zip(x, results):
        ax.text(xi, r["test_loss_bits_per_char"], f"w={r['w']}", ha="center", va="bottom",
                fontsize=6, rotation=0)

    w_s = results[0]["w_s"] if results else "?"
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Held-out test loss (bits/char)")
    ax.set_title(r"Exp 2: fixed target span $w_s=%s$, per-tokenizer solved $w$" % w_s)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", framealpha=0.85, edgecolor="none")
    fig.tight_layout(pad=0.4)
    return _save(fig, save_dir, file_stem)


def plot_slack_curves(curves, save_dir, file_stem, target_w_s=None, target_slack=None,
                       y_max=None):
    """Scaled-CDF ("slack") curves, one per tokenizer, each using its OWN w.

    curves: list of dicts with name, w, ws_values, cdf, scale (scale =
    log2(vocab_size) / ratio, per plot_markov_tokenizers.py's convention).

    Unlike plot_markov_tokenizers.py (one shared w across all curves), each
    tokenizer here uses its own already-solved w -- e.g. the w found by
    solve_w_for_target_span for a fixed (w_s, slack) target -- so every
    curve is expected to cross (target_w_s, target_slack) at roughly the
    same point by construction; the reference lines make that visible.

    Mirrors plot_markov_tokenizers.py's style: clips each curve at its own
    95th-percentile span then extends flat (removes the heavy tail without
    distorting the visible region), and special-cases delta distributions
    (e.g. `char`, whose span is deterministic given w).

    `y_max`: caps the y-axis so the region around `target_slack` (where the
    curves are actually being compared) stays readable -- a tokenizer with
    a small vocab (e.g. `char`) has a much larger asymptotic scale than the
    rest, which would otherwise squash everyone else's crossing region into
    an unreadable sliver near y=0. Defaults to `3 * target_slack` when a
    target is given (curves that exceed it, e.g. char's step, simply exit
    the top of the frame -- which is fine, the crossing point is what
    matters). Pass `y_max=False` for the full auto range instead.
    """
    colors = _palette(len(curves))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    p95_map = {}
    for c in curves:
        ws, cdf = c["ws_values"], c["cdf"]
        idx = min(int(np.searchsorted(cdf, 0.95)), len(ws) - 1)
        p95_map[c["name"]] = float(ws[idx])
    x_max = max(p95_map.values()) * 1.05

    for color, c in zip(colors, curves):
        name, w, ws, cdf, scale = c["name"], c["w"], c["ws_values"], c["cdf"], c["scale"]
        label = f"{name} (w={w})"
        if len(ws) == 1:
            ws_plot = np.array([0.0, ws[0], ws[0], x_max])
            sc_plot = np.array([0.0, 0.0, scale, scale])
            ax.plot(ws_plot, sc_plot, color=color, label=label, linewidth=1.8, zorder=3)
        else:
            x95 = p95_map[name]
            idx95 = min(int(np.searchsorted(ws, x95)), len(ws) - 1)
            ws_clip = ws[: idx95 + 1]
            sc_clip = cdf[: idx95 + 1] * scale
            ax.plot(np.append(ws_clip, x_max), np.append(sc_clip, sc_clip[-1]),
                     color=color, label=label, linewidth=1.4)

    if target_slack is not None:
        ax.axhline(target_slack, color="black", linewidth=0.9, linestyle="--", zorder=2)
        ax.text(x_max * 0.98, target_slack, f"slack={target_slack:g}",
                 ha="right", va="bottom", fontsize=6.5)
    if target_w_s is not None:
        ax.axvline(target_w_s, color="black", linewidth=0.9, linestyle=":", zorder=2)
        ax.text(target_w_s * 1.02, 0.02, f"$w_s={target_w_s:g}$",
                 ha="left", va="bottom", fontsize=6.5, transform=ax.get_xaxis_transform())

    if y_max is None and target_slack is not None:
        y_max = 3.0 * target_slack
    if y_max:
        ax.set_ylim(-0.05 * y_max, y_max)

    ax.set_xlim(0, x_max)
    ax.set_xlabel(r"Span $w_s$ (chars)")
    ax.set_ylabel(r"Scaled CDF: $P(\mathrm{span}\leq w_s)\cdot\log_2 V/\mathrm{ratio}$")
    title = "Per-tokenizer slack curves at their own solved $w$"
    if target_w_s is not None and target_slack is not None:
        title += r" (target $w_s=%g$, slack$=%g$)" % (target_w_s, target_slack)
    ax.set_title(title, fontsize=8.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=6, ncol=2, framealpha=0.85, edgecolor="none",
              handlelength=1.4)
    fig.tight_layout(pad=0.4)
    return _save(fig, save_dir, file_stem)


def plot_epsilon_curves(curves, save_dir, file_stem, target_w_s=None):
    """Raw CDF curves epsilon(w_s) = P(span <= w_s), one per tokenizer, each
    using its own solved w -- i.e. `plot_slack_curves` without the
    log2(vocab_size)/ratio scaling.

    Since epsilon is a probability, all 9 curves already live on the same
    [0, 1] y-axis regardless of vocab size, so (unlike plot_slack_curves)
    there's no y_max squashing problem to work around. Note there's also no
    single shared horizontal reference line to draw here: each tokenizer's
    "target" crossing point (scaled(w_s) = target_slack) corresponds to a
    *different* raw-epsilon value per tokenizer (target_slack / scale_i,
    since scale varies with vocab size/ratio) -- only the vertical
    reference at target_w_s is shared.
    """
    colors = _palette(len(curves))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    p95_map = {}
    for c in curves:
        ws, cdf = c["ws_values"], c["cdf"]
        idx = min(int(np.searchsorted(cdf, 0.95)), len(ws) - 1)
        p95_map[c["name"]] = float(ws[idx])
    x_max = max(p95_map.values()) * 1.05

    for color, c in zip(colors, curves):
        name, w, ws, cdf = c["name"], c["w"], c["ws_values"], c["cdf"]
        label = f"{name} (w={w})"
        if len(ws) == 1:
            ws_plot = np.array([0.0, ws[0], ws[0], x_max])
            ep_plot = np.array([0.0, 0.0, 1.0, 1.0])
            ax.plot(ws_plot, ep_plot, color=color, label=label, linewidth=1.8, zorder=3)
        else:
            x95 = p95_map[name]
            idx95 = min(int(np.searchsorted(ws, x95)), len(ws) - 1)
            ws_clip = ws[: idx95 + 1]
            ep_clip = cdf[: idx95 + 1]
            ax.plot(np.append(ws_clip, x_max), np.append(ep_clip, ep_clip[-1]),
                     color=color, label=label, linewidth=1.4)

    if target_w_s is not None:
        ax.axvline(target_w_s, color="black", linewidth=0.9, linestyle=":", zorder=2)
        ax.text(target_w_s * 1.02, 0.02, f"$w_s={target_w_s:g}$",
                 ha="left", va="bottom", fontsize=6.5, transform=ax.get_xaxis_transform())

    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"Span $w_s$ (chars)")
    ax.set_ylabel(r"$\varepsilon(w_s) = P(\mathrm{span}\leq w_s)$")
    title = "Per-tokenizer CDF curves at their own solved $w$"
    if target_w_s is not None:
        title += r" (target $w_s=%g$)" % target_w_s
    ax.set_title(title, fontsize=8.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=6, ncol=2, framealpha=0.85, edgecolor="none",
              handlelength=1.4)
    fig.tight_layout(pad=0.4)
    return _save(fig, save_dir, file_stem)


def plot_training_curves(results, save_dir, file_stem, batch_size, ce_window, title=""):
    """Smoothed training loss (bits/char) vs. cumulative characters seen.

    results: list of dicts with name, w, ratio, last_pos_loss_bits (the
    per-step last-position CE in bits/token, as saved in each result .npz).

    `ratio` (train-split average chars/token) converts both axes: steps ->
    cumulative characters (steps * batch_size * w * ratio), and bits/token
    -> bits/char (divide by ratio). This is the same average-ratio
    conversion used for text_transformer's existing plots -- unlike the
    final held-out test loss (which is exact, using each token's own
    recorded char length), per-step training char-counts weren't recorded,
    so the average ratio is what's available here.

    Every tokenizer's curve is interpolated onto a shared x-grid spanning
    [0, min over tokenizers of that tokenizer's total characters seen] --
    i.e. truncated to the range common to all of them, so they end at the
    same point regardless of how many characters each tokenizer's own run
    happened to cover.
    """
    colors = _palette(len(results))

    curves = []
    for r in results:
        loss_bits_per_token = np.asarray(r["last_pos_loss_bits"], dtype=np.float64)
        S = len(loss_bits_per_token)
        chars_per_step = batch_size * r["w"] * r["ratio"]
        x = np.arange(1, S + 1, dtype=np.float64) * chars_per_step
        y = _sliding_mean(loss_bits_per_token, ce_window) / r["ratio"]
        curves.append((r["name"], x, y))

    common_max = min(x[-1] for _, x, _ in curves)
    x_grid = np.linspace(0.0, common_max, 500)
    y_grids = [(name, np.interp(x_grid, x, y)) for name, x, y in curves]

    # The first steps drop steeply from the random-init loss (~log2(vocab)),
    # which would otherwise dominate the y-range and flatten out exactly the
    # region -- mid/late training -- where the curves actually separate.
    # Skip that initial transient (first 10% of the common x-range) when
    # picking y-limits, but still plot it.
    skip = int(0.1 * len(x_grid))
    tail_vals = np.concatenate([g[skip:] for _, g in y_grids])
    y_lo, y_hi = float(tail_vals.min()), float(tail_vals.max())
    pad = 0.08 * (y_hi - y_lo)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for color, (name, y_grid) in zip(colors, y_grids):
        ax.plot(x_grid, y_grid, color=color, linewidth=1.3, label=name)

    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_xlabel("Characters seen (training)")
    ax.set_ylabel("Training loss (bits/char, smoothed)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.85, edgecolor="none", ncol=2, fontsize=6, handlelength=1.2)
    fig.tight_layout(pad=0.4)
    return _save(fig, save_dir, file_stem)
