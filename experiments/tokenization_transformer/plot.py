"""Plotting for the Theorem 3 tokenization-transformer experiment.

Three figure families per run, each in two x-axis variants:

  * `figures/main/{stem}.{pdf,png}`      and `figures/appendix/{stem}_grad.{pdf,png}`
        Aggregate over seeds (mean +/- 1 std band). The "main" variant uses
        x = Y-symbols processed (bits); the "appendix" variant uses x =
        gradient updates.

  * `figures/grid/{stem}.{pdf,png}`      and `figures/grid/{stem}_grad.{pdf,png}`
        Grid of subplots: rows = configs, cols = seeds. Each subplot shows
        that single (config, seed) run, with both the Y-LM and the Z-LM
        smoothed curves.

  * `figures/seed0/{stem}.{pdf,png}`     and `figures/seed0/{stem}_grad.{pdf,png}`
        One row of subplots, one per config, showing only seed 0's run.

The two LMs run on different "clocks" (one update per bit vs one update per
token), so the main variant aligns them on a common bits-processed x-axis.
The grid and seed0 variants show each LM on its native clock since they're
not being compared across seeds.

The dashed bit-entropy asymptote H(Y) is intentionally deferred until the
empirical curves are validated; the hook is left below as a comment so the
follow-up is a one-line addition.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

COLOR_Y = "#378ADD"
COLOR_Z = "#D85A30"


def _sliding_mean(x, W):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    csum = np.concatenate(([0.0], np.cumsum(x)))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i + 1 - W)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def _aggregate_seeds(per_seed_curves, smoothing_window):
    """Smooth each seed's curve and aggregate (mean, std) across seeds.

    Per-seed curves can have different lengths (this happens for the Z-LM,
    whose per-seed S_Z depends on the BPE encoding of that seed's Y), so
    we truncate to the shortest seed before stacking.
    """
    if len(per_seed_curves) == 0:
        raise ValueError("per_seed_curves is empty")
    min_len = min(len(c) for c in per_seed_curves)
    smoothed = np.stack(
        [
            _sliding_mean(c, smoothing_window)[:min_len]
            for c in per_seed_curves
        ],
        axis=0,
    )
    return smoothed.mean(axis=0), smoothed.std(axis=0)


def _interp_to_grid(xs_per_seed, ys_per_seed, x_grid):
    out = np.empty((len(xs_per_seed), len(x_grid)), dtype=np.float64)
    for i, (xs, ys) in enumerate(zip(xs_per_seed, ys_per_seed)):
        out[i] = np.interp(x_grid, xs, ys)
    return out


def _build_x_grid(xs_per_seed, npoints=512, log=False):
    lo = max(float(xs[0]) for xs in xs_per_seed)
    hi = min(float(xs[-1]) for xs in xs_per_seed)
    if hi <= lo:
        hi = lo + 1.0
    if log and lo <= 0:
        lo = max(lo, 1.0)
    if log:
        return np.geomspace(lo, hi, npoints)
    return np.linspace(lo, hi, npoints)


def _seed_label(d, j):
    """Return the seed number for the j-th run of config d (0-indexed)."""
    seeds = d.get("seeds")
    if seeds is not None and j < len(seeds):
        return int(seeds[j])
    return j


def _z_label(d):
    return f"Z-LM (V={d['V']}, ctx W={d['W']}, {d.get('encoder','greedy')})"


def _y_label(d):
    return f"Y-LM (V=2, ctx W={d['W']})"


def _decorate(ax, d, xscale, xlabel):
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("bits per bit", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)


def _config_title(d):
    return (
        f"Config {d['name']}: K={d['K']}, W={d['W']}, V={d['V']}, "
        f"$\\alpha$={d['alpha']}"
    )


# ---------------------------------------------------------------------------
# Aggregate (mean +/- std over seeds)
# ---------------------------------------------------------------------------

def _plot_aggregate_main(all_cfg_data, save_dir, file_stem, xscale):
    n = len(all_cfg_data)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.3), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, d in zip(axes, all_cfg_data):
        ce_w = int(d["ce_window"])
        y_smoothed = [_sliding_mean(c, ce_w) for c in d["y_per_bit_seeds"]]
        z_smoothed = [_sliding_mean(c, ce_w) for c in d["z_per_bit_seeds"]]
        xs_y = [np.asarray(x, dtype=np.float64) for x in d["y_bits_processed_seeds"]]
        xs_z = [np.asarray(x, dtype=np.float64) for x in d["z_bits_processed_seeds"]]

        x_grid = _build_x_grid(xs_y + xs_z, npoints=512, log=(xscale == "log"))
        Y = _interp_to_grid(xs_y, y_smoothed, x_grid)
        Z = _interp_to_grid(xs_z, z_smoothed, x_grid)

        ax.plot(x_grid, Y.mean(axis=0), color=COLOR_Y, linewidth=1.5,
                label=_y_label(d))
        ax.fill_between(
            x_grid, Y.mean(axis=0) - Y.std(axis=0),
            Y.mean(axis=0) + Y.std(axis=0),
            color=COLOR_Y, alpha=0.2,
        )
        ax.plot(x_grid, Z.mean(axis=0), color=COLOR_Z, linewidth=1.5,
                label=_z_label(d))
        ax.fill_between(
            x_grid, Z.mean(axis=0) - Z.std(axis=0),
            Z.mean(axis=0) + Z.std(axis=0),
            color=COLOR_Z, alpha=0.2,
        )

        # ax.axhline(d["H_Y"], color="black", linestyle="--", linewidth=1.0,
        #            label=r"$H(Y)$")  # deferred; add when theory.py exists.

        _decorate(ax, d, xscale, "Y-symbols processed (bits)")
        ax.set_title(_config_title(d), fontsize=11)
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    pdf = os.path.join(save_dir, f"{file_stem}.pdf")
    png = os.path.join(save_dir, f"{file_stem}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def _plot_aggregate_appendix(all_cfg_data, save_dir, file_stem, xscale):
    n = len(all_cfg_data)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.3), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, d in zip(axes, all_cfg_data):
        ce_w = int(d["ce_window"])
        y_mean, y_std = _aggregate_seeds(d["y_per_bit_seeds"], ce_w)
        z_mean, z_std = _aggregate_seeds(d["z_per_bit_seeds"], ce_w)
        y_x = np.arange(1, len(y_mean) + 1)
        z_x = np.arange(1, len(z_mean) + 1)

        ax.plot(y_x, y_mean, color=COLOR_Y, linewidth=1.5, label=_y_label(d))
        ax.fill_between(y_x, y_mean - y_std, y_mean + y_std,
                        color=COLOR_Y, alpha=0.2)
        ax.plot(z_x, z_mean, color=COLOR_Z, linewidth=1.5, label=_z_label(d))
        ax.fill_between(z_x, z_mean - z_std, z_mean + z_std,
                        color=COLOR_Z, alpha=0.2)

        # ax.axhline(d["H_Y"], color="black", linestyle="--", linewidth=1.0,
        #            label=r"$H(Y)$")  # deferred; add when theory.py exists.

        _decorate(ax, d, xscale, "gradient updates")
        ax.set_title(_config_title(d), fontsize=11)
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    pdf = os.path.join(save_dir, f"{file_stem}_grad.pdf")
    png = os.path.join(save_dir, f"{file_stem}_grad.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------------
# Grid (rows = configs, cols = seeds)
# ---------------------------------------------------------------------------

def _plot_single_run(ax, d, j, x_kind):
    """Plot the j-th seed of config d on `ax`. x_kind: 'bits' | 'steps'."""
    ce_w = int(d["ce_window"])
    y_smoothed = _sliding_mean(d["y_per_bit_seeds"][j], ce_w)
    z_smoothed = _sliding_mean(d["z_per_bit_seeds"][j], ce_w)

    if x_kind == "bits":
        y_xs = np.asarray(d["y_bits_processed_seeds"][j], dtype=np.float64)
        z_xs = np.asarray(d["z_bits_processed_seeds"][j], dtype=np.float64)
    elif x_kind == "steps":
        y_xs = np.arange(1, len(y_smoothed) + 1)
        z_xs = np.arange(1, len(z_smoothed) + 1)
    else:
        raise ValueError(f"x_kind must be 'bits' or 'steps', got {x_kind!r}")

    ax.plot(y_xs, y_smoothed, color=COLOR_Y, linewidth=1.5, label=_y_label(d))
    ax.plot(z_xs, z_smoothed, color=COLOR_Z, linewidth=1.5, label=_z_label(d))


def _plot_grid(all_cfg_data, save_dir, file_stem, x_kind, xscale, suffix):
    """Grid figure: rows = configs, cols = seeds (max across configs)."""
    n_cfg = len(all_cfg_data)
    n_seeds = max(len(d["y_per_bit_seeds"]) for d in all_cfg_data)
    fig, axes = plt.subplots(
        n_cfg, n_seeds,
        figsize=(4.5 * n_seeds, 3.4 * n_cfg),
        sharey="row", squeeze=False,
    )

    xlabel = "Y-symbols processed (bits)" if x_kind == "bits" else "gradient updates"

    for i, d in enumerate(all_cfg_data):
        n_have = len(d["y_per_bit_seeds"])
        for j in range(n_seeds):
            ax = axes[i][j]
            if j >= n_have:
                ax.set_visible(False)
                continue
            _plot_single_run(ax, d, j, x_kind)
            _decorate(ax, d, xscale, xlabel)
            ax.set_title(
                f"{d['name']}, seed {_seed_label(d, j)}", fontsize=10,
            )
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    pdf = os.path.join(save_dir, f"{file_stem}{suffix}.pdf")
    png = os.path.join(save_dir, f"{file_stem}{suffix}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------------
# Single-seed (one row, one panel per config, single chosen seed)
# ---------------------------------------------------------------------------

def _plot_single_seed(
    all_cfg_data, save_dir, file_stem, x_kind, xscale, suffix, seed_index=0,
):
    n_cfg = len(all_cfg_data)
    fig, axes = plt.subplots(1, n_cfg, figsize=(5.5 * n_cfg, 4.3), sharey=False)
    if n_cfg == 1:
        axes = [axes]
    xlabel = "Y-symbols processed (bits)" if x_kind == "bits" else "gradient updates"

    for ax, d in zip(axes, all_cfg_data):
        n_have = len(d["y_per_bit_seeds"])
        if seed_index >= n_have:
            ax.set_visible(False)
            continue
        _plot_single_run(ax, d, seed_index, x_kind)
        _decorate(ax, d, xscale, xlabel)
        ax.set_title(
            f"{_config_title(d)} — seed {_seed_label(d, seed_index)}", fontsize=11,
        )
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    pdf = os.path.join(save_dir, f"{file_stem}{suffix}.pdf")
    png = os.path.join(save_dir, f"{file_stem}{suffix}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def plot_configs(
    all_cfg_data,
    save_dir_main,
    save_dir_appendix,
    file_stem,
    save_dir_grid=None,
    save_dir_single=None,
    single_seed_index=0,
    xscale_main="linear",
    xscale_appendix="linear",
):
    """Render all three figure families to disk and return their paths.

    `all_cfg_data` is a list of dicts (one per config) with:
        name, K, W, V, alpha, encoder, ce_window, mean_token_bits,
        seeds (optional, list[int]),
        y_per_bit_seeds       : list of (S_Y,) per-step bits-per-bit (Y-LM).
        y_bits_processed_seeds: list of (S_Y,) cumulative Y-bits (= step*B).
        z_per_bit_seeds       : list of (S_Z,) per-step bits-per-bit (Z-LM).
        z_bits_processed_seeds: list of (S_Z,) cumulative Y-bits (Z-LM).
    """
    if save_dir_grid is None:
        save_dir_grid = os.path.join(os.path.dirname(save_dir_main), "grid")
    if save_dir_single is None:
        save_dir_single = os.path.join(os.path.dirname(save_dir_main), "seed0")

    for d in (save_dir_main, save_dir_appendix, save_dir_grid, save_dir_single):
        os.makedirs(d, exist_ok=True)

    main_pdf, main_png = _plot_aggregate_main(
        all_cfg_data, save_dir_main, file_stem, xscale_main,
    )
    app_pdf, app_png = _plot_aggregate_appendix(
        all_cfg_data, save_dir_appendix, file_stem, xscale_appendix,
    )

    grid_main_pdf, grid_main_png = _plot_grid(
        all_cfg_data, save_dir_grid, file_stem, "bits", xscale_main, suffix="",
    )
    grid_app_pdf, grid_app_png = _plot_grid(
        all_cfg_data, save_dir_grid, file_stem, "steps", xscale_appendix,
        suffix="_grad",
    )

    seed_main_pdf, seed_main_png = _plot_single_seed(
        all_cfg_data, save_dir_single, file_stem, "bits", xscale_main,
        suffix="", seed_index=single_seed_index,
    )
    seed_app_pdf, seed_app_png = _plot_single_seed(
        all_cfg_data, save_dir_single, file_stem, "steps", xscale_appendix,
        suffix="_grad", seed_index=single_seed_index,
    )

    return {
        "main_pdf": main_pdf,
        "main_png": main_png,
        "appendix_pdf": app_pdf,
        "appendix_png": app_png,
        "grid_main_pdf": grid_main_pdf,
        "grid_main_png": grid_main_png,
        "grid_appendix_pdf": grid_app_pdf,
        "grid_appendix_png": grid_app_png,
        "seed_main_pdf": seed_main_pdf,
        "seed_main_png": seed_main_png,
        "seed_appendix_pdf": seed_app_pdf,
        "seed_appendix_png": seed_app_png,
    }
