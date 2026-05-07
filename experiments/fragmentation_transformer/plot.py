"""Plotting for the Theorem 1 transformer experiment.

Produces two figure families from per-seed logs:
  * main -- x-axis = source symbols processed
  * appendix -- x-axis = gradient steps

Each plot shows smoothed last-position loss (mean ± std over seeds) for the
source-Y and source-X models, and theoretical asymptotes. Default axis labels
and legend entries follow the paper notation (see ``DEFAULT_*`` constants).
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

COLOR_Y = "#378ADD"
COLOR_X = "#D85A30"

# Default axis labels for plot_configs (override via xlabel_main=..., etc.).
DEFAULT_XLABEL_MAIN = "Source symbols processed"
DEFAULT_YLABEL_MAIN = "Loss (bits per source symbol)"
DEFAULT_XLABEL_APPENDIX = "Iterations"
DEFAULT_YLABEL_APPENDIX = DEFAULT_YLABEL_MAIN

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


def _sliding_mean(x, W):
    """Running mean with window W; first W-1 entries use expanding window."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    csum = np.concatenate(([0.0], np.cumsum(x)))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i + 1 - W)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def _aggregate_seeds(per_seed_curves, ce_window):
    """Given a list of 1D arrays (one per seed), return (mean, std, smoothed_per_seed).

    Each per-seed curve is first smoothed with the sliding mean, then
    mean/std are computed across seeds at each index.
    """
    smoothed = np.stack([_sliding_mean(c, ce_window) for c in per_seed_curves], axis=0)
    return smoothed.mean(axis=0), smoothed.std(axis=0), smoothed


def _plot_one(ax, x_axis, mean, std, color, label):
    ax.plot(x_axis, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.2)


def _default_panel_title(d):
    return (
        f"Config {d['name']}: M={d['M']}, K={d['K']}, W={d['W']}, "
        f"$\\alpha$={d['alpha']}"
    )


def _default_legend_y(d):
    return rf"Source ($\mathcal{{Y}}$), $w={d['W']}$"


def _default_legend_x(d):
    return rf"Source ($\mathcal{{X}}$), $Mw={d['M'] * d['W']}$"


def _default_legend_L_Y(d):
    return r"$\mathcal{L}_Y(w)$"


def _default_legend_L_X(d):
    return r"$\mathcal{L}_X^{\mathrm{frag}}(Mw)$"


def plot_configs(all_cfg_data, save_dir_main, save_dir_appendix, file_stem,
                 xscale="log", title_fn=None, appendix_title_fn=None,
                 xlabel_main=None, ylabel_main=None,
                 xlabel_appendix=None, ylabel_appendix=None,
                 legend_y_fn=None, legend_x_fn=None,
                 legend_L_Y_fn=None, legend_L_X_fn=None):
    """Plot main (x = Y-symbols) and appendix (x = grad updates) figures.

    `all_cfg_data` is a list of dicts, one per config, with:
        name, M, K, W, alpha,
        y_per_symbol_seeds : list of 1D numpy arrays (len S_Y each)
        x_per_symbol_seeds : list of 1D numpy arrays (len S_Y each)
        x_per_bit_seeds    : list of 1D numpy arrays (len S_X each)
        L_Y_star, L_X_star, ce_window

    `xscale` is passed to `ax.set_xscale` for both figures (e.g. "log" or
    "linear").

    `title_fn` and `appendix_title_fn` are optional callables ``cfg_dict ->
    str`` for panel titles; defaults match the original formatting.

    Axis labels default to ``DEFAULT_XLABEL_MAIN``, ``DEFAULT_YLABEL_MAIN``,
    ``DEFAULT_XLABEL_APPENDIX``, ``DEFAULT_YLABEL_APPENDIX``; pass overrides to
    customize.

    Legend entries use callables ``cfg_dict -> str``: ``legend_y_fn`` /
    ``legend_x_fn`` for the two model curves, ``legend_L_Y_fn`` /
    ``legend_L_X_fn`` for the horizontal asymptotes.
    """
    title_main = title_fn or _default_panel_title
    title_app = appendix_title_fn if appendix_title_fn is not None else title_main
    xlab_m = xlabel_main if xlabel_main is not None else DEFAULT_XLABEL_MAIN
    ylab_m = ylabel_main if ylabel_main is not None else DEFAULT_YLABEL_MAIN
    xlab_a = xlabel_appendix if xlabel_appendix is not None else DEFAULT_XLABEL_APPENDIX
    ylab_a = ylabel_appendix if ylabel_appendix is not None else DEFAULT_YLABEL_APPENDIX
    leg_y = legend_y_fn or _default_legend_y
    leg_x = legend_x_fn or _default_legend_x
    leg_LY = legend_L_Y_fn or _default_legend_L_Y
    leg_LX = legend_L_X_fn or _default_legend_L_X
    os.makedirs(save_dir_main, exist_ok=True)
    os.makedirs(save_dir_appendix, exist_ok=True)

    # --- Main figure: x-axis = Y-symbols processed --------------------------
    n = len(all_cfg_data)
    fig_main, axes_main = plt.subplots(1, n, figsize=(4.875 * n, 3.3), sharey=False)
    if n == 1:
        axes_main = [axes_main]
    for ax, d in zip(axes_main, all_cfg_data):
        ce_w = d["ce_window"]
        y_mean, y_std, _ = _aggregate_seeds(d["y_per_symbol_seeds"], ce_w)
        x_mean, x_std, _ = _aggregate_seeds(d["x_per_symbol_seeds"], ce_w)
        start_idx = 100
        S_X = len(x_mean)
        x_axis = np.arange(start_idx + 1, S_X + 1)
        _plot_one(ax, x_axis, y_mean[start_idx:S_X], y_std[start_idx:S_X], COLOR_Y, leg_y(d))
        _plot_one(ax, x_axis, x_mean[start_idx:S_X], x_std[start_idx:S_X], COLOR_X, leg_x(d))
        ax.axhline(d["L_Y_star"], color=COLOR_Y, linestyle="--",
                   linewidth=1.1, label=leg_LY(d))
        ax.axhline(d["L_X_star"], color=COLOR_X, linestyle="--",
                   linewidth=1.1, label=leg_LX(d))
        ax.set_xscale(xscale)
        ax.set_xlabel(xlab_m)
        ax.set_ylabel(ylab_m)
        _tit = title_main(d)
        if _tit:
            ax.set_title(_tit, fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", framealpha=0.85, edgecolor="none",
                  handlelength=1.4)
    fig_main.tight_layout()
    main_pdf = os.path.join(save_dir_main, f"{file_stem}.pdf")
    main_png = os.path.join(save_dir_main, f"{file_stem}.png")
    fig_main.savefig(main_pdf)
    fig_main.savefig(main_png, dpi=300)
    plt.close(fig_main)

    # --- Appendix figure: x-axis = gradient updates -------------------------
    fig_app, axes_app = plt.subplots(1, n, figsize=(4.875 * n, 3.3), sharey=False)
    if n == 1:
        axes_app = [axes_app]
    for ax, d in zip(axes_app, all_cfg_data):
        ce_w = d["ce_window"]
        M = d["M"]
        # Y-model: 1 update per Y-symbol. x-axis = 1..S_Y.
        y_mean, y_std, _ = _aggregate_seeds(d["y_per_symbol_seeds"], ce_w)
        start_idx = 100
        S_Y = len(y_mean)
        y_x = np.arange(start_idx + 1, S_Y + 1)
        # X-model: M updates per Y-symbol. Smooth the per-bit losses in bits,
        # then plot against x-axis = 1..S_X.
        x_bit_mean, x_bit_std, _ = _aggregate_seeds(d["x_per_bit_seeds"], ce_w * M)
        S_X = len(x_bit_mean)
        x_x = np.arange(start_idx + 1, S_X + 1)

        # Both curves must be in bits per Y-symbol on the y-axis. Multiply the
        # smoothed per-bit X-curve by M to put it in the same units.
        ax.plot(y_x, y_mean[start_idx:S_Y], color=COLOR_Y, linewidth=1.5,
                label=leg_y(d))
        ax.fill_between(y_x, y_mean[start_idx:S_Y] - y_std[start_idx:S_Y], y_mean[start_idx:S_Y] + y_std[start_idx:S_Y],
                        color=COLOR_Y, alpha=0.2)
        ax.plot(x_x, x_bit_mean[start_idx:S_X] * M, color=COLOR_X, linewidth=1.5,
                label=leg_x(d))
        ax.fill_between(x_x, (x_bit_mean[start_idx:S_X] - x_bit_std[start_idx:S_X]) * M,
                        (x_bit_mean[start_idx:S_X] + x_bit_std[start_idx:S_X]) * M,
                        color=COLOR_X, alpha=0.2)
        ax.axhline(d["L_Y_star"], color=COLOR_Y, linestyle="--",
                   linewidth=1.1, label=leg_LY(d))
        ax.axhline(d["L_X_star"], color=COLOR_X, linestyle="--",
                   linewidth=1.1, label=leg_LX(d))
        ax.set_xscale(xscale)
        ax.set_xlabel(xlab_a)
        ax.set_ylabel(ylab_a)
        _tit = title_app(d)
        if _tit:
            ax.set_title(_tit, fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", framealpha=0.85, edgecolor="none",
                  handlelength=1.4)
    fig_app.tight_layout()
    app_pdf = os.path.join(save_dir_appendix, f"{file_stem}_grad.pdf")
    app_png = os.path.join(save_dir_appendix, f"{file_stem}_grad.png")
    fig_app.savefig(app_pdf)
    fig_app.savefig(app_png, dpi=300)
    plt.close(fig_app)

    return {
        "main_pdf": main_pdf,
        "main_png": main_png,
        "appendix_pdf": app_pdf,
        "appendix_png": app_png,
    }
