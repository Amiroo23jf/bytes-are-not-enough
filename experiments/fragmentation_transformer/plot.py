"""Plotting for the Theorem 1 transformer experiment.

Produces two figure families from per-seed logs:
  * main/thm1_transformer_{cfg}.{pdf,png}        -- x-axis = Y-symbols processed
  * appendix/thm1_transformer_{cfg}_grad.{pdf,png} -- x-axis = gradient updates

Each plot shows the Y-model and X-model smoothed last-position loss (bits per
Y-symbol) with +/- 1 std bands over seeds, and the two theoretical asymptotes
L_Y* and L_X* as horizontal dashed lines.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

COLOR_Y = "#378ADD"
COLOR_X = "#D85A30"


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


def plot_configs(all_cfg_data, save_dir_main, save_dir_appendix, file_stem,
                 xscale="log"):
    """Plot main (x = Y-symbols) and appendix (x = grad updates) figures.

    `all_cfg_data` is a list of dicts, one per config, with:
        name, M, K, W, alpha,
        y_per_symbol_seeds : list of 1D numpy arrays (len S_Y each)
        x_per_symbol_seeds : list of 1D numpy arrays (len S_Y each)
        x_per_bit_seeds    : list of 1D numpy arrays (len S_X each)
        L_Y_star, L_X_star, ce_window

    `xscale` is passed to `ax.set_xscale` for both figures (e.g. "log" or
    "linear").
    """
    os.makedirs(save_dir_main, exist_ok=True)
    os.makedirs(save_dir_appendix, exist_ok=True)

    # --- Main figure: x-axis = Y-symbols processed --------------------------
    n = len(all_cfg_data)
    fig_main, axes_main = plt.subplots(1, n, figsize=(5.5 * n, 4.3), sharey=False)
    if n == 1:
        axes_main = [axes_main]
    for ax, d in zip(axes_main, all_cfg_data):
        ce_w = d["ce_window"]
        y_mean, y_std, _ = _aggregate_seeds(d["y_per_symbol_seeds"], ce_w)
        x_mean, x_std, _ = _aggregate_seeds(d["x_per_symbol_seeds"], ce_w)
        S_Y = len(y_mean)
        x_axis = np.arange(1, S_Y + 1)
        _plot_one(ax, x_axis, y_mean, y_std, COLOR_Y,
                  f"Y-model (ctx W={d['W']})")
        _plot_one(ax, x_axis, x_mean, x_std, COLOR_X,
                  f"X-model (ctx M*W={d['M'] * d['W']})")
        ax.axhline(d["L_Y_star"], color=COLOR_Y, linestyle="--",
                   linewidth=1.1, label=r"$L_Y^\star$")
        ax.axhline(d["L_X_star"], color=COLOR_X, linestyle="--",
                   linewidth=1.1, label=r"$L_X^\star$")
        ax.set_xscale(xscale)
        ax.set_xlabel("Y-symbols processed", fontsize=11)
        ax.set_ylabel("bits per Y-symbol", fontsize=11)
        ax.set_title(
            f"Config {d['name']}: M={d['M']}, K={d['K']}, W={d['W']}, "
            f"$\\alpha$={d['alpha']}",
            fontsize=11,
        )
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, loc="upper right")
    fig_main.tight_layout()
    main_pdf = os.path.join(save_dir_main, f"{file_stem}.pdf")
    main_png = os.path.join(save_dir_main, f"{file_stem}.png")
    fig_main.savefig(main_pdf)
    fig_main.savefig(main_png, dpi=300)
    plt.close(fig_main)

    # --- Appendix figure: x-axis = gradient updates -------------------------
    fig_app, axes_app = plt.subplots(1, n, figsize=(5.5 * n, 4.3), sharey=False)
    if n == 1:
        axes_app = [axes_app]
    for ax, d in zip(axes_app, all_cfg_data):
        ce_w = d["ce_window"]
        M = d["M"]
        # Y-model: 1 update per Y-symbol. x-axis = 1..S_Y.
        y_mean, y_std, _ = _aggregate_seeds(d["y_per_symbol_seeds"], ce_w)
        S_Y = len(y_mean)
        y_x = np.arange(1, S_Y + 1)
        # X-model: M updates per Y-symbol. Smooth the per-bit losses in bits,
        # then plot against x-axis = 1..S_X.
        x_bit_mean, x_bit_std, _ = _aggregate_seeds(d["x_per_bit_seeds"], ce_w * M)
        S_X = len(x_bit_mean)
        x_x = np.arange(1, S_X + 1)

        # Both curves must be in bits per Y-symbol on the y-axis. Multiply the
        # smoothed per-bit X-curve by M to put it in the same units.
        ax.plot(y_x, y_mean, color=COLOR_Y, linewidth=1.5,
                label=f"Y-model (ctx W={d['W']})")
        ax.fill_between(y_x, y_mean - y_std, y_mean + y_std,
                        color=COLOR_Y, alpha=0.2)
        ax.plot(x_x, x_bit_mean * M, color=COLOR_X, linewidth=1.5,
                label=f"X-model (ctx M*W={d['M'] * d['W']})")
        ax.fill_between(x_x, (x_bit_mean - x_bit_std) * M,
                        (x_bit_mean + x_bit_std) * M,
                        color=COLOR_X, alpha=0.2)
        ax.axhline(d["L_Y_star"], color=COLOR_Y, linestyle="--",
                   linewidth=1.1, label=r"$L_Y^\star$")
        ax.axhline(d["L_X_star"], color=COLOR_X, linestyle="--",
                   linewidth=1.1, label=r"$L_X^\star$")
        ax.set_xscale(xscale)
        ax.set_xlabel("gradient updates", fontsize=11)
        ax.set_ylabel("bits per Y-symbol", fontsize=11)
        ax.set_title(
            f"Config {d['name']}: M={d['M']}, K={d['K']}, W={d['W']}, "
            f"$\\alpha$={d['alpha']}",
            fontsize=11,
        )
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, loc="upper right")
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
