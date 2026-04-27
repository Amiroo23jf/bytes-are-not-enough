"""Generate fragmentation_transformer figures from saved per-seed .npz logs.

Produces six figure families (reads logs only, does not retrain):
  * fragmentation_transformer_per_seed.{pdf,png}        -- 1xN overlay with each seed's
                                                          own L_Y*/L_X*.
  * fragmentation_transformer_single_seed.{pdf,png}     -- 1xN, seed 0 per config, one
                                                          clean pair of curves + its
                                                          own asymptotes.
  * fragmentation_transformer_grid.{pdf,png}            -- NxS grid (config x seed).
  * fragmentation_transformer_per_seed_grad.{pdf,png}   -- per_seed but x = grad updates.
  * fragmentation_transformer_all_linear.{pdf,png}      -- aggregated mean +/- std
                                                          across seeds, linear x.
  * fragmentation_transformer_all_linear_grad.{pdf,png} -- aggregated, grad updates,
                                                          linear x.
"""

import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.fragmentation_transformer.plot import plot_configs  # noqa: E402

LOG_DIR = os.path.join(ROOT, "outputs", "fragmentation_transformer", "logs")
FIG_MAIN = os.path.join(ROOT, "outputs", "fragmentation_transformer", "figures", "main")
FIG_APPX = os.path.join(ROOT, "outputs", "fragmentation_transformer", "figures", "appendix")
CFG_DIR = os.path.join(ROOT, "configs", "fragmentation_transformer")

# Blue family for Y, orange family for X. Each seed gets a progressively
# darker shade so overlays are readable.
Y_COLORS = ["#9BC4E8", "#378ADD", "#1F3E66"]
X_COLORS = ["#F2B090", "#D85A30", "#7A2F13"]


def _sliding_mean(x, W):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    csum = np.concatenate(([0.0], np.cumsum(x)))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i + 1 - W)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def _load_config_seeds(name):
    pat = os.path.join(LOG_DIR, f"{name}_seed*.npz")
    seeds = []
    for path in sorted(glob.glob(pat)):
        m = re.match(rf"{name}_seed(\d+)\.npz", os.path.basename(path))
        if not m:
            continue
        seed = int(m.group(1))
        d = np.load(path)
        seeds.append({
            "seed": seed,
            "y_per_symbol": d["y_per_symbol"],
            "x_per_symbol": d["x_per_symbol"],
            "x_per_bit": d["x_per_bit"],
            "L_Y_star": float(d["L_Y_star"]),
            "L_X_star": float(d["L_X_star"]),
        })
    return seeds


def _load_config_meta(name):
    path = os.path.join(CFG_DIR, f"{name}.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return {
        "name": cfg["name"], "M": cfg["M"], "K": cfg["K"], "W": cfg["W"],
        "alpha": cfg["alpha"], "ce_window": int(cfg["train"]["ce_window"]),
        "batch_size": int(cfg["train"]["batch_size"]),
    }


def _title(meta):
    return (f"Config {meta['name']}: K={meta['K']}, M={meta['M']}, "
            f"W={meta['W']}, $\\alpha$={meta['alpha']}")


def _panel_per_seed(ax, meta, seeds, x_axis_mode="y_symbols"):
    ce_w = meta["ce_window"]
    M = meta["M"]
    B = meta["batch_size"]
    for i, s in enumerate(seeds):
        y_sm = _sliding_mean(s["y_per_symbol"], ce_w)
        x_sm = _sliding_mean(s["x_per_symbol"], ce_w)
        if x_axis_mode == "y_symbols":
            x_y = np.arange(1, len(y_sm) + 1) * B
            x_x = np.arange(1, len(x_sm) + 1) * B
        elif x_axis_mode == "grad_updates":
            x_bit_sm = _sliding_mean(s["x_per_bit"], ce_w * M)
            x_sm = x_bit_sm * M
            x_y = np.arange(1, len(y_sm) + 1)
            x_x = np.arange(1, len(x_sm) + 1)
        else:
            raise ValueError(x_axis_mode)

        ycol = Y_COLORS[i % len(Y_COLORS)]
        xcol = X_COLORS[i % len(X_COLORS)]
        ax.plot(x_y, y_sm, color=ycol, linewidth=1.2,
                label=f"Y-model seed {s['seed']}")
        ax.plot(x_x, x_sm, color=xcol, linewidth=1.2,
                label=f"X-model seed {s['seed']}")
        ax.axhline(s["L_Y_star"], color=ycol, linestyle="--", linewidth=0.9)
        ax.axhline(s["L_X_star"], color=xcol, linestyle="--", linewidth=0.9)

    ax.set_xlabel("Y-symbols processed" if x_axis_mode == "y_symbols"
                  else "gradient updates", fontsize=10)
    ax.set_ylabel("bits per Y-symbol", fontsize=10)
    ax.set_title(_title(meta), fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6, loc="upper right", ncol=2)


def _panel_single_seed(ax, meta, seed_data):
    ce_w = meta["ce_window"]
    B = meta["batch_size"]
    y_sm = _sliding_mean(seed_data["y_per_symbol"], ce_w)
    x_sm = _sliding_mean(seed_data["x_per_symbol"], ce_w)
    x_axis = np.arange(1, len(y_sm) + 1) * B
    ycol = "#378ADD"
    xcol = "#D85A30"
    ax.plot(x_axis, y_sm, color=ycol, linewidth=1.6,
            label=f"Y-model (ctx W={meta['W']})")
    ax.plot(x_axis, x_sm, color=xcol, linewidth=1.6,
            label=f"X-model (ctx M$\\cdot$W={meta['M'] * meta['W']})")
    ax.axhline(seed_data["L_Y_star"], color=ycol, linestyle="--", linewidth=1.1,
               label=rf"$L_Y^\star$={seed_data['L_Y_star']:.3f}")
    ax.axhline(seed_data["L_X_star"], color=xcol, linestyle="--", linewidth=1.1,
               label=rf"$L_X^\star$={seed_data['L_X_star']:.3f}")
    ax.set_xlabel("Y-symbols processed", fontsize=10)
    ax.set_ylabel("bits per Y-symbol", fontsize=10)
    ax.set_title(_title(meta) + f" (seed {seed_data['seed']})", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="upper right")


def main():
    names = sys.argv[1:] if len(sys.argv) > 1 else ["A", "B", "C"]
    metas = [_load_config_meta(n) for n in names]
    seeds_per_cfg = [_load_config_seeds(n) for n in names]

    os.makedirs(FIG_MAIN, exist_ok=True)
    os.makedirs(FIG_APPX, exist_ok=True)

    n = len(metas)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.3))
    if n == 1:
        axes = [axes]
    for ax, meta, seeds in zip(axes, metas, seeds_per_cfg):
        _panel_per_seed(ax, meta, seeds, x_axis_mode="y_symbols")
    fig.tight_layout()
    p1_pdf = os.path.join(FIG_MAIN, "fragmentation_transformer_per_seed.pdf")
    p1_png = os.path.join(FIG_MAIN, "fragmentation_transformer_per_seed.png")
    fig.savefig(p1_pdf); fig.savefig(p1_png, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.3))
    if n == 1:
        axes = [axes]
    for ax, meta, seeds in zip(axes, metas, seeds_per_cfg):
        _panel_per_seed(ax, meta, seeds, x_axis_mode="grad_updates")
    fig.tight_layout()
    p2_pdf = os.path.join(FIG_APPX, "fragmentation_transformer_per_seed_grad.pdf")
    p2_png = os.path.join(FIG_APPX, "fragmentation_transformer_per_seed_grad.png")
    fig.savefig(p2_pdf); fig.savefig(p2_png, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.3))
    if n == 1:
        axes = [axes]
    for ax, meta, seeds in zip(axes, metas, seeds_per_cfg):
        seed0 = next(s for s in seeds if s["seed"] == 0)
        _panel_single_seed(ax, meta, seed0)
    fig.tight_layout()
    p3_pdf = os.path.join(FIG_MAIN, "fragmentation_transformer_single_seed.pdf")
    p3_png = os.path.join(FIG_MAIN, "fragmentation_transformer_single_seed.png")
    fig.savefig(p3_pdf); fig.savefig(p3_png, dpi=300)
    plt.close(fig)

    max_seeds = max(len(s) for s in seeds_per_cfg)
    fig, axes = plt.subplots(n, max_seeds, figsize=(5.0 * max_seeds, 3.7 * n),
                              squeeze=False)
    for i, (meta, seeds) in enumerate(zip(metas, seeds_per_cfg)):
        for j in range(max_seeds):
            ax = axes[i, j]
            if j < len(seeds):
                _panel_single_seed(ax, meta, seeds[j])
            else:
                ax.axis("off")
    fig.tight_layout()
    p4_pdf = os.path.join(FIG_MAIN, "fragmentation_transformer_grid.pdf")
    p4_png = os.path.join(FIG_MAIN, "fragmentation_transformer_grid.png")
    fig.savefig(p4_pdf); fig.savefig(p4_png, dpi=300)
    plt.close(fig)

    all_cfg_data = []
    for meta, seeds in zip(metas, seeds_per_cfg):
        L_Y_star = float(np.mean([s["L_Y_star"] for s in seeds]))
        L_X_star = float(np.mean([s["L_X_star"] for s in seeds]))
        all_cfg_data.append({
            "name": meta["name"],
            "M": meta["M"], "K": meta["K"], "W": meta["W"],
            "alpha": meta["alpha"],
            "ce_window": meta["ce_window"],
            "L_Y_star": L_Y_star,
            "L_X_star": L_X_star,
            "y_per_symbol_seeds": [s["y_per_symbol"] for s in seeds],
            "x_per_symbol_seeds": [s["x_per_symbol"] for s in seeds],
            "x_per_bit_seeds":    [s["x_per_bit"]    for s in seeds],
        })
    stem_linear = ("fragmentation_transformer_all_linear"
                   if len(all_cfg_data) > 1
                   else f"fragmentation_transformer_{all_cfg_data[0]['name']}_linear")
    linear_paths = plot_configs(
        all_cfg_data, FIG_MAIN, FIG_APPX,
        file_stem=stem_linear, xscale="linear",
    )
    p5_pdf = linear_paths["main_pdf"]
    p6_pdf = linear_paths["appendix_pdf"]

    print("Per-seed plateau residuals (last 10% of training):")
    print(f"{'cfg':>3} {'seed':>4} | {'L_Y*':>7} {'Y_plat':>7} {'dY':>7} | "
          f"{'L_X*':>7} {'X_plat':>7} {'dX':>7} | {'gap_th':>7} {'gap_emp':>8}")
    for meta, seeds in zip(metas, seeds_per_cfg):
        for s in seeds:
            n_tail = max(1, int(0.1 * len(s["y_per_symbol"])))
            y_plat = float(np.mean(s["y_per_symbol"][-n_tail:]))
            x_plat = float(np.mean(s["x_per_symbol"][-n_tail:]))
            dY = y_plat - s["L_Y_star"]
            dX = x_plat - s["L_X_star"]
            gap_th = s["L_X_star"] - s["L_Y_star"]
            gap_emp = x_plat - y_plat
            print(f"{meta['name']:>3} {s['seed']:>4} | "
                  f"{s['L_Y_star']:>7.4f} {y_plat:>7.4f} {dY:>+7.4f} | "
                  f"{s['L_X_star']:>7.4f} {x_plat:>7.4f} {dX:>+7.4f} | "
                  f"{gap_th:>7.4f} {gap_emp:>8.4f}")

    print("\nFigures:")
    for p in (p1_pdf, p2_pdf, p3_pdf, p4_pdf, p5_pdf, p6_pdf):
        print("  " + p)


if __name__ == "__main__":
    main()
