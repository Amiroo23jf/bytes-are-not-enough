"""Sweep over Dirichlet concentration alpha and context offset K - k.

Runs the fragmentation_ngram pipeline for each (alpha, K_offset) combination,
saves each individual run's artifacts (figure, CSV, JSON) under a unique
tag, and produces one aggregation figure + CSV summarizing how the
decomposition components vary with alpha.
"""

import copy
import json
import os
import time

import matplotlib.pyplot as plt
import pandas as pd

from .run import run_from_config


def alpha_tag(alpha):
    return f"a{alpha:g}".replace(".", "p")


def sweep_tag(K_offset, alpha):
    return f"fragmentation_ngram_K{K_offset}_{alpha_tag(alpha)}"


def _config_label(k, M):
    return f"k={k},M={M}"


def _save_sweep_csv(rows, path):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def _plot_alpha_sweep(rows, save_dir, file_stem):
    """Plot phase_MI, orc_res, emp_penalty as a function of alpha per (k,M).

    Two panels, one per K_offset value. Each panel has one triple of lines per
    (k,M) configuration. The x-axis is alpha (log scale); y is bits/Y-symbol.
    """
    df = pd.DataFrame(rows)
    K_offsets = sorted(df["K_offset"].unique())
    configs = sorted(set(zip(df["k"], df["M"])))

    fig, axes = plt.subplots(1, len(K_offsets), figsize=(6.5 * len(K_offsets), 5),
                              sharey=False)
    if len(K_offsets) == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    for ax, K_off in zip(axes, K_offsets):
        sub = df[df["K_offset"] == K_off].sort_values("alpha")
        for idx, (k, M) in enumerate(configs):
            series = sub[(sub["k"] == k) & (sub["M"] == M)].sort_values("alpha")
            color = cmap(idx % 10)
            ax.plot(series["alpha"], series["phase_MI_mean"], "-o",
                    color=color, label=f"{_config_label(k, M)} phase_MI")
            ax.plot(series["alpha"], series["orc_res_mean"], "--s",
                    color=color, alpha=0.7,
                    label=f"{_config_label(k, M)} orc_res")
        ax.set_xscale("log")
        ax.set_xlabel("Dirichlet concentration alpha", fontsize=11)
        ax.set_ylabel("bits per Y symbol", fontsize=11)
        ax.set_title(f"(K = k{'+%d' % K_off if K_off else ''})", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)
        ax.axhline(0.0, color="black", linewidth=0.6)
    axes[-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))

    fig.tight_layout()
    pdf_path = os.path.join(save_dir, f"{file_stem}.pdf")
    png_path = os.path.join(save_dir, f"{file_stem}.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


def run_alpha_sweep(
    base_cfg,
    out_dir,
    alphas,
    K_offsets,
    n_matrices,
    n_sequence,
    verbose=True,
):
    """Run the alpha x K_offset sweep and write all artifacts under `out_dir`.

    Parameters
    ----------
    base_cfg : dict
        Base YAML config to copy and override per sweep cell.
    out_dir : str
        Experiment output root; per-cell artifacts go to `<out_dir>/figures/`
        and `<out_dir>/raw/`, and the cross-cell summary goes to
        `<out_dir>/figures/<sweep>.{pdf,png,csv}` plus
        `<out_dir>/raw/<sweep>.json`.
    alphas : list of float
        Dirichlet concentrations to sweep.
    K_offsets : list of int
        Values of `K - k` to sweep.
    n_matrices, n_sequence : int
        Per-cell sample sizes; override the values in `base_cfg`.

    Returns
    -------
    dict with keys csv, json, fig_pdf, fig_png.
    """
    fig_dir = os.path.join(out_dir, "figures")
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    rows = []
    t_total_start = time.time()
    for K_off in K_offsets:
        for alpha in alphas:
            tag = sweep_tag(K_off, alpha)
            cfg = copy.deepcopy(base_cfg)
            cfg["sampling"]["n_matrices"] = n_matrices
            cfg["sampling"]["n_sequence"] = n_sequence
            cfg["sampling"]["alpha"] = alpha
            cfg["theory"]["n_sequence_marginal"] = n_sequence
            cfg["ngram"]["K"] = None
            cfg["ngram"]["K_offset"] = K_off

            if verbose:
                print(f"\n{'#' * 72}\n# {tag}  (alpha={alpha}, K_offset={K_off})\n"
                      f"{'#' * 72}", flush=True)
            t0 = time.time()
            all_results, _, _ = run_from_config(cfg, tag, out_dir=out_dir,
                                                verbose=verbose)
            if verbose:
                print(f"[sweep] {tag} finished in {time.time() - t0:.1f}s")

            for r in all_results:
                m, s = r["mean"], r["std"]
                rows.append({
                    "K_offset": K_off,
                    "alpha": alpha,
                    "k": r["k"],
                    "M": r["M"],
                    "K": r["K"],
                    "H_Y_th_mean": m["H_Y_th"],
                    "loss_Y_ngram_mean": m["loss_Y_ngram"],
                    "loss_Y_ngram_std": s["loss_Y_ngram"],
                    "loss_X_ngram_per_Y_mean": m["loss_X_ngram_per_Y"],
                    "loss_X_ngram_per_Y_std": s["loss_X_ngram_per_Y"],
                    "phase_MI_mean": m["th_penalty_per_Y"],
                    "phase_MI_std": s["th_penalty_per_Y"],
                    "orc_res_mean": m["oracle_residual_per_Y"],
                    "orc_res_std": s["oracle_residual_per_Y"],
                    "th_total_mean": m["th_total_penalty_per_Y"],
                    "th_total_std": s["th_total_penalty_per_Y"],
                    "emp_pen_mean": m["emp_penalty_per_Y"],
                    "emp_pen_std": s["emp_penalty_per_Y"],
                    "decomp_rel_err":
                        abs(m["emp_penalty_per_Y"] - m["th_total_penalty_per_Y"])
                        / max(abs(m["emp_penalty_per_Y"]), 1e-12),
                })

    csv_path = _save_sweep_csv(
        rows, os.path.join(fig_dir, "fragmentation_ngram_sweep_summary.csv")
    )
    fig_paths = _plot_alpha_sweep(rows, fig_dir, "fig1_fragmentation_ngram_alpha_sweep")
    raw_path = os.path.join(raw_dir, "fragmentation_ngram_sweep_summary.json")
    with open(raw_path, "w") as f:
        json.dump(rows, f, indent=2)

    if verbose:
        print("\n" + "=" * 72)
        print(f"Sweep CSV:   {csv_path}")
        print(f"Sweep JSON:  {raw_path}")
        print(f"Sweep fig:   {fig_paths['pdf']}")
        print(f"             {fig_paths['png']}")
        print(f"Total sweep runtime: {time.time() - t_total_start:.1f}s")

    return {
        "csv": csv_path,
        "json": raw_path,
        "fig_pdf": fig_paths["pdf"],
        "fig_png": fig_paths["png"],
    }
