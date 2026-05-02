"""Figure generation for the fragmentation_ngram experiment (paper Theorem 3.1).

Two-panel figure:
  (a) per-symbol loss comparison: n-gram on Y, n-gram on X (normalized), and
      the theoretical conditional entropy H_Y.
  (b) fragmentation penalty: theoretical decomposition (phase MI + oracle
      residual) vs empirical (loss_X_ngram_per_Y - loss_Y_ngram).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_Y = "#1D9E75"
COLOR_X = "#D85A30"
COLOR_TH = "#378ADD"          # phase_MI segment (bottom of the stack)
COLOR_RES = "#9BC4E8"         # oracle_residual segment (top of the stack)


def _config_label(r):
    return f"k={r['k']},M={r['M']}"


def _save_csv(all_results, save_dir, file_stem):
    rows = []
    for r in all_results:
        row = {"k": r["k"], "M": r["M"], "K": r["K"], "V_y": r["V_y"]}
        for key, val in r["mean"].items():
            row[f"{key}_mean"] = val
        for key, val in r["std"].items():
            row[f"{key}_std"] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, f"{file_stem}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def plot_fig1(all_results, save_dir, file_stem="fig1_fragmentation_ngram"):
    """Plot Figure 1 for the fragmentation_ngram experiment.

    Parameters
    ----------
    all_results : list of dict
        One entry per configuration, as returned by run_single_config.
    save_dir : str
        Directory to write the output files.
    file_stem : str
        Filename stem (without extension) for the saved PDF, PNG, and CSV.
    """
    os.makedirs(save_dir, exist_ok=True)

    labels = [_config_label(r) for r in all_results]
    x = np.arange(len(labels))
    bar_w = 0.3
    gap = 0.02

    loss_Y_mean = np.array([r["mean"]["loss_Y_ngram"] for r in all_results])
    loss_Y_std = np.array([r["std"]["loss_Y_ngram"] for r in all_results])
    loss_X_mean = np.array([r["mean"]["loss_X_ngram_per_Y"] for r in all_results])
    loss_X_std = np.array([r["std"]["loss_X_ngram_per_Y"] for r in all_results])
    H_Y_mean = np.array([r["mean"]["H_Y_th"] for r in all_results])

    phase_MI_mean = np.array([r["mean"]["th_penalty_per_Y"] for r in all_results])
    orc_res_mean = np.array([r["mean"]["oracle_residual_per_Y"] for r in all_results])
    th_total_std = np.array([r["std"]["th_total_penalty_per_Y"] for r in all_results])
    emp_pen_mean = np.array([r["mean"]["emp_penalty_per_Y"] for r in all_results])
    emp_pen_std = np.array([r["std"]["emp_penalty_per_Y"] for r in all_results])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))

    # -------- Panel (a) --------
    off_Y = -(bar_w + gap)
    off_X = 0.0
    off_TH = bar_w + gap

    ax_a.bar(
        x + off_Y,
        loss_Y_mean,
        width=bar_w,
        yerr=loss_Y_std,
        color=COLOR_Y,
        capsize=4,
        label="$w+1$-gram on $\mathcal{Y}$",
    )
    ax_a.bar(
        x + off_X,
        loss_X_mean,
        width=bar_w,
        yerr=loss_X_std,
        color=COLOR_X,
        capsize=4,
        label="$mW+1$-gram on $\mathcal{X}$",
    )
    # H_Y drawn as a short horizontal segment centered at each group.
    for i, h in enumerate(H_Y_mean):
        seg_left = x[i] + off_Y - bar_w / 2
        seg_right = x[i] + off_TH + bar_w / 2
        ax_a.hlines(
            h,
            seg_left,
            seg_right,
            colors=COLOR_TH,
            linewidth=2.2,
            label="H_Y theoretical" if i == 0 else None,
        )

    # Penalty bracket on a representative configuration (the middle one).
    rep = len(all_results) // 2
    x_arrow = x[rep] + off_X + bar_w / 2 + 0.05
    y_top = loss_X_mean[rep]
    y_bot = H_Y_mean[rep]
    ax_a.annotate(
        "",
        xy=(x_arrow, y_top),
        xytext=(x_arrow, y_bot),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
    )
    ax_a.text(
        x_arrow + 0.05,
        0.5 * (y_top + y_bot),
        "penalty",
        fontsize=9,
        va="center",
        ha="left",
    )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=9)
    ax_a.tick_params(axis="y", labelsize=9)
    ax_a.set_ylabel("bits per Y symbol", fontsize=11)
    ax_a.set_title("(a) Per-symbol loss comparison across configurations", fontsize=11)
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.legend(fontsize=9, loc="best")

    # -------- Panel (b) --------
    # Left bar: stacked decomposition phase_MI (bottom) + orc_residual (top).
    # Right bar: empirical penalty. The two bar totals should match.
    off_th = -(bar_w / 2 + gap / 2)
    off_emp = bar_w / 2 + gap / 2

    ax_b.axhline(0.0, color="black", linewidth=0.8)
    ax_b.bar(
        x + off_th,
        phase_MI_mean,
        width=bar_w,
        color=COLOR_TH,
        label="Phase ambiguity (theoretical)",
    )
    ax_b.bar(
        x + off_th,
        orc_res_mean,
        width=bar_w,
        bottom=phase_MI_mean,
        yerr=th_total_std,
        color=COLOR_RES,
        capsize=4,
        label="Context deficit (theoretical)",
    )
    ax_b.bar(
        x + off_emp,
        emp_pen_mean,
        width=bar_w,
        yerr=emp_pen_std,
        color=COLOR_X,
        capsize=4,
        label="Empirical penalty ($L_X - L_Y$)",
    )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=9)
    ax_b.tick_params(axis="y", labelsize=9)
    ax_b.set_ylabel("bits per Y symbol", fontsize=11)
    ax_b.set_title(
        "(b) Fragmentation penalty decomposition: theoretical vs empirical",
        fontsize=11,
    )
    ax_b.grid(axis="y", alpha=0.25)
    ax_b.legend(fontsize=9, loc="best")

    fig.tight_layout()

    pdf_path = os.path.join(save_dir, f"{file_stem}.pdf")
    png_path = os.path.join(save_dir, f"{file_stem}.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    csv_path = _save_csv(all_results, save_dir, file_stem)
    return {"pdf": pdf_path, "png": png_path, "csv": csv_path}
