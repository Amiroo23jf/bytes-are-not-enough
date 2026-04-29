"""Plotting for the text_transformer experiment.

Compares byte-level and tokenised LMs on a bits-per-byte (BPB) y-axis.

Figure families (each in two x-axis variants):

  figures/main/<stem>.{pdf,png}          -- aggregate mean ± std over seeds;
                                            x = source bytes processed.
  figures/appendix/<stem>_grad.{pdf,png} -- same but x = gradient steps.
  figures/grid/<stem>.{pdf,png}          -- rows = configs, cols = seeds.
  figures/seed0/<stem>.{pdf,png}         -- seed-0 view, one panel per config.

``all_cfg_data`` is a list of dicts, one per config run, each containing:
    name              : str
    mode              : "bytes" or "tokens"
    vocab_size        : int
    seq_len           : int
    ce_window         : int
    mean_token_bytes  : float  (1.0 for bytes mode)
    seeds             : list[int]
    loss_bits_seeds   : list of (S,) float32 arrays  -- CE in bits per step
    bytes_processed_seeds : list of (S,) int64 arrays -- cumul. source bytes
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

COLOR_BYTES = "#378ADD"
COLOR_TOKENS = "#D85A30"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _color(d: dict) -> str:
    return COLOR_BYTES if d["mode"] == "bytes" else COLOR_TOKENS


def _label(d: dict) -> str:
    mode = d["mode"].capitalize()
    return f"{mode} (V={d['vocab_size']}, W={d['seq_len']})"


def _bpb(d: dict, losses: np.ndarray) -> np.ndarray:
    """Convert per-step CE (bits/token) to bits-per-source-byte."""
    return np.asarray(losses, dtype=np.float64) / float(d["mean_token_bytes"])


def _sliding_mean(x: np.ndarray, W: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    csum = np.concatenate(([0.0], np.cumsum(x)))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i + 1 - W)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def _interp_to_grid(
    xs_per_seed: list, ys_per_seed: list, x_grid: np.ndarray
) -> np.ndarray:
    out = np.empty((len(xs_per_seed), len(x_grid)), dtype=np.float64)
    for i, (xs, ys) in enumerate(zip(xs_per_seed, ys_per_seed)):
        out[i] = np.interp(x_grid, np.asarray(xs, dtype=np.float64), ys)
    return out


def _build_x_grid(
    xs_per_seed: list, npoints: int = 512, log: bool = False
) -> np.ndarray:
    lo = max(float(xs[0]) for xs in xs_per_seed)
    hi = min(float(xs[-1]) for xs in xs_per_seed)
    if hi <= lo:
        hi = lo + 1.0
    if log and lo <= 0:
        lo = max(lo, 1.0)
    return np.geomspace(lo, hi, npoints) if log else np.linspace(lo, hi, npoints)


def _decorate(ax, xscale: str, xlabel: str) -> None:
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("bits per byte (BPB)", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)


def _seed_label(d: dict, j: int) -> int:
    seeds = d.get("seeds")
    if seeds is not None and j < len(seeds):
        return int(seeds[j])
    return j


# ---------------------------------------------------------------------------
# Aggregate (mean ± std over seeds)
# ---------------------------------------------------------------------------

def _plot_aggregate(
    all_cfg_data: list,
    save_dir: str,
    file_stem: str,
    xscale: str,
    x_kind: str,  # "bytes" | "steps"
    suffix: str,
) -> tuple[str, str]:
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    all_xs: list = []
    for d in all_cfg_data:
        if x_kind == "bytes":
            all_xs.extend(d["bytes_processed_seeds"])
        else:
            for losses in d["loss_bits_seeds"]:
                all_xs.append(np.arange(1, len(losses) + 1))

    x_grid = _build_x_grid(all_xs, npoints=512, log=(xscale == "log"))

    for d in all_cfg_data:
        ce_w = int(d["ce_window"])
        bpb_seeds = [_bpb(d, losses) for losses in d["loss_bits_seeds"]]
        smoothed = [_sliding_mean(bpb, ce_w) for bpb in bpb_seeds]

        if x_kind == "bytes":
            xs_seeds = [np.asarray(x, dtype=np.float64) for x in d["bytes_processed_seeds"]]
        else:
            xs_seeds = [np.arange(1, len(s) + 1, dtype=np.float64) for s in smoothed]

        Y = _interp_to_grid(xs_seeds, smoothed, x_grid)
        color = _color(d)
        ax.plot(x_grid, Y.mean(axis=0), color=color, linewidth=1.5, label=_label(d))
        ax.fill_between(
            x_grid,
            Y.mean(axis=0) - Y.std(axis=0),
            Y.mean(axis=0) + Y.std(axis=0),
            color=color, alpha=0.2,
        )

    xlabel = "source bytes processed" if x_kind == "bytes" else "gradient steps"
    _decorate(ax, xscale, xlabel)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    pdf = os.path.join(save_dir, f"{file_stem}{suffix}.pdf")
    png = os.path.join(save_dir, f"{file_stem}{suffix}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------------
# Grid (rows = configs, cols = seeds)
# ---------------------------------------------------------------------------

def _plot_grid(
    all_cfg_data: list,
    save_dir: str,
    file_stem: str,
    x_kind: str,
    xscale: str,
    suffix: str,
) -> tuple[str, str]:
    n_cfg = len(all_cfg_data)
    n_seeds = max(len(d["loss_bits_seeds"]) for d in all_cfg_data)
    fig, axes = plt.subplots(
        n_cfg, n_seeds,
        figsize=(4.5 * n_seeds, 3.4 * n_cfg),
        sharey="row",
        squeeze=False,
    )
    xlabel = "source bytes processed" if x_kind == "bytes" else "gradient steps"

    for i, d in enumerate(all_cfg_data):
        ce_w = int(d["ce_window"])
        n_have = len(d["loss_bits_seeds"])
        for j in range(n_seeds):
            ax = axes[i][j]
            if j >= n_have:
                ax.set_visible(False)
                continue
            bpb = _bpb(d, d["loss_bits_seeds"][j])
            smoothed = _sliding_mean(bpb, ce_w)
            if x_kind == "bytes":
                xs = np.asarray(d["bytes_processed_seeds"][j], dtype=np.float64)
            else:
                xs = np.arange(1, len(smoothed) + 1, dtype=np.float64)
            ax.plot(xs, smoothed, color=_color(d), linewidth=1.2)
            _decorate(ax, xscale, xlabel)
            ax.set_title(
                f"{d['name']}, seed {_seed_label(d, j)}", fontsize=10
            )

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    pdf = os.path.join(save_dir, f"{file_stem}{suffix}.pdf")
    png = os.path.join(save_dir, f"{file_stem}{suffix}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------------
# Single-seed (one panel per config)
# ---------------------------------------------------------------------------

def _plot_single_seed(
    all_cfg_data: list,
    save_dir: str,
    file_stem: str,
    x_kind: str,
    xscale: str,
    suffix: str,
    seed_index: int = 0,
) -> tuple[str, str]:
    n_cfg = len(all_cfg_data)
    fig, axes = plt.subplots(1, n_cfg, figsize=(5.5 * n_cfg, 4.3), sharey=False)
    if n_cfg == 1:
        axes = [axes]
    xlabel = "source bytes processed" if x_kind == "bytes" else "gradient steps"

    for ax, d in zip(axes, all_cfg_data):
        if seed_index >= len(d["loss_bits_seeds"]):
            ax.set_visible(False)
            continue
        ce_w = int(d["ce_window"])
        bpb = _bpb(d, d["loss_bits_seeds"][seed_index])
        smoothed = _sliding_mean(bpb, ce_w)
        if x_kind == "bytes":
            xs = np.asarray(d["bytes_processed_seeds"][seed_index], dtype=np.float64)
        else:
            xs = np.arange(1, len(smoothed) + 1, dtype=np.float64)
        ax.plot(xs, smoothed, color=_color(d), linewidth=1.5, label=_label(d))
        _decorate(ax, xscale, xlabel)
        ax.set_title(
            f"{d['name']} — seed {_seed_label(d, seed_index)}", fontsize=11
        )
        ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
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
    all_cfg_data: list,
    save_dir_main: str,
    save_dir_appendix: str,
    file_stem: str,
    save_dir_grid: str | None = None,
    save_dir_single: str | None = None,
    single_seed_index: int = 0,
    xscale_main: str = "linear",
    xscale_appendix: str = "linear",
) -> dict:
    """Render all figure families to disk and return their paths.

    Parameters
    ----------
    all_cfg_data:
        List of per-config dicts (see module docstring for required keys).
    save_dir_main:
        Directory for the aggregate figures (x = source bytes).
    save_dir_appendix:
        Directory for the aggregate-vs-grad figures (x = gradient steps).
    file_stem:
        Base filename (no extension) for all saved figures.
    save_dir_grid / save_dir_single:
        Override directories; default to siblings of ``save_dir_main``.
    single_seed_index:
        Which seed index to show in the seed0 figures.
    xscale_main / xscale_appendix:
        Matplotlib scale string (``"linear"`` or ``"log"``).

    Returns
    -------
    dict mapping figure variant names to ``(pdf, png)`` path pairs.
    """
    if save_dir_grid is None:
        save_dir_grid = os.path.join(os.path.dirname(save_dir_main), "grid")
    if save_dir_single is None:
        save_dir_single = os.path.join(os.path.dirname(save_dir_main), "seed0")

    main_pdf, main_png = _plot_aggregate(
        all_cfg_data, save_dir_main, file_stem,
        xscale_main, "bytes", suffix="",
    )
    app_pdf, app_png = _plot_aggregate(
        all_cfg_data, save_dir_appendix, file_stem,
        xscale_appendix, "steps", suffix="_grad",
    )
    grid_pdf, grid_png = _plot_grid(
        all_cfg_data, save_dir_grid, file_stem,
        "bytes", xscale_main, suffix="",
    )
    grid_app_pdf, grid_app_png = _plot_grid(
        all_cfg_data, save_dir_grid, file_stem,
        "steps", xscale_appendix, suffix="_grad",
    )
    seed_pdf, seed_png = _plot_single_seed(
        all_cfg_data, save_dir_single, file_stem,
        "bytes", xscale_main, suffix="",
        seed_index=single_seed_index,
    )
    seed_app_pdf, seed_app_png = _plot_single_seed(
        all_cfg_data, save_dir_single, file_stem,
        "steps", xscale_appendix, suffix="_grad",
        seed_index=single_seed_index,
    )

    return {
        "main_pdf": main_pdf,
        "main_png": main_png,
        "appendix_pdf": app_pdf,
        "appendix_png": app_png,
        "grid_pdf": grid_pdf,
        "grid_png": grid_png,
        "grid_appendix_pdf": grid_app_pdf,
        "grid_appendix_png": grid_app_png,
        "seed_pdf": seed_pdf,
        "seed_png": seed_png,
        "seed_appendix_pdf": seed_app_pdf,
        "seed_appendix_png": seed_app_png,
    }
