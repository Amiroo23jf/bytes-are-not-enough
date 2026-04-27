"""Re-render figures and (re)compute summary.json from saved per-seed npz logs.

Use after a successful training run if you only need to regenerate plots or
the summary -- e.g. after a plotting bug or styling change. Does NOT retrain.

Usage
-----
    python scripts/replot_tokenization_transformer.py
        # rebuild figures + summary.json across every config that has logs

    python scripts/replot_tokenization_transformer.py A_greedy A_priority
        # only the named configs (matched against the {name}_seed*.npz files)
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.tokenization_transformer.plot import plot_configs  # noqa: E402

OUT_ROOT = os.path.join(ROOT, "outputs", "tokenization_transformer")
LOG_DIR = os.path.join(OUT_ROOT, "logs")
FIG_MAIN = os.path.join(OUT_ROOT, "figures", "main")
FIG_APPX = os.path.join(OUT_ROOT, "figures", "appendix")

CE_WINDOW_DEFAULT = 1000
ALPHA_DEFAULT = 0.5
TAIL_FRAC = 0.1


def _tail_mean(arr, frac=TAIL_FRAC):
    n_tail = max(1, int(frac * len(arr)))
    return float(np.mean(arr[-n_tail:]))


def _scalar(npz, key, default):
    return npz[key].item() if key in npz.files else default


def _load_per_config(name_filter=None):
    grouped = defaultdict(list)
    for fname in sorted(os.listdir(LOG_DIR)):
        if not fname.endswith(".npz"):
            continue
        stem = fname[:-4]
        if "_seed" not in stem:
            continue
        name, seed_str = stem.rsplit("_seed", 1)
        if name_filter is not None and name not in name_filter:
            continue
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        grouped[name].append((seed, os.path.join(LOG_DIR, fname)))

    plot_data = []
    summary = []
    for name, items in sorted(grouped.items()):
        items.sort(key=lambda x: x[0])
        seeds = [s for s, _ in items]

        plot_d = {
            "name": name,
            "y_per_bit_seeds": [],
            "y_bits_processed_seeds": [],
            "z_per_bit_seeds": [],
            "z_bits_processed_seeds": [],
        }
        K = W = V = None
        alpha = None
        ce_window = None
        encoder = None
        mtbs = []
        n_params_y = None
        n_params_z = None
        y_per_bit_per_seed = []
        z_per_bit_per_seed = []

        for seed, p in items:
            f = np.load(p)
            plot_d["y_per_bit_seeds"].append(f["y_per_bit"])
            plot_d["y_bits_processed_seeds"].append(f["y_bits_processed"])
            plot_d["z_per_bit_seeds"].append(f["z_per_bit"])
            plot_d["z_bits_processed_seeds"].append(f["z_bits_processed"])
            y_per_bit_per_seed.append(f["y_per_bit"])
            z_per_bit_per_seed.append(f["z_per_bit"])

            K, W, V = int(f["K"]), int(f["W"]), int(f["V"])
            mtbs.append(float(f["mean_token_bits"]))
            n_params_y = int(_scalar(f, "num_parameters_Y", 0))
            n_params_z = int(_scalar(f, "num_parameters_Z", 0))
            alpha = float(_scalar(f, "alpha", ALPHA_DEFAULT))
            ce_window = int(_scalar(f, "ce_window", CE_WINDOW_DEFAULT))
            if "encoder" in f.files:
                encoder = bytes(f["encoder"]).decode("utf-8", errors="replace")

        if encoder is None:
            encoder = (
                "greedy" if "greedy" in name
                else "priority" if "priority" in name
                else "?"
            )

        plot_d["K"], plot_d["W"], plot_d["V"] = K, W, V
        plot_d["alpha"] = alpha
        plot_d["ce_window"] = ce_window
        plot_d["encoder"] = encoder
        plot_d["seeds"] = list(seeds)
        plot_d["mean_token_bits"] = float(np.mean(mtbs))

        y_tails = [_tail_mean(c) for c in y_per_bit_per_seed]
        z_tails = [_tail_mean(c) for c in z_per_bit_per_seed]
        y_plat = float(np.mean(y_tails))
        y_plat_std = float(np.std(y_tails))
        z_plat = float(np.mean(z_tails))
        z_plat_std = float(np.std(z_tails))

        summary.append({
            "name": name,
            "K": K, "W": W, "V": V, "alpha": alpha,
            "encoder": encoder,
            "seeds": seeds,
            "ce_window": ce_window,
            "mean_token_bits": float(np.mean(mtbs)),
            "y_plateau_mean_bpb": y_plat,
            "y_plateau_std_bpb": y_plat_std,
            "z_plateau_mean_bpb": z_plat,
            "z_plateau_std_bpb": z_plat_std,
            "gap_empirical_bpb": y_plat - z_plat,
            "num_parameters_Y": n_params_y,
            "num_parameters_Z": n_params_z,
        })
        plot_data.append(plot_d)

    return plot_data, summary


def main():
    name_filter = set(sys.argv[1:]) or None
    plot_data, summary = _load_per_config(name_filter)
    if not plot_data:
        print(f"No matching logs in {LOG_DIR}.")
        sys.exit(1)

    summary_path = os.path.join(OUT_ROOT, "summary.json")
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    stem = (
        "tokenization_transformer_all" if len(plot_data) > 1
        else f"tokenization_transformer_{plot_data[0]['name']}"
    )
    paths = plot_configs(plot_data, FIG_MAIN, FIG_APPX, file_stem=stem)

    print("Summary (bits per bit, plateau = last 10% of training):")
    print(
        f"{'cfg':>10} | {'enc':>8} | {'#Y':>10} {'#Z':>10} | "
        f"{'mtb':>6} | {'Y_plat':>7} {'Z_plat':>7} | {'gap':>7}"
    )
    for s in summary:
        print(
            f"{s['name']:>10} | {s['encoder']:>8} | "
            f"{s['num_parameters_Y']:>10,} {s['num_parameters_Z']:>10,} | "
            f"{s['mean_token_bits']:>6.3f} | "
            f"{s['y_plateau_mean_bpb']:>7.4f} {s['z_plateau_mean_bpb']:>7.4f} | "
            f"{s['gap_empirical_bpb']:>+7.4f}"
        )
    print(f"Main figure:        {paths['main_pdf']}")
    print(f"Appendix figure:    {paths['appendix_pdf']}")
    print(f"Grid figure:        {paths['grid_main_pdf']}")
    print(f"Grid (grad) figure: {paths['grid_appendix_pdf']}")
    print(f"Seed-0 figure:      {paths['seed_main_pdf']}")
    print(f"Seed-0 (grad):      {paths['seed_appendix_pdf']}")
    print(f"Summary JSON:       {summary_path}")


if __name__ == "__main__":
    main()
