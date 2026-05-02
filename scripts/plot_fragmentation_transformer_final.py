"""Rebuild fragmentation_transformer final figures from saved .npz logs (no retraining).

Matches [run_fragmentation_transformer.py](scripts/run_fragmentation_transformer.py):
same smoothing (``ce_window`` sliding mean), mean ± std across seeds, asymptotes,
and default axis/legend styling from
[experiments/fragmentation_transformer/plot.py](experiments/fragmentation_transformer/plot.py).

Figures use **no panel title** by default; set ``title_main`` / ``title_appendix``
to return a string if you want one. Optional CLI:
``python scripts/plot_fragmentation_transformer_final.py [config_stem]``
(default stem ``final`` → ``configs/fragmentation_transformer/final.yaml`` and
``logs/final_seed*.npz``).
"""

import glob
import os
import re
import sys

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


# ---------------------------------------------------------------------------
# Leave returning "" for no title; or return a string (supports math text).
# ---------------------------------------------------------------------------
def title_main(d):
    return ""


def title_appendix(d):
    return title_main(d)


XSCALE = "linear"
CFG_CE_WINDOW = 500

def _load_config_seeds(name):
    pat = os.path.join(LOG_DIR, f"{name}_seed*.npz")
    seeds = []
    for path in sorted(glob.glob(pat)):
        m = re.match(rf"{re.escape(name)}_seed(\d+)\.npz", os.path.basename(path))
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
        "name": cfg["name"],
        "M": cfg["M"],
        "K": cfg["K"],
        "W": cfg["W"],
        "alpha": cfg["alpha"],
        "ce_window": CFG_CE_WINDOW,
    }


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "final"
    seeds = _load_config_seeds(stem)
    if not seeds:
        print(
            f"No npz logs matching {LOG_DIR}/{stem}_seed*.npz — run training first.",
            file=sys.stderr,
        )
        sys.exit(1)

    meta = _load_config_meta(stem)
    L_Y_star = float(np.mean([s["L_Y_star"] for s in seeds]))
    L_X_star = float(np.mean([s["L_X_star"] for s in seeds]))

    all_cfg_data = [{
        "name": meta["name"],
        "M": meta["M"],
        "K": meta["K"],
        "W": meta["W"],
        "alpha": meta["alpha"],
        "ce_window": meta["ce_window"],
        "L_Y_star": L_Y_star,
        "L_X_star": L_X_star,
        "y_per_symbol_seeds": [s["y_per_symbol"] for s in seeds],
        "x_per_symbol_seeds": [s["x_per_symbol"] for s in seeds],
        "x_per_bit_seeds": [s["x_per_bit"] for s in seeds],
    }]

    file_stem = f"fragmentation_transformer_{meta['name']}"
    paths = plot_configs(
        all_cfg_data,
        FIG_MAIN,
        FIG_APPX,
        file_stem=file_stem,
        xscale=XSCALE,
        title_fn=title_main,
        appendix_title_fn=title_appendix,
    )
    print("Wrote:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
