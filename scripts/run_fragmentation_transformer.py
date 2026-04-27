"""Entry point: train the Y-model and X-model for one or more configs of the
fragmentation_transformer experiment (paper Theorem 1).

Usage:
    python scripts/run_fragmentation_transformer.py
        # run A, B, C
    python scripts/run_fragmentation_transformer.py configs/fragmentation_transformer/C.yaml
        # single config
    python scripts/run_fragmentation_transformer.py --smoke
        # quick smoke test

Artifacts:
    outputs/fragmentation_transformer/logs/{name}_seed{s}.npz
    outputs/fragmentation_transformer/figures/main/fragmentation_transformer_all.{pdf,png}
    outputs/fragmentation_transformer/figures/appendix/fragmentation_transformer_all_grad.{pdf,png}
    outputs/fragmentation_transformer/summary.json
"""

import json
import os
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.fragmentation_transformer.data import build_streams  # noqa: E402
from experiments.fragmentation_transformer.plot import plot_configs  # noqa: E402
from experiments.fragmentation_transformer.theory import compute_asymptotes  # noqa: E402
from experiments.fragmentation_transformer.train import (  # noqa: E402
    train_X_model,
    train_Y_model,
    x_loss_to_per_Y_symbol,
)

OUT_ROOT = os.path.join(ROOT, "outputs", "fragmentation_transformer")
LOG_DIR = os.path.join(OUT_ROOT, "logs")
FIG_MAIN = os.path.join(OUT_ROOT, "figures", "main")
FIG_APPX = os.path.join(OUT_ROOT, "figures", "appendix")

DEFAULT_CONFIGS = [
    os.path.join(ROOT, "configs", "fragmentation_transformer", "A.yaml"),
    os.path.join(ROOT, "configs", "fragmentation_transformer", "B.yaml"),
    os.path.join(ROOT, "configs", "fragmentation_transformer", "C.yaml"),
]


def _run_single_seed(cfg, seed, device, verbose=True):
    M, K, W = cfg["M"], cfg["K"], cfg["W"]
    t0 = time.time()
    streams = build_streams(
        M=M, K=K, W=W,
        alpha=cfg["alpha"],
        n_sequence=cfg["n_sequence"],
        batch_size=cfg["train"]["batch_size"],
        seed=seed,
        device=device,
    )
    if verbose:
        print(f"    [seed {seed}] streams ready S_Y={streams['S_Y']} "
              f"S_X={streams['S_X']} ({time.time() - t0:.1f}s)", flush=True)

    asymp = compute_asymptotes(
        P=streams["P"], V_y=streams["V_y"],
        K=K, W=W, M=M,
        n_marginal=cfg["n_sequence"],
        rng_seed=seed + 10_000,
    )
    if verbose:
        print(f"    [seed {seed}] L_Y*={asymp['L_Y_star']:.4f}  "
              f"L_X*={asymp['L_X_star']:.4f}  gap={asymp['L_X_star'] - asymp['L_Y_star']:.4f}",
              flush=True)

    name = cfg.get("name", "?")
    t_y0 = time.time()
    y_out = train_Y_model(streams, cfg, seed, device,
                          progress_desc=f"[{name} s{seed}] Y")
    if verbose:
        print(f"    [seed {seed}] Y-model ({y_out['num_parameters']:,} params) "
              f"done in {time.time() - t_y0:.1f}s", flush=True)

    t_x0 = time.time()
    x_out = train_X_model(streams, cfg, seed, device,
                          progress_desc=f"[{name} s{seed}] X")
    if verbose:
        print(f"    [seed {seed}] X-model ({x_out['num_parameters']:,} params) "
              f"done in {time.time() - t_x0:.1f}s", flush=True)

    S_Y = streams["S_Y"]
    y_per_symbol = y_out["last_pos_loss_bits"]
    x_per_bit = x_out["last_pos_loss_bits"]
    x_per_symbol = x_loss_to_per_Y_symbol(x_per_bit, M, S_Y)

    return {
        "seed": seed,
        "L_Y_star": asymp["L_Y_star"],
        "L_X_star": asymp["L_X_star"],
        "y_per_symbol": y_per_symbol,
        "x_per_symbol": x_per_symbol,
        "x_per_bit": x_per_bit,
        "S_Y": S_Y,
        "S_X": streams["S_X"],
        "num_parameters_Y": int(y_out["num_parameters"]),
        "num_parameters_X": int(x_out["num_parameters"]),
    }


def run_config(cfg, device, verbose=True):
    """Run all seeds for one config and save per-seed logs."""
    name = cfg["name"]
    seeds = cfg["seeds"]
    os.makedirs(LOG_DIR, exist_ok=True)

    results_per_seed = []
    asymp_list = []
    for s in seeds:
        if verbose:
            print(f"  [{name}] seed={s}", flush=True)
        r = _run_single_seed(cfg, s, device, verbose=verbose)
        results_per_seed.append(r)
        asymp_list.append((r["L_Y_star"], r["L_X_star"]))

        npz_path = os.path.join(LOG_DIR, f"{name}_seed{s}.npz")
        np.savez_compressed(
            npz_path,
            y_per_symbol=r["y_per_symbol"].astype(np.float32),
            x_per_symbol=r["x_per_symbol"].astype(np.float32),
            x_per_bit=r["x_per_bit"].astype(np.float32),
            L_Y_star=r["L_Y_star"],
            L_X_star=r["L_X_star"],
            num_parameters_Y=r["num_parameters_Y"],
            num_parameters_X=r["num_parameters_X"],
        )
        if verbose:
            print(f"    saved {npz_path}", flush=True)

    L_Y_star = float(np.mean([a[0] for a in asymp_list]))
    L_X_star = float(np.mean([a[1] for a in asymp_list]))

    tail_frac = 0.1

    def _tail_mean(arr):
        n_tail = max(1, int(tail_frac * len(arr)))
        return float(np.mean(arr[-n_tail:]))

    y_plat = float(np.mean([_tail_mean(r["y_per_symbol"]) for r in results_per_seed]))
    y_plat_std = float(np.std([_tail_mean(r["y_per_symbol"]) for r in results_per_seed]))
    x_plat = float(np.mean([_tail_mean(r["x_per_symbol"]) for r in results_per_seed]))
    x_plat_std = float(np.std([_tail_mean(r["x_per_symbol"]) for r in results_per_seed]))

    gap_th = L_X_star - L_Y_star
    gap_emp = x_plat - y_plat
    rel_err = abs(gap_emp - gap_th) / max(abs(gap_th), 1e-12)

    num_params_Y = int(results_per_seed[0]["num_parameters_Y"])
    num_params_X = int(results_per_seed[0]["num_parameters_X"])
    return {
        "name": name,
        "M": cfg["M"], "K": cfg["K"], "W": cfg["W"], "alpha": cfg["alpha"],
        "seeds": seeds,
        "ce_window": int(cfg["train"]["ce_window"]),
        "L_Y_star": L_Y_star,
        "L_X_star": L_X_star,
        "y_plateau_mean": y_plat, "y_plateau_std": y_plat_std,
        "x_plateau_mean": x_plat, "x_plateau_std": x_plat_std,
        "gap_theoretical": gap_th,
        "gap_empirical": gap_emp,
        "gap_rel_err": rel_err,
        "num_parameters_Y": num_params_Y,
        "num_parameters_X": num_params_X,
        "_per_seed": results_per_seed,
    }


def _load_cfg(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    expected = "fragmentation_transformer"
    found = cfg.get("experiment")
    if found is not None and found != expected:
        raise ValueError(
            f"Config 'experiment: {found!r}' does not match expected {expected!r}"
        )
    return cfg


def main():
    args = sys.argv[1:]
    smoke = "--smoke" in args
    args = [a for a in args if a != "--smoke"]

    if args:
        cfg_paths = [os.path.abspath(a) if not os.path.isabs(a) else a for a in args]
    else:
        cfg_paths = DEFAULT_CONFIGS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    cfgs = [_load_cfg(p) for p in cfg_paths]
    if smoke:
        for c in cfgs:
            c["n_sequence"] = 50_000
            c["seeds"] = [0]
            c["train"]["warmup_steps"] = 50
            c["train"]["ce_window"] = 200

    all_cfg_data = []
    summary = []
    t_all = time.time()
    for cfg, path in zip(cfgs, cfg_paths):
        print("\n" + "#" * 72)
        print(f"# Config: {cfg['name']}  (from {os.path.basename(path)})")
        print("#" * 72, flush=True)
        res = run_config(cfg, device, verbose=True)
        summary.append({k: v for k, v in res.items() if not k.startswith("_")})

        all_cfg_data.append({
            "name": res["name"],
            "M": res["M"], "K": res["K"], "W": res["W"], "alpha": res["alpha"],
            "ce_window": res["ce_window"],
            "L_Y_star": res["L_Y_star"],
            "L_X_star": res["L_X_star"],
            "y_per_symbol_seeds": [r["y_per_symbol"] for r in res["_per_seed"]],
            "x_per_symbol_seeds": [r["x_per_symbol"] for r in res["_per_seed"]],
            "x_per_bit_seeds":    [r["x_per_bit"]    for r in res["_per_seed"]],
        })

    stem = ("fragmentation_transformer_all" if len(all_cfg_data) > 1
            else f"fragmentation_transformer_{all_cfg_data[0]['name']}")
    paths = plot_configs(all_cfg_data, FIG_MAIN, FIG_APPX, file_stem=stem)

    summary_path = os.path.join(OUT_ROOT, "summary.json")
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("Summary (bits per Y-symbol, plateau = last 10% of training):")
    print(f"{'cfg':>3} | {'#Y':>10} {'#X':>10} | "
          f"{'L_Y*':>7} {'Y_plat':>7} | {'L_X*':>7} {'X_plat':>7} | "
          f"{'gap_th':>7} {'gap_emp':>8} {'rel_err':>8}")
    for s in summary:
        print(f"{s['name']:>3} | "
              f"{s['num_parameters_Y']:>10,} {s['num_parameters_X']:>10,} | "
              f"{s['L_Y_star']:>7.4f} {s['y_plateau_mean']:>7.4f} | "
              f"{s['L_X_star']:>7.4f} {s['x_plateau_mean']:>7.4f} | "
              f"{s['gap_theoretical']:>7.4f} {s['gap_empirical']:>8.4f} "
              f"{s['gap_rel_err'] * 100:>7.2f}%")
    print(f"Main figure:     {paths['main_pdf']}")
    print(f"Appendix figure: {paths['appendix_pdf']}")
    print(f"Summary JSON:    {summary_path}")
    print(f"Total runtime:   {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
