"""Entry point: train the Y-LM and Z-LM for one or more configs of the
tokenization_transformer experiment (paper Theorem 3).

Usage:
    python scripts/run_tokenization_transformer.py
        # run A_greedy and A_priority
    python scripts/run_tokenization_transformer.py configs/tokenization_transformer/A_greedy.yaml
        # single config
    python scripts/run_tokenization_transformer.py --smoke
        # quick smoke test

Artifacts (per `(name, seed)`):
    outputs/tokenization_transformer/logs/{name}_seed{seed}.npz
    outputs/tokenization_transformer/artifacts/{name}_seed{seed}_bpe.json
    outputs/tokenization_transformer/artifacts/{name}_seed{seed}_bpe_hf.json
    outputs/tokenization_transformer/figures/main/{stem}.{pdf,png}
    outputs/tokenization_transformer/figures/appendix/{stem}_grad.{pdf,png}
    outputs/tokenization_transformer/summary.json
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

from experiments.tokenization_transformer.data import build_streams  # noqa: E402
from experiments.tokenization_transformer.plot import plot_configs  # noqa: E402
from experiments.tokenization_transformer.train import (  # noqa: E402
    train_Y_model,
    train_Z_model,
)

OUT_ROOT = os.path.join(ROOT, "outputs", "tokenization_transformer")
LOG_DIR = os.path.join(OUT_ROOT, "logs")
ART_DIR = os.path.join(OUT_ROOT, "artifacts")
FIG_MAIN = os.path.join(OUT_ROOT, "figures", "main")
FIG_APPX = os.path.join(OUT_ROOT, "figures", "appendix")

DEFAULT_CONFIGS = [
    os.path.join(ROOT, "configs", "tokenization_transformer", "A_greedy.yaml"),
    os.path.join(ROOT, "configs", "tokenization_transformer", "A_priority.yaml"),
]


def _run_single_seed(cfg, seed, device, verbose=True):
    K = cfg["K"]
    W = cfg["W"]
    V = cfg["V"]
    tok_cfg = cfg.get("tokenizer", {})
    encoder = tok_cfg.get("encoder", "greedy")
    train_bits = tok_cfg.get("train_bits", None)
    chunk_size = tok_cfg.get("chunk_size", 8192)

    t0 = time.time()
    streams = build_streams(
        K=K, W=W, V=V,
        alpha=cfg["alpha"],
        n_sequence=cfg["n_sequence"],
        batch_size=cfg["train"]["batch_size"],
        seed=seed,
        device=device,
        encoder=encoder,
        train_bits=train_bits,
        chunk_size=chunk_size,
    )
    if verbose:
        print(
            f"    [seed {seed}] streams ready S_Y={streams['S_Y']} "
            f"S_Z={streams['S_Z']} "
            f"mean_token_bits={streams['mean_token_bits']:.4f} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

    name = cfg.get("name", "?")

    t_y0 = time.time()
    y_out = train_Y_model(
        streams["Y_streams"], W=W, cfg=cfg, seed=seed, device=device,
        progress_desc=f"[{name} s{seed}] Y",
    )
    if verbose:
        print(
            f"    [seed {seed}] Y-LM ({y_out['num_parameters']:,} params) "
            f"done in {time.time() - t_y0:.1f}s",
            flush=True,
        )

    t_z0 = time.time()
    z_out = train_Z_model(
        streams["Z_streams"],
        W=W, V=streams["V"],
        mean_token_bits=streams["mean_token_bits"],
        token_bits=streams["token_bits"],
        cfg=cfg, seed=seed, device=device,
        progress_desc=f"[{name} s{seed}] Z",
    )
    if verbose:
        print(
            f"    [seed {seed}] Z-LM ({z_out['num_parameters']:,} params) "
            f"done in {time.time() - t_z0:.1f}s",
            flush=True,
        )

    os.makedirs(ART_DIR, exist_ok=True)
    bpe_path = os.path.join(ART_DIR, f"{name}_seed{seed}_bpe.json")
    streams["tokenizer"].save(bpe_path)

    return {
        "seed": seed,
        "y_per_bit": y_out["last_pos_loss_bits"],
        "y_bits_processed": y_out["bits_processed"],
        "z_per_token_bits": z_out["last_pos_loss_bits_per_token"],
        "z_per_bit": z_out["last_pos_loss_bits_per_bit"],
        "z_bits_processed_per_step": z_out["bits_processed_per_step"],
        "z_bits_processed": z_out["bits_processed"],
        "S_Y": int(y_out["S"]),
        "S_Z": int(z_out["S"]),
        "mean_token_bits": float(streams["mean_token_bits"]),
        "token_bits": streams["token_bits"],
        "num_parameters_Y": int(y_out["num_parameters"]),
        "num_parameters_Z": int(z_out["num_parameters"]),
        "bpe_path": bpe_path,
    }


def run_config(cfg, device, verbose=True):
    name = cfg["name"]
    seeds = cfg["seeds"]
    os.makedirs(LOG_DIR, exist_ok=True)

    K = int(cfg["K"])
    W = int(cfg["W"])
    V = int(cfg["V"])
    encoder = cfg.get("tokenizer", {}).get("encoder", "greedy")

    results_per_seed = []
    for s in seeds:
        if verbose:
            print(f"  [{name}] seed={s}", flush=True)
        r = _run_single_seed(cfg, s, device, verbose=verbose)
        results_per_seed.append(r)

        npz_path = os.path.join(LOG_DIR, f"{name}_seed{s}.npz")
        np.savez_compressed(
            npz_path,
            y_per_bit=r["y_per_bit"].astype(np.float32),
            y_bits_processed=r["y_bits_processed"].astype(np.int64),
            z_per_token_bits=r["z_per_token_bits"].astype(np.float32),
            z_per_bit=r["z_per_bit"].astype(np.float32),
            z_bits_processed_per_step=r["z_bits_processed_per_step"].astype(np.int64),
            z_bits_processed=r["z_bits_processed"].astype(np.int64),
            mean_token_bits=np.float32(r["mean_token_bits"]),
            token_bits=r["token_bits"].astype(np.int64),
            K=np.int32(K),
            W=np.int32(W),
            V=np.int32(V),
            alpha=np.float32(cfg["alpha"]),
            ce_window=np.int32(cfg["train"]["ce_window"]),
            encoder=np.bytes_(encoder.encode("utf-8")),
            num_parameters_Y=np.int64(r["num_parameters_Y"]),
            num_parameters_Z=np.int64(r["num_parameters_Z"]),
        )
        if verbose:
            print(f"    saved {npz_path}", flush=True)

    tail_frac = 0.1

    def _tail_mean(arr):
        n_tail = max(1, int(tail_frac * len(arr)))
        return float(np.mean(arr[-n_tail:]))

    y_plat = float(np.mean([_tail_mean(r["y_per_bit"]) for r in results_per_seed]))
    y_plat_std = float(np.std([_tail_mean(r["y_per_bit"]) for r in results_per_seed]))
    z_plat = float(np.mean([_tail_mean(r["z_per_bit"]) for r in results_per_seed]))
    z_plat_std = float(np.std([_tail_mean(r["z_per_bit"]) for r in results_per_seed]))
    mean_token_bits = float(
        np.mean([r["mean_token_bits"] for r in results_per_seed])
    )

    return {
        "name": name,
        "K": K, "W": W, "V": V, "alpha": float(cfg["alpha"]),
        "encoder": encoder,
        "seeds": list(seeds),
        "ce_window": int(cfg["train"]["ce_window"]),
        "mean_token_bits": mean_token_bits,
        "y_plateau_mean_bpb": y_plat, "y_plateau_std_bpb": y_plat_std,
        "z_plateau_mean_bpb": z_plat, "z_plateau_std_bpb": z_plat_std,
        # gap = Y_plat - Z_plat ; positive => Z-LM compresses better per bit.
        "gap_empirical_bpb": y_plat - z_plat,
        "num_parameters_Y": int(results_per_seed[0]["num_parameters_Y"]),
        "num_parameters_Z": int(results_per_seed[0]["num_parameters_Z"]),
        "_per_seed": results_per_seed,
    }


def _load_cfg(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    expected = "tokenization_transformer"
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
            "K": res["K"], "W": res["W"], "V": res["V"], "alpha": res["alpha"],
            "encoder": res["encoder"],
            "ce_window": res["ce_window"],
            "mean_token_bits": res["mean_token_bits"],
            "seeds":                   list(res["seeds"]),
            "y_per_bit_seeds":         [r["y_per_bit"]         for r in res["_per_seed"]],
            "y_bits_processed_seeds":  [r["y_bits_processed"]  for r in res["_per_seed"]],
            "z_per_bit_seeds":         [r["z_per_bit"]         for r in res["_per_seed"]],
            "z_bits_processed_seeds":  [r["z_bits_processed"]  for r in res["_per_seed"]],
        })

    summary_path = os.path.join(OUT_ROOT, "summary.json")
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    stem = (
        "tokenization_transformer_all" if len(all_cfg_data) > 1
        else f"tokenization_transformer_{all_cfg_data[0]['name']}"
    )
    paths = plot_configs(all_cfg_data, FIG_MAIN, FIG_APPX, file_stem=stem)

    print("\n" + "=" * 72)
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
    print(f"Total runtime:      {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
