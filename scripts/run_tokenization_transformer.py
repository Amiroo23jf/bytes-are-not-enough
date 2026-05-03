"""Entry point for the Markov tokenization-transformer experiment.

For each seed in the config:
  1. Generate one binary Markov-K sequence Y.
  2. For each V in cfg["V"]: tokenize Y with BPE (or use Y directly if V==2).
  3. For each (V, W) combination: train a TransformerLM and save results.

All (W, V) combinations are swept for every seed. One Y is generated per
seed and shared across all (V, W) runs for that seed.

Artifacts
---------
  outputs/tokenization_transformer/logs/{name}_V{V}_W{W}_seed{seed}.npz
  outputs/tokenization_transformer/artifacts/{name}_V{V}_seed{seed}_bpe.json
  outputs/tokenization_transformer/summary.json

Usage
-----
  python scripts/run_tokenization_transformer.py
      # uses configs/tokenization_transformer/markov.yaml
  python scripts/run_tokenization_transformer.py path/to/config.yaml
      # explicit config
  python scripts/run_tokenization_transformer.py --smoke
      # quick smoke test (short sequence, 1 seed, few iterations)
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

from experiments.tokenization_transformer.data import build_sequences   # noqa: E402
from experiments.tokenization_transformer.train import train_model       # noqa: E402

OUT_ROOT = os.path.join(ROOT, "outputs", "tokenization_transformer")
LOG_DIR = os.path.join(OUT_ROOT, "logs")
ART_DIR = os.path.join(OUT_ROOT, "artifacts")

DEFAULT_CONFIG = os.path.join(
    ROOT, "configs", "tokenization_transformer", "markov.yaml"
)

TAIL_FRAC = 0.1


def _tail_mean(arr):
    n = max(1, int(TAIL_FRAC * len(arr)))
    return float(np.mean(arr[-n:]))


def _load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    found = cfg.get("experiment")
    if found is not None and found != "tokenization_transformer":
        raise ValueError(
            f"Config experiment={found!r} does not match 'tokenization_transformer'"
        )
    return cfg


def run(cfg, device, verbose=True):
    name = cfg["name"]
    K = int(cfg["K"])
    W_list = [int(w) for w in cfg["W"]]
    V_list = [int(v) for v in cfg["V"]]
    alpha = float(cfg["alpha"])
    n_sequence = int(cfg["n_sequence"])
    seeds = list(cfg["seeds"])
    tok_cfg = cfg.get("tokenizer", {})
    encoder = tok_cfg.get("encoder", "greedy")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(ART_DIR, exist_ok=True)

    # plateau_losses[(V, W)] collects one tail-mean per seed for the summary.
    plateau_losses = {(V, W): [] for V in V_list for W in W_list}
    num_params_map = {}          # (V, W) -> num_parameters (last seed wins)
    mean_ratios = {V: [] for V in V_list}
    entropy_rates = []

    for seed in seeds:
        if verbose:
            print(
                f"\n[{name}] seed={seed}: generating sequence and tokenizing ...",
                flush=True,
            )
        t0 = time.time()
        data = build_sequences(K, V_list, alpha, n_sequence, seed, tok_cfg)
        if verbose:
            print(f"  all sequences ready ({time.time() - t0:.1f}s total)", flush=True)

        entropy_rates.append(data["entropy_rate"])

        # Save BPE tokenizer artifacts for V > 2.
        for V in V_list:
            info = data["per_V"][V]
            mean_ratios[V].append(info["ratio"])
            if V > 2 and info["tokenizer"] is not None:
                art_path = os.path.join(
                    ART_DIR, f"{name}_V{V}_seed{seed}_bpe.json"
                )
                info["tokenizer"].save(art_path)

        # Train one model per (V, W) combination.
        for V in V_list:
            info = data["per_V"][V]
            Z = info["Z"]
            ratio = info["ratio"]
            token_bits = info["token_bits"]
            actual_V = info["actual_V"]

            for W in W_list:
                if len(Z) <= W:
                    if verbose:
                        print(
                            f"  SKIP V={V} W={W} seed={seed}: "
                            f"len(Z)={len(Z)} <= W",
                            flush=True,
                        )
                    continue

                desc = f"[{name} V={V} W={W} s={seed}]"
                if verbose:
                    print(f"  {desc} training ...", flush=True)
                t1 = time.time()

                out = train_model(
                    Z=Z,
                    ratio=ratio,
                    token_bits=token_bits,
                    V=actual_V,
                    W=W,
                    cfg=cfg,
                    seed=seed,
                    device=device,
                    progress_desc=desc,
                )

                plateau = _tail_mean(out["loss_per_iter"])
                if verbose:
                    print(
                        f"  {desc} done in {time.time() - t1:.1f}s  "
                        f"params={out['num_parameters']:,}  "
                        f"plateau={plateau:.4f} bpb",
                        flush=True,
                    )

                # Save per-(V, W, seed) log.
                npz_path = os.path.join(
                    LOG_DIR, f"{name}_V{V}_W{W}_seed{seed}.npz"
                )
                np.savez_compressed(
                    npz_path,
                    loss_per_iter=out["loss_per_iter"],
                    bits_seen_per_iter=out["bits_seen_per_iter"],
                    num_parameters=np.int64(out["num_parameters"]),
                    K=np.int32(K),
                    W=np.int32(W),
                    V=np.int32(V),
                    actual_V=np.int32(actual_V),
                    alpha=np.float32(alpha),
                    ratio=np.float32(ratio),
                    seed=np.int64(seed),
                    n_sequence=np.int64(n_sequence),
                    encoder=np.bytes_(
                        encoder.encode("utf-8") if V > 2 else b"none"
                    ),
                )
                if verbose:
                    print(f"    saved {npz_path}", flush=True)

                plateau_losses[(V, W)].append(plateau)
                num_params_map[(V, W)] = int(out["num_parameters"])

    # Build summary: one row per (V, W), plateau loss aggregated over seeds.
    summary = []
    for V in V_list:
        for W in W_list:
            tails = plateau_losses[(V, W)]
            if not tails:
                continue
            summary.append({
                "name": name,
                "K": K,
                "V": V,
                "W": W,
                "alpha": alpha,
                "encoder": encoder if V > 2 else "none",
                "seeds": list(seeds),
                "entropy_rate_mean": float(np.mean(entropy_rates)),
                "entropy_rate_std": float(np.std(entropy_rates)),
                "mean_ratio": float(np.mean(mean_ratios[V])),
                "plateau_loss_mean": float(np.mean(tails)),
                "plateau_loss_std": float(np.std(tails)),
                "num_parameters": num_params_map.get((V, W), 0),
            })

    os.makedirs(OUT_ROOT, exist_ok=True)
    summary_path = os.path.join(OUT_ROOT, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"\nSummary written to {summary_path}", flush=True)

    return summary


def main():
    args = sys.argv[1:]
    smoke = "--smoke" in args
    args = [a for a in args if a != "--smoke"]

    cfg_path = args[0] if args else DEFAULT_CONFIG
    cfg = _load_cfg(cfg_path)

    if smoke:
        cfg["n_sequence"] = 50_000
        cfg["seeds"] = [0]
        cfg["train"]["warmup_steps"] = 20
        cfg["train"]["iterations"] = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    t_all = time.time()
    summary = run(cfg, device, verbose=True)

    print("\n" + "=" * 72)
    print("Summary — normalized plateau loss (bits per source symbol):")
    print(
        f"  {'V':>5}  {'W':>5}  {'#params':>12}  "
        f"{'ratio':>7}  {'plat_mean':>10}  {'plat_std':>9}"
    )
    for s in summary:
        print(
            f"  {s['V']:>5}  {s['W']:>5}  {s['num_parameters']:>12,}  "
            f"{s['mean_ratio']:>7.4f}  {s['plateau_loss_mean']:>10.4f}  "
            f"{s['plateau_loss_std']:>9.4f}"
        )
    print(f"\nTotal runtime: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
