"""Entry point for the wikitext_span_transformer experiment.

Two experiments, both driven by the same script (`window.mode` in the
config selects between them):

  Exp 1 (configs/wikitext_span_transformer/exp1.yaml): fixed token window w
  for every tokenizer; measures each tokenizer's resulting character span
  w_s and end-to-end held-out loss.

  Exp 2 (configs/wikitext_span_transformer/exp2.yaml): fixed target
  character span w_s; solves per-tokenizer for the token window w that
  reaches it, then measures end-to-end held-out loss.

Stages (run separately so SLURM can parallelize across tokenizers):

    python scripts/run_wikitext_span_transformer.py <config.yaml> --stage prepare [--tokenizer NAME]
    python scripts/run_wikitext_span_transformer.py <config.yaml> --stage train   [--tokenizer NAME]
    python scripts/run_wikitext_span_transformer.py <config.yaml> --stage plot

Omitting --tokenizer processes every tokenizer in the config sequentially.

A tokenizer's resolved (w) result is cached under
outputs/wikitext_span_transformer/results/<tokenizer>/w<W>.npz, keyed by
tokenizer + resolved window (not by experiment name) -- so if two
experiments resolve to the same w for a tokenizer (this always happens for
`char`, whose 1-char-per-token ratio makes both experiments resolve to
w=128), the second one reuses the cached result instead of re-training.

Artifacts:
    outputs/wikitext_span_transformer/data/<tokenizer>/
        ids_{train,validation,test}.npy, char_lens_{train,validation,test}.npy,
        manifest.json
    outputs/wikitext_span_transformer/artifacts/         -- tokenizer files
    outputs/wikitext_span_transformer/results/<tokenizer>/w<W>.npz
    outputs/wikitext_span_transformer/figures/<exp_name>.{pdf,png}
    outputs/wikitext_span_transformer/summary_<exp_name>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

# Import HuggingFace datasets BEFORE adding project root to sys.path, because
# the project has its own datasets/ package that would otherwise shadow it.
from datasets import load_dataset as _hf_load_dataset  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from experiments.wikitext_span_transformer.data import (  # noqa: E402
    SPLITS,
    load_prepared_tokenizer_data,
    prepare_tokenizer_data,
    solve_w_for_target_span,
)
from experiments.wikitext_span_transformer.plot import (  # noqa: E402
    plot_exp1,
    plot_exp2,
    plot_training_curves,
)
from experiments.wikitext_span_transformer.train import evaluate_model, train_model  # noqa: E402
from experiments.wikitext_tokenizers.analysis import compute_span_distribution  # noqa: E402

OUT_ROOT = os.path.join(ROOT, "outputs", "wikitext_span_transformer")
DATA_ROOT = os.path.join(OUT_ROOT, "data")
ARTIFACTS_DIR = os.path.join(OUT_ROOT, "artifacts")
RESULTS_ROOT = os.path.join(OUT_ROOT, "results")
FIG_DIR = os.path.join(OUT_ROOT, "figures")

_SOLVE_SAMPLE_CAP = 20_000_000  # for measuring w_s at a fixed w (Exp 1 side)


def _load_wikitext103_splits(smoke=False):
    """Load WikiText-103 train/validation/test as separate text blobs."""
    print("Loading WikiText-103 ...", flush=True)
    ds = _hf_load_dataset("wikitext", "wikitext-103-raw-v1")

    def _join(split):
        return "\n".join(line for line in ds[split]["text"] if line.strip())

    splits_text = {}
    for split in SPLITS:
        text = _join(split)
        if smoke:
            text = text[:200_000] if split == "train" else text[:50_000]
        splits_text[split] = text
        print(f"  {split}: {len(text):,} chars", flush=True)
    return splits_text


def _load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    found = cfg.get("experiment")
    if found is not None and found != "wikitext_span_transformer":
        raise ValueError(f"Config experiment={found!r} does not match 'wikitext_span_transformer'")
    return cfg


def _resolve_window(window_cfg, manifest, char_lens_train):
    """Return (w, w_s) for one tokenizer under this config's window rule."""
    mode = window_cfg["mode"]
    vocab_size = manifest["vocab_size"]
    ratio = manifest["ratio"]
    slack = float(window_cfg["slack"])

    if mode == "fixed_w":
        w = int(window_cfg["w"])
        sample = char_lens_train[:_SOLVE_SAMPLE_CAP]
        pmf, cdf, ws_values, mean_span = compute_span_distribution(sample, w)
        scale = np.log2(vocab_size) / ratio
        scaled = cdf * scale
        if len(ws_values) == 1:
            w_s = float(ws_values[0])
        elif slack <= scaled[0]:
            w_s = float(ws_values[0])
        elif slack >= scaled[-1]:
            w_s = float(ws_values[-1])
        else:
            w_s = float(np.interp(slack, scaled, ws_values))
        return w, w_s

    if mode == "fixed_w_s":
        w_s_target = float(window_cfg["w_s"])
        w = solve_w_for_target_span(char_lens_train, vocab_size, ratio, w_s_target, slack)
        return w, w_s_target

    raise ValueError(f"Unknown window mode: {mode!r}")


def _result_path(tok_name, w):
    return os.path.join(RESULTS_ROOT, tok_name, f"w{w}.npz")


def stage_prepare(cfg, tokenizer_filter, smoke):
    splits_text = _load_wikitext103_splits(smoke=smoke)
    tok_cfgs = cfg["tokenizers"]
    if tokenizer_filter:
        tok_cfgs = [t for t in tok_cfgs if t["name"] == tokenizer_filter]
        if not tok_cfgs:
            raise ValueError(f"No tokenizer named {tokenizer_filter!r} in config")

    for tok_cfg in tok_cfgs:
        name = tok_cfg["name"]
        print(f"\n[prepare] {name}", flush=True)
        data_dir = os.path.join(DATA_ROOT, name)
        prepare_tokenizer_data(tok_cfg, splits_text, data_dir, ARTIFACTS_DIR)


def stage_train(cfg, tokenizer_filter, device, seed):
    window_cfg = cfg["window"]
    tok_cfgs = cfg["tokenizers"]
    if tokenizer_filter:
        tok_cfgs = [t for t in tok_cfgs if t["name"] == tokenizer_filter]
        if not tok_cfgs:
            raise ValueError(f"No tokenizer named {tokenizer_filter!r} in config")

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    for tok_cfg in tok_cfgs:
        name = tok_cfg["name"]
        data_dir = os.path.join(DATA_ROOT, name)
        manifest, arrays = load_prepared_tokenizer_data(data_dir)

        w, w_s = _resolve_window(window_cfg, manifest, arrays["char_lens_train"])
        print(f"\n[train] {name}: vocab={manifest['vocab_size']} ratio={manifest['ratio']:.3f} "
              f"-> w={w}, w_s={w_s:.2f}", flush=True)

        result_path = _result_path(name, w)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        if os.path.exists(result_path):
            print(f"  reusing cached result at {result_path} (same tokenizer+w already trained)",
                  flush=True)
            continue

        out = train_model(
            data=arrays["ids_train"], vocab_size=manifest["vocab_size"], seq_len=w,
            cfg=cfg, seed=seed, device=torch.device(device),
            progress_desc=f"{name} w={w}",
        )
        test_loss_bits_per_token, test_loss_bits_per_char = evaluate_model(
            out["model"], arrays["ids_test"], arrays["char_lens_test"], manifest["vocab_size"], w,
            device=torch.device(device),
            amp_dtype=cfg["train"].get("amp_dtype", "bfloat16"),
            batch_size=int(cfg["train"].get("eval_batch_size", cfg["train"]["batch_size"])),
        )

        print(f"  [{name}] test loss: {test_loss_bits_per_token:.4f} bits/token "
              f"= {test_loss_bits_per_char:.4f} bits/char", flush=True)

        np.savez_compressed(
            result_path,
            name=name, vocab_size=manifest["vocab_size"], ratio=manifest["ratio"],
            w=w, w_s=w_s,
            last_pos_loss_bits=out["last_pos_loss_bits"],
            test_loss_bits_per_token=test_loss_bits_per_token,
            test_loss_bits_per_char=test_loss_bits_per_char,
            num_parameters=out["num_parameters"],
        )
        print(f"  saved {result_path}", flush=True)


def stage_plot(cfg):
    window_cfg = cfg["window"]
    tok_cfgs = cfg["tokenizers"]
    mode = window_cfg["mode"]

    # If a previous run of this stage already recorded each tokenizer's
    # resolved w in summary_<name>.json, prefer that -- it lets replotting
    # (e.g. after pulling back only the small results/ directory, without
    # the much larger data/ arrays) work without needing data/ at all.
    # Only falls back to re-resolving from data/ (fixed_w_s's binary search
    # needs char_lens_train) for tokenizers missing from that cache.
    prior_w = {}
    prior_summary_path = os.path.join(OUT_ROOT, f"summary_{cfg['name']}.json")
    if os.path.exists(prior_summary_path):
        with open(prior_summary_path) as f:
            prior_w = {r["name"]: int(r["w"]) for r in json.load(f)}

    results = []
    training_results = []
    for tok_cfg in tok_cfgs:
        name = tok_cfg["name"]

        if mode == "fixed_w":
            # w is already known from config; no need to load data/ at all
            # -- w_s is read from the cached result below (computed once,
            # at train time).
            w = int(window_cfg["w"])
        elif name in prior_w and os.path.exists(_result_path(name, prior_w[name])):
            # Ignore a prior w whose result no longer exists (e.g. it was
            # deleted to retrain after re-preparing that tokenizer's data).
            w = prior_w[name]
        else:
            data_dir = os.path.join(DATA_ROOT, name)
            try:
                manifest, arrays = load_prepared_tokenizer_data(data_dir)
            except FileNotFoundError:
                print(f"  [skip] no prepared data for {name} ({data_dir}) "
                      "and not in a prior summary", flush=True)
                continue
            w, _ = _resolve_window(window_cfg, manifest, arrays["char_lens_train"])

        result_path = _result_path(name, w)
        if not os.path.exists(result_path):
            print(f"  [skip] no result for {name} at w={w} ({result_path})", flush=True)
            continue
        r = np.load(result_path, allow_pickle=True)
        results.append({
            "name": name,
            "w": int(r["w"]),
            "w_s": float(r["w_s"]),
            "test_loss_bits_per_char": float(r["test_loss_bits_per_char"]),
        })
        training_results.append({
            "name": name,
            "w": int(r["w"]),
            "ratio": float(r["ratio"]),
            "last_pos_loss_bits": r["last_pos_loss_bits"],
        })

    if not results:
        print("No results found; run the 'train' stage first.")
        return

    name = cfg["name"]
    if window_cfg["mode"] == "fixed_w":
        paths = plot_exp1(results, FIG_DIR, file_stem=f"wikitext_span_{name}")
    else:
        paths = plot_exp2(results, FIG_DIR, file_stem=f"wikitext_span_{name}")

    t_cfg = cfg["train"]
    train_paths = plot_training_curves(
        training_results, FIG_DIR, file_stem=f"wikitext_span_{name}_training",
        batch_size=int(t_cfg["batch_size"]), ce_window=int(t_cfg["ce_window"]),
        title=f"{name}: training loss vs. characters seen",
    )

    summary_path = os.path.join(OUT_ROOT, f"summary_{name}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFigure: {paths[0]}")
    print(f"         {paths[1]}")
    print(f"Training curves: {train_paths[0]}")
    print(f"                 {train_paths[1]}")
    print(f"Summary: {summary_path}")
    print()
    print(f"{'tokenizer':<20}{'w':>6}{'w_s':>10}{'loss (bits/char)':>20}")
    for r in sorted(results, key=lambda x: x["w_s"]):
        print(f"{r['name']:<20}{r['w']:>6}{r['w_s']:>10.2f}{r['test_loss_bits_per_char']:>20.4f}")


def main():
    ap = argparse.ArgumentParser(description="wikitext_span_transformer: prepare/train/plot")
    ap.add_argument("config", help="YAML config path (exp1.yaml or exp2.yaml)")
    ap.add_argument("--stage", choices=["prepare", "train", "plot"], required=True)
    ap.add_argument("--tokenizer", default=None, help="Restrict to one tokenizer by name")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="Truncated corpus, prepare stage only")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    cfg = _load_cfg(cfg_path)

    if args.stage == "prepare":
        stage_prepare(cfg, args.tokenizer, smoke=args.smoke)
    elif args.stage == "train":
        stage_train(cfg, args.tokenizer, device=args.device, seed=args.seed)
    else:
        stage_plot(cfg)


if __name__ == "__main__":
    main()
