"""Entry point: text_transformer experiment — prepare data, train, and plot.

Usage:
    python scripts/run_text_transformer.py
        # default: bytes_enwik8 + tokens_enwik8 configs
    python scripts/run_text_transformer.py configs/text_transformer/bytes_enwik8.yaml
    python scripts/run_text_transformer.py --smoke
        # tiny temp corpus, no enwik8 required
    python scripts/run_text_transformer.py --seeds 0 --device cpu

Artifacts (per config):
    outputs/text_transformer/<stem>_bytes/ or <stem>_v<vocab>/
        data/                  -- bytes_uint8.npy or tokens + tokenizer
        manifest.json
        resolved_config.yaml
        seeds/seed_<N>.npz     -- per-seed loss arrays
    outputs/text_transformer/figures/
        main/, appendix/, grid/, seed0/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.text_transformer.prepare import prepare_data, run_dir_for_config  # noqa: E402
from experiments.text_transformer.train import train_model     # noqa: E402
from experiments.text_transformer.plot import plot_configs     # noqa: E402

DEFAULT_CONFIGS = [
    os.path.join(ROOT, "configs", "text_transformer", "bytes_enwik8.yaml"),
    os.path.join(ROOT, "configs", "text_transformer", "tokens_enwik8.yaml"),
]

FIGURES_ROOT = os.path.join(ROOT, "outputs", "text_transformer", "figures")


def _smoke_config_path():
    body = (b"ab " * 2000) + b"\x00\xff"
    data_tmp = tempfile.NamedTemporaryFile(
        prefix="text_transformer_smoke_", suffix=".bin", delete=False
    )
    try:
        data_tmp.write(body)
        data_tmp.flush()
        data_path = data_tmp.name
    finally:
        data_tmp.close()

    yml = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="text_transformer_smoke_",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    )
    try:
        yml.write(
            textwrap.dedent(
                f"""
                experiment: text_transformer
                name: smoke
                data:
                  path: {data_path}
                  vocab_size: null
                train:
                  batch_size: 4
                  seq_len: 8
                  seeds: [0]
                  residual: null
                  lr: 3.0e-4
                  weight_decay: 0.01
                  betas: [0.9, 0.95]
                  warmup_steps: 5
                  ce_window: 10
                  grad_clip: 1.0
                model:
                  n_layers: 1
                  d_embd: 32
                  d_ffn: 64
                  n_heads: 2
                  init_std: 0.02
                """
            ).lstrip()
        )
    finally:
        yml.close()
    return yml.name


def _load_token_byte_lengths(manifest: dict, run_dir: Path, vocab_size: int) -> np.ndarray:
    """Return per-token byte-lengths for the given run.

    bytes mode: all 1s (256 elements).
    tokens mode: derived from the HF ByteLevel tokenizer vocab.  In HF's
    byte-to-unicode mapping each character in a token string represents exactly
    one source byte, so len(token_str) == byte count for that token.
    """
    if manifest["mode"] == "bytes":
        return np.ones(vocab_size, dtype=np.int64)

    from tokenizers import Tokenizer  # noqa: PLC0415

    tok_path = run_dir / manifest["tokenizer_json"]
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab = tokenizer.get_vocab()  # {token_str: id}
    byte_lengths = np.ones(vocab_size, dtype=np.int64)
    for token_str, token_id in vocab.items():
        if 0 <= token_id < vocab_size:
            byte_lengths[token_id] = len(token_str)
    return byte_lengths


def _mean_token_bytes(
    token_byte_lengths: np.ndarray, data: np.ndarray, vocab_size: int
) -> float:
    counts = np.bincount(data.astype(np.int64), minlength=vocab_size).astype(np.float64)
    freq = counts / counts.sum()
    return float(np.sum(freq * token_byte_lengths.astype(np.float64)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="text_transformer: prepare + train + plot")
    ap.add_argument(
        "configs",
        nargs="*",
        default=None,
        help="YAML config paths (default: bytes_enwik8 + tokens_enwik8)",
    )
    ap.add_argument("--smoke", action="store_true", help="Quick run on tiny temp corpus")
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override seeds from config (e.g. --seeds 0 1 2)",
    )
    ap.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    ap.add_argument(
        "--file-stem",
        default="enwik8",
        help="Base filename for output figures (default: enwik8)",
    )
    ap.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip data preparation; load existing manifest from the run directory",
    )
    ap.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run data preparation only; skip training and plotting",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation (useful when running individual seeds)",
    )
    args = ap.parse_args(argv)

    if args.smoke:
        cfg_paths = [_smoke_config_path()]
    elif args.configs:
        cfg_paths = args.configs
    else:
        cfg_paths = DEFAULT_CONFIGS

    root_path = Path(ROOT)
    all_cfg_data = []

    for cfg_arg in cfg_paths:
        cfg_path = Path(cfg_arg)
        if not cfg_path.is_absolute():
            cfg_path = (root_path / cfg_arg).resolve()
        else:
            cfg_path = cfg_path.resolve()

        # ── Phase 1: prepare data ──────────────────────────────────────────
        if args.skip_prepare:
            with open(cfg_path, encoding="utf-8") as _f:
                _cfg_tmp = yaml.safe_load(_f)
            run_dir = run_dir_for_config(_cfg_tmp, root_path)
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Manifest not found at {manifest_path}; "
                    "run without --skip-prepare first"
                )
            with open(manifest_path, encoding="utf-8") as _f:
                manifest = json.load(_f)
            print(f"\n[prepare] {cfg_path.name} (skipped — loaded {manifest_path})")
        else:
            print(f"\n[prepare] {cfg_path.name}")
            prep = prepare_data(root_path, cfg_path)
            manifest = prep["manifest"]
            run_dir = Path(manifest["run_dir"])
            print(json.dumps(manifest, indent=2))

        if args.prepare_only:
            continue

        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        t_cfg = cfg["train"]
        seq_len = int(t_cfg.get("seq_len", 16))
        seeds = args.seeds if args.seeds is not None else list(t_cfg.get("seeds", [0]))
        vocab_size = int(manifest["vocab_size_actual"])
        mode = manifest["mode"]

        # ── Load data ──────────────────────────────────────────────────────
        seq_npy = run_dir / manifest["sequence_npy"]
        data = np.load(seq_npy)

        token_byte_lengths = _load_token_byte_lengths(manifest, run_dir, vocab_size)
        mean_tb = _mean_token_bytes(token_byte_lengths, data, vocab_size)

        # ── Phase 2: train per seed ────────────────────────────────────────
        seeds_dir = run_dir / "seeds"
        seeds_dir.mkdir(exist_ok=True)

        loss_bits_seeds = []
        bytes_processed_seeds = []

        for seed in seeds:
            npz_path = seeds_dir / f"seed_{seed}.npz"
            if npz_path.exists():
                print(f"  [seed {seed}] loading cached {npz_path.name}")
                saved = np.load(npz_path)
                result = {k: saved[k] for k in saved.files}
            else:
                desc = f"{cfg_path.stem} seed={seed}"
                print(f"  [seed {seed}] training …")
                result = train_model(
                    data=data,
                    vocab_size=vocab_size,
                    seq_len=seq_len,
                    token_byte_lengths=token_byte_lengths,
                    cfg=cfg,
                    seed=seed,
                    device=args.device,
                    progress_desc=desc,
                )
                np.savez(
                    npz_path,
                    last_pos_loss_bits=result["last_pos_loss_bits"],
                    cumulative_loss_bits=result["cumulative_loss_bits"],
                    bytes_processed=result["bytes_processed"],
                    bytes_per_step=result["bytes_per_step"],
                    seed=np.int64(seed),
                    S=np.int64(result["S"]),
                    num_parameters=np.int64(result["num_parameters"]),
                )
                print(
                    f"  [seed {seed}] S={result['S']}, "
                    f"params={result['num_parameters']:,}, "
                    f"final_loss={result['last_pos_loss_bits'][-1]:.4f} bits"
                )

            loss_bits_seeds.append(result["last_pos_loss_bits"])
            bytes_processed_seeds.append(result["bytes_processed"])

        all_cfg_data.append(
            {
                "name": cfg.get("name", cfg_path.stem),
                "mode": mode,
                "vocab_size": vocab_size,
                "seq_len": seq_len,
                "ce_window": int(t_cfg.get("ce_window", 1000)),
                "mean_token_bytes": mean_tb,
                "seeds": seeds,
                "loss_bits_seeds": loss_bits_seeds,
                "bytes_processed_seeds": bytes_processed_seeds,
            }
        )

    # ── Phase 3: plot ──────────────────────────────────────────────────────
    if args.prepare_only or args.no_plot:
        print("Skipping plot generation.")
        return

    if len(all_cfg_data) == 0:
        print("No configs processed; nothing to plot.")
        return

    file_stem = args.file_stem
    figures = plot_configs(
        all_cfg_data,
        save_dir_main=os.path.join(FIGURES_ROOT, "main"),
        save_dir_appendix=os.path.join(FIGURES_ROOT, "appendix"),
        file_stem=file_stem,
        save_dir_grid=os.path.join(FIGURES_ROOT, "grid"),
        save_dir_single=os.path.join(FIGURES_ROOT, "seed0"),
    )
    print("\n[figures]")
    for name, path in figures.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
