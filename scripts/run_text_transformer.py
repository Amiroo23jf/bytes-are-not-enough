"""Entry point: text_transformer experiment — phase 1 prepares numpy byte/token data.

Usage:
    python scripts/run_text_transformer.py
        # default: configs/text_transformer/bytes_enwik8.yaml
    python scripts/run_text_transformer.py configs/text_transformer/tokens_enwik8.yaml
    python scripts/run_text_transformer.py --smoke
        # uses a tiny temp file (no enwik8 required)

Artifacts (per config):
    outputs/text_transformer/<stem>_bytes/   or  <stem>_v<vocab>/
        data/                 -- bytes_uint8.npy or tokens + tokenizer (from text_pipeline)
        manifest.json
        resolved_config.yaml
"""

import argparse
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.text_transformer.prepare import prepare_data  # noqa: E402

DEFAULT_CONFIG = os.path.join(ROOT, "configs", "text_transformer", "bytes_enwik8.yaml")


def _smoke_config_path():
    body = (b"ab " * 2000) + b"\x00\xff"
    data_tmp = tempfile.NamedTemporaryFile(
        prefix="text_transformer_smoke_",
        suffix=".bin",
        delete=False,
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
                  residual: null
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


def main(argv=None):
    ap = argparse.ArgumentParser(description="text_transformer phase-1 data preparation")
    ap.add_argument(
        "configs",
        nargs="*",
        default=None,
        help="YAML config paths (default: configs/text_transformer/bytes_enwik8.yaml)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny temp corpus + byte-mode config (no enwik8)",
    )
    args = ap.parse_args(argv)

    if args.smoke:
        paths = [_smoke_config_path()]
    elif args.configs:
        paths = args.configs
    else:
        paths = [DEFAULT_CONFIG]

    root_path = Path(ROOT)
    for cfg_arg in paths:
        cfg_path = Path(cfg_arg)
        if not cfg_path.is_absolute():
            cfg_path = (root_path / cfg_arg).resolve()
        else:
            cfg_path = cfg_path.resolve()
        out = prepare_data(root_path, cfg_path)
        print(json.dumps(out["manifest"], indent=2))
        print("run_dir:", out["run_dir"])


if __name__ == "__main__":
    main()
