"""Entry point for the fragmentation_ngram experiment (paper Theorem 3.1).

Usage:
    python scripts/run_fragmentation_ngram.py
        # uses configs/fragmentation_ngram/base.yaml
    python scripts/run_fragmentation_ngram.py configs/fragmentation_ngram/Kp1.yaml
        # K = k + 1 variant

Output filenames are derived from the config basename so different variants
do not overwrite each other's results.
"""

import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.fragmentation_ngram.run import run_from_config  # noqa: E402

DEFAULT_CONFIG = os.path.join(ROOT, "configs", "fragmentation_ngram", "base.yaml")
OUT_DIR = os.path.join(ROOT, "outputs", "fragmentation_ngram")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)
    base = os.path.splitext(os.path.basename(config_path))[0]
    tag = f"fragmentation_ngram_{base}"
    print(f"Config: {config_path}")
    print(f"Tag:    {tag}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    expected = "fragmentation_ngram"
    found = cfg.get("experiment")
    if found is not None and found != expected:
        raise ValueError(
            f"Config 'experiment: {found!r}' does not match expected {expected!r}"
        )

    run_from_config(cfg, tag, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
