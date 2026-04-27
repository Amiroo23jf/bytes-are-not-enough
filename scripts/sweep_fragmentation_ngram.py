"""Sweep fragmentation_ngram over Dirichlet alpha and context offset K - k.

Usage:
    python scripts/sweep_fragmentation_ngram.py

Saves per-cell artifacts plus an aggregate figure / CSV / JSON under
    outputs/fragmentation_ngram/{figures,raw}/
"""

import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.fragmentation_ngram.sweep import run_alpha_sweep  # noqa: E402

BASE_CONFIG = os.path.join(ROOT, "configs", "fragmentation_ngram", "base.yaml")
OUT_DIR = os.path.join(ROOT, "outputs", "fragmentation_ngram")

ALPHAS = [0.1, 0.5, 2.0, 5.0]
K_OFFSETS = [0, 1]
N_MATRICES = 20
N_SEQUENCE = 1_000_000


def main():
    with open(BASE_CONFIG, "r") as f:
        base_cfg = yaml.safe_load(f)
    run_alpha_sweep(
        base_cfg,
        out_dir=OUT_DIR,
        alphas=ALPHAS,
        K_offsets=K_OFFSETS,
        n_matrices=N_MATRICES,
        n_sequence=N_SEQUENCE,
    )


if __name__ == "__main__":
    main()
