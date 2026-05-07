"""Entry point for the Markov tokenizer span-distribution experiment.

Generates a Markov-K binary sequence, trains BPE tokenizers for each V,
and computes P(span(Z^w) = w_s) for each window size w.

Usage
-----
  python scripts/run_markov_tokenizers.py
      # uses configs/markov_tokenizers/markov.yaml
  python scripts/run_markov_tokenizers.py path/to/config.yaml
"""

import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.markov_tokenizers.run import run  # noqa: E402

DEFAULT_CONFIG = os.path.join(ROOT, "configs", "markov_tokenizers", "markov.yaml")


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    found = cfg.get("experiment")
    if found is not None and found != "markov_tokenizers":
        sys.exit(f"Config experiment={found!r} does not match 'markov_tokenizers'")
    run(cfg, verbose=True)
    print("Done.")


if __name__ == "__main__":
    main()
