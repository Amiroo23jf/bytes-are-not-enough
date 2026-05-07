"""Entry point for the WikiText-103 tokenizer span-distribution experiment.

For each tokenizer defined in the config:
  1. Build (and train if needed) the tokenizer on the WikiText-103 train split.
  2. Encode the full text (train + validation + test) to get per-token char lengths.
  3. For each window size W, compute P(span(Z^W) = ws) and its CDF.
  4. Save results.

Artifacts
---------
  outputs/wikitext_tokenizers/results/{name}.npz
      pmf_W{w}, cdf_W{w}, ws_W{w}, mean_span_W{w}  for each w in W
      vocab_size, ratio, num_chars, num_tokens, name
  outputs/wikitext_tokenizers/artifacts/
      SentencePiece .model files for trained tokenizers
  outputs/wikitext_tokenizers/summary.json

Usage
-----
  python scripts/run_wikitext_tokenizers.py
  python scripts/run_wikitext_tokenizers.py path/to/config.yaml
  python scripts/run_wikitext_tokenizers.py --smoke
"""

import json
import os
import sys
import time

import numpy as np
import yaml

# Import HuggingFace datasets BEFORE adding project root to sys.path, because
# the project has its own datasets/ package that would otherwise shadow it.
from datasets import load_dataset as _hf_load_dataset  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.wikitext_tokenizers.build_tokenizers import build_tokenizer  # noqa: E402
from experiments.wikitext_tokenizers.analysis import compute_span_distribution  # noqa: E402

OUT_ROOT = os.path.join(ROOT, "outputs", "wikitext_tokenizers")
RESULTS_DIR = os.path.join(OUT_ROOT, "results")
ART_DIR = os.path.join(OUT_ROOT, "artifacts")

DEFAULT_CONFIG = os.path.join(
    ROOT, "configs", "wikitext_tokenizers", "wikitext103.yaml"
)


def _load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    found = cfg.get("experiment")
    if found is not None and found != "wikitext_tokenizers":
        raise ValueError(
            f"Config experiment={found!r} does not match 'wikitext_tokenizers'"
        )
    return cfg


def _load_wikitext103(splits=("train", "validation", "test"), smoke=False):
    """Load WikiText-103 and return (train_text, full_text)."""
    print("Loading WikiText-103 ...", flush=True)
    ds = _hf_load_dataset("wikitext", "wikitext-103-raw-v1")

    def _join(split):
        return "\n".join(line for line in ds[split]["text"] if line.strip())

    train_text = _join("train")
    if smoke:
        train_text = train_text[:200_000]

    parts = []
    for split in splits:
        text = _join(split)
        if smoke:
            text = text[:100_000]
        parts.append(text)
    full_text = "\n".join(parts)

    print(
        f"  train: {len(train_text):,} chars  |  full ({'+'.join(splits)}): {len(full_text):,} chars",
        flush=True,
    )
    return train_text, full_text


def run(cfg, smoke=False, verbose=True):
    name = cfg["name"]
    W_list = [int(w) for w in cfg["W"]]
    splits = list(cfg.get("data", {}).get("splits", ["train", "validation", "test"]))
    tok_cfgs = cfg["tokenizers"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ART_DIR, exist_ok=True)

    train_text, full_text = _load_wikitext103(splits=splits, smoke=smoke)
    num_chars = len(full_text)

    summary = []

    for tok_cfg in tok_cfgs:
        tok_name = tok_cfg["name"]
        if verbose:
            print(f"\n[{name}] tokenizer={tok_name}", flush=True)

        # Build / train tokenizer on train split.
        t0 = time.time()
        tok = build_tokenizer(tok_cfg, train_text, ART_DIR)
        if verbose:
            print(
                f"  [{tok_name}] tokenizer ready in {time.time() - t0:.1f}s  "
                f"vocab_size={tok.vocab_size:,}",
                flush=True,
            )

        # Encode full text.
        if verbose:
            print(f"  [{tok_name}] encoding {num_chars:,} chars ...", flush=True)
        t1 = time.time()
        ids, char_lens = tok.encode_with_char_lengths(full_text)
        num_tokens = len(char_lens)
        ratio = num_chars / num_tokens
        del ids  # only char_lens is needed from here on; free memory
        if verbose:
            print(
                f"  [{tok_name}] encoded in {time.time() - t1:.1f}s  "
                f"tokens={num_tokens:,}  ratio={ratio:.4f}",
                flush=True,
            )

        # Compute span distributions for each W.
        dist_results = {}
        for w in W_list:
            if num_tokens < w:
                if verbose:
                    print(f"  [{tok_name}] SKIP W={w}: too few tokens", flush=True)
                continue
            t2 = time.time()
            pmf, cdf, ws_values, mean_span = compute_span_distribution(char_lens, w)
            dist_results[w] = (pmf, cdf, ws_values, mean_span)
            if verbose:
                print(
                    f"  [{tok_name}] W={w:4d}  mean_span={mean_span:.2f}  "
                    f"max_span={int(ws_values[-1])}  ({time.time() - t2:.2f}s)",
                    flush=True,
                )

        # Save per-tokenizer npz.
        npz_path = os.path.join(RESULTS_DIR, f"{tok_name}.npz")
        save_dict = {
            "vocab_size": np.int32(tok.vocab_size),
            "ratio": np.float32(ratio),
            "num_chars": np.int64(num_chars),
            "num_tokens": np.int64(num_tokens),
            "name": np.bytes_(tok_name.encode()),
            "W_list": np.array([w for w in W_list if w in dist_results], dtype=np.int32),
        }
        for w, (pmf, cdf, ws_values, mean_span) in dist_results.items():
            save_dict[f"pmf_W{w}"] = pmf
            save_dict[f"cdf_W{w}"] = cdf
            save_dict[f"ws_W{w}"] = ws_values
            save_dict[f"mean_span_W{w}"] = np.float32(mean_span)
        np.savez_compressed(npz_path, **save_dict)
        del char_lens  # free before next tokenizer
        if verbose:
            print(f"  [{tok_name}] saved {npz_path}", flush=True)

        summary.append({
            "name": tok_name,
            "type": tok_cfg["type"],
            "vocab_size": int(tok.vocab_size),
            "num_chars": int(num_chars),
            "num_tokens": int(num_tokens),
            "ratio": float(ratio),
            "mean_span": {w: float(ms) for w, (_, _, _, ms) in dist_results.items()},
        })

    # Save summary JSON.
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
        cfg["W"] = [4, 8, 16]

    t_all = time.time()
    summary = run(cfg, smoke=smoke, verbose=True)

    print("\n" + "=" * 72)
    print("Summary:")
    print(f"  {'tokenizer':<24}  {'vocab':>8}  {'ratio':>7}  {'mean_span(W=max)':>16}")
    W_list = [int(w) for w in cfg["W"]]
    for s in summary:
        last_w = max((w for w in W_list if w in s["mean_span"]), default=None)
        ms = f"{s['mean_span'][last_w]:.2f}" if last_w else "—"
        print(
            f"  {s['name']:<24}  {s['vocab_size']:>8,}  {s['ratio']:>7.4f}  {ms:>16}"
        )
    print(f"\nTotal runtime: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
