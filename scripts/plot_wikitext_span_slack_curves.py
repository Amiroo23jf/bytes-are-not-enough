"""Per-tokenizer slack curves at their w_s=128, slack=0.05 solved w.

For each of the 9 WikiText tokenizers, using the token window w that was
already solved (in the wikitext_span_transformer Exp 2 run) to reach
w_s=128 (chars) at slack=0.05, recomputes the FULL scaled-CDF span curve

    scaled(w_s) = P(span(Z^w) <= w_s) * log2(V) / ratio

(see wikitext_tokenizers.analysis.compute_span_distribution) and plots all
9 on one figure. Since each tokenizer's own w was solved specifically to
make its curve cross (w_s=128, slack=0.05), every curve should pass through
(or very near) that point -- this both visualizes each tokenizer's span
distribution *shape* around that point and sanity-checks the earlier
binary search.

Requires outputs/wikitext_span_transformer/data/<tokenizer>/{manifest.json,
char_lens_train.npy} (from the 'prepare' stage) and a summary JSON recording
each tokenizer's solved w (default: summary_exp2_fixed_w_s.json) -- but only
on the FIRST run: the computed per-tokenizer curves (ws_values, cdf, scale)
are cached to slack_curves_cache.npz, which is tiny (a few hundred KB, no
full char_lens arrays) and portable -- copy it anywhere and replot (e.g.
with --full-range, or a different --file-stem) without touching data/ again.

Usage:
    python scripts/plot_wikitext_span_slack_curves.py
    python scripts/plot_wikitext_span_slack_curves.py --full-range
    python scripts/plot_wikitext_span_slack_curves.py --recompute  # ignore cache
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.wikitext_span_transformer.plot import (  # noqa: E402
    plot_epsilon_curves,
    plot_slack_curves,
)
from experiments.wikitext_tokenizers.analysis import compute_span_distribution  # noqa: E402

OUT_ROOT = os.path.join(ROOT, "outputs", "wikitext_span_transformer")
DATA_ROOT = os.path.join(OUT_ROOT, "data")
FIG_DIR = os.path.join(OUT_ROOT, "figures")
DEFAULT_SUMMARY = os.path.join(OUT_ROOT, "summary_exp2_fixed_w_s.json")
CACHE_PATH = os.path.join(OUT_ROOT, "slack_curves_cache.npz")

TARGET_W_S = 128.0
TARGET_SLACK = 0.05


def _load_cache(path):
    cache = np.load(path, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    curves = []
    for name in names:
        curves.append({
            "name": name,
            "w": int(cache[f"w_{name}"]),
            "ws_values": cache[f"ws_{name}"],
            "cdf": cache[f"cdf_{name}"],
            "scale": float(cache[f"scale_{name}"]),
        })
    return curves


def _save_cache(path, curves):
    save_dict = {"names": np.array([c["name"] for c in curves])}
    for c in curves:
        save_dict[f"w_{c['name']}"] = c["w"]
        save_dict[f"ws_{c['name']}"] = c["ws_values"]
        save_dict[f"cdf_{c['name']}"] = c["cdf"]
        save_dict[f"scale_{c['name']}"] = c["scale"]
    np.savez_compressed(path, **save_dict)


def _compute_curves(summary_path):
    with open(summary_path) as f:
        summary = json.load(f)
    w_by_name = {r["name"]: int(r["w"]) for r in summary}
    print(f"Solved w per tokenizer (from {summary_path}):")
    for name, w in w_by_name.items():
        print(f"  {name:<18} w={w}")

    curves = []
    for name, w in w_by_name.items():
        data_dir = os.path.join(DATA_ROOT, name)
        with open(os.path.join(data_dir, "manifest.json")) as f:
            manifest = json.load(f)
        char_lens = np.load(os.path.join(data_dir, "char_lens_train.npy"))

        print(f"[{name}] computing span distribution at w={w} "
              f"({len(char_lens):,} train tokens) ...", flush=True)
        pmf, cdf, ws_values, mean_span = compute_span_distribution(char_lens, w)
        scale = float(np.log2(manifest["vocab_size"]) / manifest["ratio"])
        scaled_at_target = float(np.interp(TARGET_W_S, ws_values, cdf) * scale) \
            if len(ws_values) > 1 else scale
        print(f"  mean_span={mean_span:.2f}  scale={scale:.4f}  "
              f"scaled(w_s={TARGET_W_S:g})={scaled_at_target:.4f} "
              f"(target slack={TARGET_SLACK:g})", flush=True)

        curves.append({
            "name": name, "w": w, "ws_values": ws_values, "cdf": cdf, "scale": scale,
        })
    return curves


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=DEFAULT_SUMMARY,
                     help="Summary JSON recording each tokenizer's solved w")
    ap.add_argument("--file-stem", default="wikitext_span_slack_curves")
    ap.add_argument("--full-range", action="store_true",
                     help="Full auto y-range instead of zooming to the target-slack region")
    ap.add_argument("--recompute", action="store_true",
                     help="Ignore slack_curves_cache.npz and recompute from data/")
    ap.add_argument("--epsilon", action="store_true",
                     help="Plot the raw CDF (unscaled) instead of the scaled slack curve")
    args = ap.parse_args()

    if os.path.exists(CACHE_PATH) and not args.recompute:
        print(f"Loading cached curves from {CACHE_PATH} (pass --recompute to refresh)", flush=True)
        curves = _load_cache(CACHE_PATH)
    else:
        curves = _compute_curves(args.summary)
        _save_cache(CACHE_PATH, curves)
        print(f"Cached curves to {CACHE_PATH}", flush=True)

    if args.epsilon:
        pdf, png = plot_epsilon_curves(curves, FIG_DIR, file_stem=args.file_stem,
                                        target_w_s=TARGET_W_S)
    else:
        pdf, png = plot_slack_curves(
            curves, FIG_DIR, file_stem=args.file_stem,
            target_w_s=TARGET_W_S, target_slack=TARGET_SLACK,
            y_max=False if args.full_range else None,
        )
    print(f"\nSaved: {pdf}\n       {png}")


if __name__ == "__main__":
    main()
