"""Data preparation for the wikitext_span_transformer experiment.

For a given tokenizer, trains it once on the WikiText-103 train split (or
loads a pretrained one), encodes train/validation/test separately with that
one fixed tokenizer, and persists everything to disk so the expensive
tokenizer-training + full-corpus-encoding step never has to repeat across the
two experiments (fixed-w and fixed-w_s) that reuse the same tokenizer.

Also provides `solve_w_for_target_span`, the inverse of the span-CDF lookup
already used by `wikitext_tokenizers`: instead of "what span w_s does a
w-token window cover", it answers "how many tokens w does this tokenizer need
to reliably (at some confidence/slack level) cover a target span w_s".
"""

from __future__ import annotations

import json
import os

import numpy as np

from experiments.wikitext_tokenizers.build_tokenizers import (
    CharTokenizer,
    HFTokenizer,
    SPTokenizer,
    TransformersTokenizer,
)

MANIFEST_JSON = "manifest.json"
SPLITS = ("train", "validation", "test")

# Cap on the number of tokens used to *solve* for w (binary search on the
# span CDF) -- large enough for the CDF to have converged, far cheaper than
# scanning the full multi-hundred-million-token train split every step of
# the search. The full array is still used for actual training.
_SOLVE_MAX_TOKENS = 20_000_000


# ---------------------------------------------------------------------------
# Tokenizer build/load with on-disk caching (extends build_tokenizers.py's
# factory with persistence for BPE/WordPiece, which it doesn't save today).
# ---------------------------------------------------------------------------

def get_or_build_tokenizer(tok_cfg, train_text, artifacts_dir):
    """Load a cached tokenizer artifact if present, else train and save one.

    Mirrors `wikitext_tokenizers.build_tokenizers.build_tokenizer`'s dispatch,
    but adds a persistence layer for BPE/WordPiece (SentencePiece already
    saves its model file; character and pretrained tokenizers are cheap /
    inherently reproducible and don't need a cache file).
    """
    name = tok_cfg["name"]
    typ = tok_cfg["type"]
    os.makedirs(artifacts_dir, exist_ok=True)

    if typ == "character":
        return CharTokenizer.train(train_text)

    if typ == "pretrained":
        return TransformersTokenizer.from_pretrained(name, tok_cfg["model"])

    if typ == "bpe" or typ == "wordpiece":
        from tokenizers import Tokenizer  # noqa: PLC0415

        cache_path = os.path.join(artifacts_dir, f"{name}.tokenizer.json")
        if os.path.exists(cache_path):
            print(f"  [{name}] loading cached tokenizer from {cache_path}", flush=True)
            return HFTokenizer(name, Tokenizer.from_file(cache_path))

        V = int(tok_cfg["vocab_size"])
        lines = [l for l in train_text.split("\n") if l.strip()]
        if typ == "bpe":
            print(f"  [{name}] training BPE tokenizer (vocab_size={V}) ...", flush=True)
            tok = HFTokenizer.train_bpe(name, lines, V)
        else:
            print(f"  [{name}] training WordPiece tokenizer (vocab_size={V}) ...", flush=True)
            tok = HFTokenizer.train_wordpiece(name, lines, V)
        tok._tok.save(cache_path)
        print(f"  [{name}] saved tokenizer to {cache_path}", flush=True)
        return tok

    if typ == "sentencepiece":
        V = int(tok_cfg["vocab_size"])
        model_path = os.path.join(artifacts_dir, f"{name}.model")
        if os.path.exists(model_path):
            print(f"  [{name}] loading cached tokenizer from {model_path}", flush=True)
            import sentencepiece as spm  # noqa: PLC0415

            sp = spm.SentencePieceProcessor()
            sp.Load(model_path)
            return SPTokenizer(name, sp, model_path)
        print(f"  [{name}] training SentencePiece tokenizer (vocab_size={V}) ...", flush=True)
        return SPTokenizer.train(name, train_text, V, artifacts_dir)

    raise ValueError(f"Unknown tokenizer type: {typ!r}")


# ---------------------------------------------------------------------------
# Encode + persist all three splits for one tokenizer.
# ---------------------------------------------------------------------------

def prepare_tokenizer_data(tok_cfg, splits_text, data_dir, artifacts_dir):
    """Build/load the tokenizer, encode train/validation/test, save to disk.

    Parameters
    ----------
    tok_cfg : dict
        One entry from the config's `tokenizers` list.
    splits_text : dict[str, str]
        {"train": ..., "validation": ..., "test": ...} raw text.
    data_dir : str
        Directory to write this tokenizer's `ids_<split>.npy`,
        `char_lens_<split>.npy`, and `manifest.json`.
    artifacts_dir : str
        Directory for tokenizer artifacts (shared across tokenizers).

    Returns
    -------
    dict : the manifest (also written to `data_dir/manifest.json`).
    """
    name = tok_cfg["name"]
    os.makedirs(data_dir, exist_ok=True)

    tok = get_or_build_tokenizer(tok_cfg, splits_text["train"], artifacts_dir)

    manifest = {"name": name, "type": tok_cfg["type"], "vocab_size": int(tok.vocab_size)}
    for split in SPLITS:
        text = splits_text[split]
        print(f"  [{name}] encoding {split} ({len(text):,} chars) ...", flush=True)
        ids, char_lens = tok.encode_with_char_lengths(text)
        np.save(os.path.join(data_dir, f"ids_{split}.npy"), ids.astype(np.int32))
        np.save(os.path.join(data_dir, f"char_lens_{split}.npy"), char_lens.astype(np.int32))
        manifest[f"num_chars_{split}"] = int(len(text))
        manifest[f"num_tokens_{split}"] = int(len(ids))
        if split == "train":
            manifest["ratio"] = float(len(text)) / float(len(ids))

    manifest_path = os.path.join(data_dir, MANIFEST_JSON)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [{name}] manifest saved to {manifest_path}", flush=True)
    return manifest


def load_prepared_tokenizer_data(data_dir):
    """Load a manifest + all three splits' ids/char_lens arrays from disk."""
    manifest_path = os.path.join(data_dir, MANIFEST_JSON)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"No prepared data at {data_dir}; run the 'prepare' stage first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    arrays = {}
    for split in SPLITS:
        arrays[f"ids_{split}"] = np.load(os.path.join(data_dir, f"ids_{split}.npy"))
        arrays[f"char_lens_{split}"] = np.load(os.path.join(data_dir, f"char_lens_{split}.npy"))
    return manifest, arrays


# ---------------------------------------------------------------------------
# Solve w (tokens) such that a window of w tokens covers a target span w_s
# at a given confidence/slack level -- the inverse of the span-CDF lookup in
# wikitext_tokenizers.analysis.compute_span_distribution.
# ---------------------------------------------------------------------------

def _p_span_le(char_lens, w, w_s_target):
    """P(span of w consecutive tokens <= w_s_target), via sliding-window sum."""
    n = len(char_lens)
    if n <= w:
        raise ValueError(f"Sequence length {n} <= window w={w}")
    cs = np.empty(n + 1, dtype=np.int64)
    cs[0] = 0
    np.cumsum(char_lens, dtype=np.int64, out=cs[1:])
    spans = cs[w:] - cs[:-w]
    return float(np.mean(spans <= w_s_target))


def solve_w_for_target_span(char_lens, vocab_size, ratio, w_s_target, slack):
    """Find the largest token-window w such that scaled P(span <= w_s_target)
    is still >= `slack` -- i.e. the token budget that reaches the target
    character span w_s_target at the given confidence level, matching the
    scaled-CDF convention used by `plot_markov_tokenizers.py`
    (scale = log2(vocab_size) / ratio).

    Deterministic (ratio == 1.0, e.g. the character tokenizer) is a
    degenerate step function -- returns `w_s_target` directly.
    """
    if ratio == 1.0:
        return int(w_s_target)

    if len(char_lens) > _SOLVE_MAX_TOKENS:
        char_lens = char_lens[:_SOLVE_MAX_TOKENS]

    scale = float(np.log2(vocab_size) / ratio)

    def scaled(w):
        return _p_span_le(char_lens, w, w_s_target) * scale

    lo, hi = 1, max(2, int(w_s_target))
    # scaled(w) decreases monotonically as w grows (bigger context -> span
    # more surely exceeds the fixed target). Grow hi until it drops below
    # slack, then bisect.
    while scaled(hi) > slack:
        hi *= 2
        if hi > len(char_lens) - 1:
            hi = len(char_lens) - 1
            break
    if scaled(hi) > slack:
        return int(hi)  # never drops below slack within available data

    while lo < hi:
        mid = (lo + hi + 1) // 2
        if scaled(mid) > slack:
            lo = mid
        else:
            hi = mid - 1
    return int(lo)
