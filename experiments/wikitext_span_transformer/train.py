"""Training + held-out evaluation for the wikitext_span_transformer experiment.

Unlike `experiments.text_transformer.train` (sequential B-parallel streams,
appropriate for that experiment's byte/BPE-on-enwik8 comparison), training
here draws a **fresh random window start per step**, matching the approach
`experiments.fragmentation_transformer.train` already uses for its
non-stationary-ish sources. This matters specifically for real text: since
`iterations` is a small bounded budget relative to a train split that can be
hundreds of millions of tokens long, sequential streams would only ever
touch a handful of fixed, evenly-spaced local slices of the corpus -- never
sampling its full topical diversity. Random-window sampling avoids
conflating "this tokenizer's loss" with "the topics that happened to fall in
its slice."

Also adds what this experiment needs that neither existing pipeline has:

  - bf16 autocast. At the vocab sizes here (up to ~50k) the output
    projection + softmax dominate memory; fp32 spills into slow, thrashing
    allocations at longer context lengths on memory-constrained GPUs, while
    bf16 comfortably fits and is faster on top.
  - an `epochs` parameter, which (since sampling is with replacement) is
    just a multiplier on total steps: total_steps = iterations * epochs.
  - a separate held-out `evaluate_model` pass, exhaustive (not sampled) over
    every non-overlapping window of the eval split, averaging loss over
    every position (not just the last) -- text_transformer only ever logs
    training-time loss, never a held-out pass.

Both training and eval keep the token-id array on host (numpy) memory and
transfer only the small per-step/per-batch slice to the GPU, rather than
uploading the full tokenized corpus (the `char` tokenizer alone produces
~500M tokens for WikiText-103 -- fine on an A100, but there's no reason to
require it).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from models.transformer_lm import TransformerLM

LOG2 = math.log(2.0)

_AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": None}


def _amp_dtype(name):
    if name not in _AMP_DTYPES:
        raise ValueError(f"Unknown amp_dtype {name!r}; expected one of {list(_AMP_DTYPES)}")
    return _AMP_DTYPES[name]


def _warmup_lr(step, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr


def _set_lr(optim, lr):
    for g in optim.param_groups:
        g["lr"] = lr


def _build_model(vocab_size, seq_len, model_cfg, device):
    return TransformerLM(
        vocab_size=int(vocab_size),
        max_seq_len=int(seq_len + 1),
        n_layers=model_cfg["n_layers"],
        d_embd=model_cfg["d_embd"],
        d_ffn=model_cfg["d_ffn"],
        n_heads=model_cfg["n_heads"],
        init_std=model_cfg["init_std"],
    ).to(device)


def train_model(data, vocab_size, seq_len, cfg, seed, device, progress_desc="LM"):
    """Train a TransformerLM on `data` (1D token-id array).

    Each of the `total_steps = iterations * epochs` steps draws a fresh
    random batch of `batch_size` windows (with replacement) from anywhere in
    `data` -- see module docstring for why this is shuffled rather than
    sequential.

    Returns dict with:
        last_pos_loss_bits : (S,) float32 -- per-step last-position CE, bits.
        S                   : int, total steps.
        num_parameters      : int.
        model               : the trained TransformerLM (for immediate eval).
    """
    torch.manual_seed(seed * 31337 + 1)
    rng = np.random.default_rng(seed * 7919 + 3)
    t_cfg = cfg["train"]
    B = int(t_cfg["batch_size"])
    grad_clip = float(t_cfg.get("grad_clip", 1.0))
    epochs = int(t_cfg.get("epochs", 1))
    total_steps = int(t_cfg["iterations"]) * epochs
    amp_dtype = _amp_dtype(t_cfg.get("amp_dtype", "bfloat16"))
    use_amp = amp_dtype is not None
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]

    data_np = np.asarray(data, dtype=np.int64)
    N = len(data_np)
    if N <= seq_len + 1:
        raise ValueError(f"Corpus too short: len={N}, need > seq_len+1={seq_len + 1}")
    offs = np.arange(seq_len + 1, dtype=np.int64)

    model = _build_model(vocab_size, seq_len, cfg["model"], device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(t_cfg["lr"]),
        weight_decay=float(t_cfg["weight_decay"]),
        betas=tuple(t_cfg["betas"]),
    )

    last_loss_arr = torch.empty(total_steps, device=device)
    model.train()
    pbar = tqdm(
        range(total_steps), desc=progress_desc, dynamic_ncols=True,
        mininterval=0.5, miniters=10, leave=False,
    )

    for step in range(total_steps):
        lr = _warmup_lr(step, int(t_cfg["warmup_steps"]), float(t_cfg["lr"]))
        _set_lr(optim, lr)

        starts = rng.integers(0, N - seq_len - 1, size=B)
        batch_np = data_np[starts[:, None] + offs[None, :]]
        batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
        x = batch[:, :seq_len]
        y = batch[:, 1:]

        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = (
                F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), reduction="mean")
                / LOG2
            )

        with torch.no_grad():
            last_loss = (
                F.cross_entropy(logits[:, -1, :].float(), y[:, -1], reduction="mean") / LOG2
            )
            last_loss_arr[step] = last_loss

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        pbar.set_postfix(
            loss=f"{last_loss.item():.4f}",
            refresh=((step + 1) % 10 == 0 or step == total_steps - 1),
        )
        pbar.update(1)
    pbar.close()

    return {
        "last_pos_loss_bits": last_loss_arr.detach().cpu().numpy().astype(np.float32),
        "S": int(total_steps),
        "num_parameters": int(model.num_parameters()),
        "model": model,
    }


@torch.no_grad()
def evaluate_model(model, data, char_lens, vocab_size, seq_len, device, amp_dtype="bfloat16", batch_size=128):
    """Held-out evaluation over every predicted position in non-overlapping
    (seq_len + 1)-token windows of `data`.

    Unlike the training loop's last-position-only logging (needed there to
    track incremental progress on overlapping stride-1 windows), evaluation
    uses every position of every window since there's no "progress" to plot
    -- just a single held-out number, so extracting a signal from the whole
    window is more sample-efficient.

    Returns (bits_per_token, bits_per_char). `bits_per_char` is exact --
    total cross-entropy bits summed over every scored token, divided by the
    total characters those *same* tokens span (via `char_lens`, aligned
    1:1 with `data`) -- not a whole-split average ratio. The tail remainder
    that doesn't fill a complete window, and each window's non-predicted
    first (context-only) position, are excluded from both sums identically,
    so the ratio reflects exactly the tokens that were actually scored.
    """
    was_training = model.training
    model.eval()
    dtype = _amp_dtype(amp_dtype)
    use_amp = dtype is not None
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]

    stride = seq_len + 1
    data_np = np.asarray(data, dtype=np.int64)
    char_lens_np = np.asarray(char_lens, dtype=np.int64)
    N = len(data_np)
    if len(char_lens_np) != N:
        raise ValueError(f"data/char_lens length mismatch: {N} vs {len(char_lens_np)}")
    n_windows = N // stride
    if n_windows == 0:
        raise ValueError(f"Eval corpus too short: len={N}, need >= {stride}")

    data_np = data_np[: n_windows * stride].reshape(n_windows, stride)
    char_lens_np = char_lens_np[: n_windows * stride].reshape(n_windows, stride)

    total_loss_bits = 0.0
    total_tokens = 0
    total_chars = 0
    for start in range(0, n_windows, batch_size):
        batch = torch.as_tensor(
            data_np[start : start + batch_size], dtype=torch.long, device=device
        )
        x = batch[:, :seq_len]
        y = batch[:, 1 : seq_len + 1]
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=use_amp):
            logits = model(x)
        loss_sum = (
            F.cross_entropy(logits.reshape(-1, vocab_size).float(), y.reshape(-1), reduction="sum")
            / LOG2
        )
        total_loss_bits += float(loss_sum.item())
        total_tokens += y.numel()
        total_chars += int(char_lens_np[start : start + batch_size, 1 : seq_len + 1].sum())

    if was_training:
        model.train()
    return total_loss_bits / total_tokens, total_loss_bits / total_chars
