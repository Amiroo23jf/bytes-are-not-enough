"""Training loop for the text_transformer experiment.

Stride-1 contiguous-window LM training over B parallel streams.
Supports bytes mode (vocab=256) and ByteBPE token mode (vocab=V).

Conventions:
    - Losses are in bits (nats / log 2).
    - ``last_pos_loss_bits[s]``: last-position CE averaged over B streams at
      step s.  For bytes mode this equals BPB directly.  For tokens mode,
      divide by ``mean_token_bytes`` to obtain BPB.
    - ``bytes_processed[s]``: cumulative source bytes consumed up to step s.
      bytes mode: B bytes/step (each token is one byte).
      tokens mode: sum of byte-lengths of the B last-position target tokens
      per step, accumulated.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from models.transformer_lm import TransformerLM

LOG2 = math.log(2.0)


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


def _warmup_lr(step: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr


def _set_lr(optim, lr: float) -> None:
    for g in optim.param_groups:
        g["lr"] = lr


def _build_model(vocab_size: int, seq_len: int, model_cfg: dict, device):
    return TransformerLM(
        vocab_size=int(vocab_size),
        max_seq_len=int(seq_len + 1),
        n_layers=model_cfg["n_layers"],
        d_embd=model_cfg["d_embd"],
        d_ffn=model_cfg["d_ffn"],
        n_heads=model_cfg["n_heads"],
        init_std=model_cfg["init_std"],
    ).to(device)


def _build_optim(model, train_cfg: dict):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=tuple(train_cfg["betas"]),
    )


def train_model(
    data: np.ndarray,
    vocab_size: int,
    seq_len: int,
    token_byte_lengths: np.ndarray,
    cfg: dict,
    seed: int,
    device,
    progress_desc: str = "LM",
) -> dict:
    """Train a TransformerLM on ``data`` and return per-step loss arrays.

    The corpus is split into ``batch_size`` non-overlapping streams processed
    in parallel.  At each step *s* the model sees context s..s+seq_len-1 and
    predicts position s+seq_len for all B streams simultaneously.

    Parameters
    ----------
    data:
        1-D token/byte id array (uint8 or int32/int64), length N.
    vocab_size:
        Vocabulary size for the model.
    seq_len:
        Context window length W.
    token_byte_lengths:
        ``(vocab_size,)`` integer array: source bytes per token.
        All 1s for bytes mode; BPE byte-lengths for tokens mode.
    cfg:
        Full YAML config with ``train`` and ``model`` sub-dicts.
    seed:
        Model initialisation seed.
    device:
        PyTorch device.
    progress_desc:
        tqdm bar label.

    Returns
    -------
    dict with
        last_pos_loss_bits   : ``(S,)`` float32 -- CE in bits per step.
        cumulative_loss_bits : ``(S,)`` float64 -- running sum of above.
        bytes_processed      : ``(S,)`` int64   -- cumulative source bytes.
        bytes_per_step       : ``(S,)`` int64   -- source bytes per step.
        S                    : int.
        num_parameters       : int.
    """
    torch.manual_seed(seed * 31337 + 1)
    t_cfg = cfg["train"]
    B = int(t_cfg["batch_size"])
    grad_clip = float(t_cfg.get("grad_clip", 1.0))

    N = len(data)
    iter_num = N // B                                      # stream length
    ind = np.arange(B, dtype=np.int64) * iter_num         # stream start offsets
    S = iter_num - seq_len                                 # gradient steps
    if S <= 0:
        raise ValueError(
            f"Corpus too short: batch_size={B}, seq_len={seq_len}, "
            f"stream_len={iter_num}, steps={S}"
        )

    # Build (B, iter_num) streams tensor — equivalent to applying strided_app
    # with L=seq_len+1, S=1 to each of the B contiguous stream slices.
    row_idx = ind[:, None] + np.arange(iter_num, dtype=np.int64)[None, :]
    streams = torch.as_tensor(
        data[row_idx].astype(np.int64), dtype=torch.long, device=device
    )

    model = _build_model(vocab_size, seq_len, cfg["model"], device)
    optim = _build_optim(model, t_cfg)

    byte_len_t = torch.as_tensor(
        np.asarray(token_byte_lengths, dtype=np.int64), dtype=torch.long, device=device
    )

    last_loss_arr = torch.empty(S, device=device)
    bytes_per_step_arr = torch.empty(S, device=device, dtype=torch.int64)
    model.train()

    pbar = tqdm(
        range(S),
        desc=progress_desc,
        dynamic_ncols=True,
        mininterval=0.5,
        miniters=10,
        leave=False,
    )
    for s in pbar:
        _set_lr(optim, _warmup_lr(s, int(t_cfg["warmup_steps"]), float(t_cfg["lr"])))

        x = streams[:, s : s + seq_len]
        y = streams[:, s + 1 : s + seq_len + 1]

        logits = model(x)
        loss = (
            F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), reduction="mean")
            / LOG2
        )

        with torch.no_grad():
            last_targets = y[:, -1]
            last_loss = (
                F.cross_entropy(logits[:, -1, :], last_targets, reduction="mean") / LOG2
            )
            last_loss_arr[s] = last_loss
            bytes_per_step_arr[s] = byte_len_t[last_targets].sum()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        pbar.set_postfix(
            loss=f"{last_loss.item():.4f}",
            refresh=((s + 1) % 10 == 0 or s == S - 1),
        )
    pbar.close()

    per_step = last_loss_arr.detach().cpu().numpy().astype(np.float32)
    bps = bytes_per_step_arr.detach().cpu().numpy().astype(np.int64)

    return {
        "last_pos_loss_bits": per_step,
        "cumulative_loss_bits": np.cumsum(per_step.astype(np.float64)),
        "bytes_processed": np.cumsum(bps).astype(np.int64),
        "bytes_per_step": bps,
        "S": int(S),
        "num_parameters": int(model.num_parameters()),
    }
