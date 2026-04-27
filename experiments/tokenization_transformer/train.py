"""Training loop for the Theorem 3 tokenization-transformer experiment.

Stride-1 contiguous-window training over B parallel streams. The Y-LM
operates on raw bits (V_y = 2) and the Z-LM on BPE token ids (V), but
both share the same context length W and the same training procedure
modulo `vocab_size` and `input_len`.

Conventions:
    - All logged losses are in bits (log base 2).
    - Y-LM input length = W bits; Z-LM input length = W tokens.
    - `last_pos_loss_bits[s]` is the cross-entropy of the model's last-token
      prediction at iteration `s`, averaged over the B streams. For the Y-LM
      this is already bits-per-bit. For the Z-LM it is bits-per-token, and
      `train_Z_model` additionally returns `last_pos_loss_bits_per_bit =
      last_pos_loss_bits / mean_token_bits` so the two curves can share the
      bits-per-bit y-axis.
    - For the y-axis (Y-symbols processed) we track:
        Y-LM: every gradient step processes B last-position bits, so
              cumulative `bits_processed = (s + 1) * B`.
        Z-LM: every gradient step predicts B tokens, each of variable
              bit-length; we sum `token_bits[predicted_target_token_ids]`
              across the B targets per step. The cumulative sum is the
              x-axis.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from models.transformer_lm import TransformerLM

LOG2 = math.log(2.0)


def _warmup_lr(step, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr


def _set_lr(optim, lr):
    for g in optim.param_groups:
        g["lr"] = lr


def _build_model(vocab_size, input_len, model_cfg, device):
    return TransformerLM(
        vocab_size=int(vocab_size),
        max_seq_len=int(input_len + 1),
        n_layers=model_cfg["n_layers"],
        d_embd=model_cfg["d_embd"],
        d_ffn=model_cfg["d_ffn"],
        n_heads=model_cfg["n_heads"],
        init_std=model_cfg["init_std"],
    ).to(device)


def _build_optim(model, train_cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg["betas"]),
    )


def train_Y_model(
    streams,
    W,
    cfg,
    seed,
    device,
    progress_desc="Y-model",
):
    """Train the bit-level Y-model.

    Returns dict with:
        last_pos_loss_bits    : (S_Y,) float32 -- bits-per-bit per step.
        bits_processed        : (S_Y,) int64   -- cumulative bits at each
                                                  step's end (= (s+1) * B).
        S                     : int.
        num_parameters        : int.
    """
    torch.manual_seed(seed * 31337 + 1)
    vocab_size = 2
    input_len = int(W)
    model = _build_model(vocab_size, input_len, cfg["model"], device)
    t_cfg = cfg["train"]
    optim = _build_optim(model, t_cfg)
    grad_clip = float(t_cfg.get("grad_clip", 1.0))

    B, stream_len = streams.shape
    S = stream_len - input_len
    assert S > 0, f"stream too short: stream_len={stream_len}, W={input_len}"

    last_loss_bits = torch.empty(S, device=device)
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
        lr = _warmup_lr(s, t_cfg["warmup_steps"], t_cfg["lr"])
        _set_lr(optim, lr)

        x = streams[:, s : s + input_len]
        y = streams[:, s + 1 : s + input_len + 1]

        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1), reduction="mean"
        ) / LOG2

        with torch.no_grad():
            last_logits = logits[:, -1, :]
            last_targets = y[:, -1]
            last_loss = F.cross_entropy(
                last_logits, last_targets, reduction="mean"
            ) / LOG2
            last_loss_bits[s] = last_loss

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optim.step()

        pbar.set_postfix(
            loss=f"{last_loss.item():.4f}",
            refresh=((s + 1) % 10 == 0 or s == S - 1),
        )
    pbar.close()

    bits_processed = (np.arange(1, S + 1, dtype=np.int64)) * int(B)
    return {
        "last_pos_loss_bits": last_loss_bits.detach().cpu().numpy().astype(np.float32),
        "bits_processed": bits_processed,
        "S": int(S),
        "num_parameters": int(model.num_parameters()),
    }


def train_Z_model(
    streams,
    W,
    V,
    mean_token_bits,
    token_bits,
    cfg,
    seed,
    device,
    progress_desc="Z-model",
):
    """Train the token-level Z-model.

    Returns dict with:
        last_pos_loss_bits_per_token : (S_Z,) float32 -- raw CE in bits.
        last_pos_loss_bits_per_bit   : (S_Z,) float32 -- divided by
                                                         `mean_token_bits` so
                                                         it lives on the Y-LM's
                                                         bits-per-bit y-axis.
        bits_processed_per_step      : (S_Z,) int64   -- per step, sum over
                                                         the B last-position
                                                         predicted target token
                                                         bit-lengths.
        bits_processed               : (S_Z,) int64   -- cumulative sum.
        S                            : int.
        num_parameters               : int.
    """
    if mean_token_bits <= 0:
        raise ValueError(f"mean_token_bits must be > 0, got {mean_token_bits}")
    torch.manual_seed(seed * 31337 + 2)
    vocab_size = int(V)
    input_len = int(W)
    model = _build_model(vocab_size, input_len, cfg["model"], device)
    t_cfg = cfg["train"]
    optim = _build_optim(model, t_cfg)
    grad_clip = float(t_cfg.get("grad_clip", 1.0))

    B, stream_len = streams.shape
    S = stream_len - input_len
    assert S > 0, f"stream too short: stream_len={stream_len}, W={input_len}"

    token_bits_t = torch.as_tensor(
        np.asarray(token_bits, dtype=np.int64), device=device
    )

    last_loss_bits = torch.empty(S, device=device)
    bits_per_step = torch.empty(S, device=device, dtype=torch.int64)
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
        lr = _warmup_lr(s, t_cfg["warmup_steps"], t_cfg["lr"])
        _set_lr(optim, lr)

        x = streams[:, s : s + input_len]
        y = streams[:, s + 1 : s + input_len + 1]

        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1), reduction="mean"
        ) / LOG2

        with torch.no_grad():
            last_logits = logits[:, -1, :]
            last_targets = y[:, -1]
            last_loss = F.cross_entropy(
                last_logits, last_targets, reduction="mean"
            ) / LOG2
            last_loss_bits[s] = last_loss
            bits_per_step[s] = token_bits_t[last_targets].sum()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optim.step()

        pbar.set_postfix(
            loss=f"{last_loss.item():.4f}",
            refresh=((s + 1) % 10 == 0 or s == S - 1),
        )
    pbar.close()

    per_token = last_loss_bits.detach().cpu().numpy().astype(np.float32)
    per_bit = (per_token / float(mean_token_bits)).astype(np.float32)
    bps = bits_per_step.detach().cpu().numpy().astype(np.int64)
    bits_cum = np.cumsum(bps).astype(np.int64)

    return {
        "last_pos_loss_bits_per_token": per_token,
        "last_pos_loss_bits_per_bit": per_bit,
        "bits_processed_per_step": bps,
        "bits_processed": bits_cum,
        "S": int(S),
        "num_parameters": int(model.num_parameters()),
    }
