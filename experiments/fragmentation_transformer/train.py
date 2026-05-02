"""Training loop for the Theorem 1 transformer experiment.

Stride-1 contiguous-window training over B parallel streams. Each iteration
runs one forward/backward on the B-wide batch and logs the last-position CE
(in bits) -- the quantity Theorem 1 bounds.

Conventions:
    - All logged losses are in bits (log base 2).
    - Y-model input length = W, X-model input length = m*W; the last target
      is predicted from all W (resp. m*W) context tokens.
    - loss_X_per_Y_symbol[s] = sum over M consecutive X-iteration last-position
      bit-losses (r = M*s, ..., M*s + M - 1).
"""

import math

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


def _train_model(stream, iterations, input_len, vocab_size, cfg, seed, device, progress_desc=""):
    """Train one autoregressive transformer on the given stream tensor.

    Parameters
    ----------
    stream : LongTensor (stream_len,)  -- single long sequence; batch draws B windows
    input_len : int
        Input context length (W for Y-model, M*W for X-model).
    vocab_size : int
    cfg : dict
    seed : int
        Seeds the model initialization.
    device : torch.device

    Returns
    -------
    dict with:
        last_pos_loss_bits : np.ndarray (S,) -- per-iteration last-position CE
            averaged across the B streams (bits per target token).
    """
    torch.manual_seed(seed)
    if not isinstance(stream, torch.Tensor):
        stream = torch.as_tensor(stream, dtype=torch.long, device=device)
    else:
        stream = stream.to(device)
    max_seq_len = input_len + 1
    B = int(cfg["train"]["batch_size"])
    m_cfg = cfg["model"]
    model = TransformerLM(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        n_layers=m_cfg["n_layers"],
        d_embd=m_cfg["d_embd"],
        d_ffn=m_cfg["d_ffn"],
        n_heads=m_cfg["n_heads"],
        init_std=m_cfg["init_std"],
    ).to(device)

    t_cfg = cfg["train"]
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg["lr"],
        weight_decay=t_cfg["weight_decay"],
        betas=tuple(t_cfg["betas"]),
    )
    grad_clip = float(t_cfg.get("grad_clip", 1.0))

    stream_len = stream.shape[0]
    last_loss_bits = torch.empty(iterations, device=device)
    model.train()
    pbar = tqdm(
        range(iterations),
        desc=progress_desc,
        dynamic_ncols=True,
        mininterval=0.5,
        miniters=10,
        leave=False,
    )
    for s in pbar:
        lr = _warmup_lr(s, t_cfg["warmup_steps"], t_cfg["lr"])
        _set_lr(optim, lr)

        # B random windows from the 1D stream: shape (B, input_len + 1).
        starts = torch.randint(
            0, stream_len - input_len - 1, (B,), device=device, dtype=torch.long
        )
        offs = torch.arange(input_len + 1, device=device, dtype=torch.long)
        data = stream[starts.unsqueeze(1) + offs.unsqueeze(0)]
        x = data[:, :input_len]
        y = data[:, 1:]

        logits = model(x)
        # Full-window CE (bits) for gradient.
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1), reduction="mean"
        ) / LOG2

        # Last-position CE (bits) for logging.
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
            refresh=((s + 1) % 10 == 0 or s == iterations - 1),
        )

    pbar.close()
    return {
        "last_pos_loss_bits": last_loss_bits.detach().cpu().numpy(),
        "iterations": int(iterations),
        "num_parameters": int(model.num_parameters()),
    }


def train_Y_model(streams, cfg, seed, device, progress_desc="Y-model"):
    """Train the Y-model: context W Y-symbols, predict next Y-symbol."""
    W = cfg["W"]
    V_y = streams["V_y"]
    return _train_model(
        stream=streams["Y"],
        iterations=int(cfg["train"]["iterations"]),
        input_len=W,
        vocab_size=V_y,
        cfg=cfg,
        seed=seed * 31337 + 1,
        device=device,
        progress_desc=progress_desc,
    )


def train_X_model(streams, cfg, seed, device, progress_desc="X-model"):
    """Train the X-model: context M*W X-bits, predict next X-bit."""
    M = cfg["M"]
    W = cfg["W"]
    return _train_model(
        stream=streams["X"],
        iterations=int(cfg["train"]["iterations"]),
        input_len=M * W,
        vocab_size=2,
        cfg=cfg,
        seed=seed * 31337 + 2,
        device=device,
        progress_desc=progress_desc,
    )


def x_loss_to_per_Y_symbol(last_pos_loss_bits_x, M, n_Y):
    """Sum M consecutive X-iteration last-position bit-losses into per-Y-symbol.

    Returns an array of length n_Y with
        loss_X_per_Y_symbol[s] = sum_{j=0}^{M-1} last_pos_loss_bits_x[M*s + j]
    using the first M*n_Y entries of `last_pos_loss_bits_x`.
    """
    return last_pos_loss_bits_x[: M * n_Y].reshape(n_Y, M).sum(axis=1)
