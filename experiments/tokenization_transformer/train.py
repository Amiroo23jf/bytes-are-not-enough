"""Training loop for the Markov tokenization-transformer experiment.

Trains a single TransformerLM on a (possibly BPE-tokenized) sequence for a
fixed number of iterations. Each iteration samples batch_size windows of
length W+1 uniformly at random from the sequence. The logged loss is the
last-token cross-entropy normalized by the compression ratio to give bits
per source symbol (bits per original binary symbol).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from models.transformer_lm import TransformerLM

LOG2 = math.log(2.0)


def train_model(Z, ratio, token_bits, V, W, cfg, seed, device, progress_desc=""):
    """Train one TransformerLM on token sequence Z.

    Parameters
    ----------
    Z : np.ndarray (n,) int64
        Token sequence. Raw binary if V==2, BPE-encoded otherwise.
    ratio : float
        len(Y) / len(Z). Dividing the per-token CE loss by this converts it
        to bits per original source symbol (bits per bit).
    token_bits : np.ndarray (V,) int64
        Bit-length of each token id; used to count source bits in each step.
    V : int
        Vocabulary size (== model vocab_size).
    W : int
        Context window length. Model sees W tokens and predicts the next one.
    cfg : dict
        Full experiment config dict.
    seed : int
        Seeds both model initialization and the window-sampling RNG so that
        results are fully reproducible.
    device : torch.device

    Returns
    -------
    dict with:
        loss_per_iter      (iterations,) float32 -- last-token CE averaged
                           over the batch, normalized to bits per source
                           symbol, one entry per gradient step.
        bits_seen_per_iter (iterations,) int64   -- total source bits in the
                           W input tokens across the batch per step
                           (= sum of token_bits[x] over all B*W input tokens).
        num_parameters     int
    """
    torch.manual_seed(seed)

    m_cfg = cfg["model"]
    t_cfg = cfg["train"]
    iterations = int(t_cfg["iterations"])
    B = int(t_cfg["batch_size"])
    warmup = int(t_cfg.get("warmup_steps", 0))
    grad_clip = float(t_cfg.get("grad_clip", 1.0))
    base_lr = float(t_cfg["lr"])

    model = TransformerLM(
        vocab_size=V,
        max_seq_len=W + 1,
        n_layers=m_cfg["n_layers"],
        d_embd=m_cfg["d_embd"],
        d_ffn=m_cfg["d_ffn"],
        n_heads=m_cfg["n_heads"],
        init_std=m_cfg["init_std"],
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(t_cfg["weight_decay"]),
        betas=tuple(t_cfg["betas"]),
    )

    Z_t = torch.as_tensor(Z, dtype=torch.long, device=device)
    token_bits_t = torch.as_tensor(token_bits, dtype=torch.long, device=device)
    n = len(Z)

    loss_arr = np.empty(iterations, dtype=np.float32)
    bits_arr = np.empty(iterations, dtype=np.int64)

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
        lr = base_lr * (s + 1) / warmup if s < warmup else base_lr
        for g in optim.param_groups:
            g["lr"] = lr

        # Sample B windows of length W+1; valid starts are [0, n-W-1].
        starts = torch.randint(0, n - W, (B,), device=device)
        offs = torch.arange(W + 1, device=device)
        data = Z_t[starts.unsqueeze(1) + offs.unsqueeze(0)]  # (B, W+1)
        x = data[:, :W]   # (B, W) -- input tokens
        y = data[:, 1:]   # (B, W) -- target tokens

        logits = model(x)

        # Full-window CE for the gradient step.
        loss = F.cross_entropy(
            logits.reshape(-1, V), y.reshape(-1), reduction="mean"
        ) / LOG2

        with torch.no_grad():
            # Last-token CE, normalized to bits per source symbol.
            last_loss = F.cross_entropy(
                logits[:, -1, :], y[:, -1], reduction="mean"
            ) / LOG2
            loss_arr[s] = float(last_loss) / ratio

            # Source bits seen: sum L(z) over all B*W input tokens.
            bits_arr[s] = int(token_bits_t[x].sum())

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optim.step()

        pbar.set_postfix(
            loss=f"{loss_arr[s]:.4f}",
            refresh=((s + 1) % 10 == 0 or s == iterations - 1),
        )

    pbar.close()

    return {
        "loss_per_iter": loss_arr,
        "bits_seen_per_iter": bits_arr,
        "num_parameters": int(model.num_parameters()),
    }
