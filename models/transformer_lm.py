"""Vanilla causal transformer with sinusoidal PE, pre-LN, weight tying."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_pe(max_seq_len, d_embd):
    pe = torch.zeros(max_seq_len, d_embd)
    pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_embd, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_embd)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class CausalSelfAttention(nn.Module):
    def __init__(self, d_embd, n_heads):
        super().__init__()
        assert d_embd % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_embd // n_heads
        self.qkv = nn.Linear(d_embd, 3 * d_embd, bias=False)
        self.out = nn.Linear(d_embd, d_embd, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, d_embd, n_heads, d_ffn):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_embd)
        self.attn = CausalSelfAttention(d_embd, n_heads)
        self.ln2 = nn.LayerNorm(d_embd)
        self.ffn = nn.Sequential(
            nn.Linear(d_embd, d_ffn, bias=False),
            nn.GELU(),
            nn.Linear(d_ffn, d_embd, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Autoregressive transformer with sinusoidal PE, untied input/output."""

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        n_layers=4,
        d_embd=256,
        d_ffn=1024,
        n_heads=4,
        init_std=0.02,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.init_std = init_std

        self.embed = nn.Embedding(vocab_size, d_embd)
        self.register_buffer(
            "pos", _sinusoidal_pe(max_seq_len, d_embd), persistent=False
        )
        self.blocks = nn.ModuleList(
            [Block(d_embd, n_heads, d_ffn) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_embd)
        self.lm_head = nn.Linear(d_embd, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=self.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=self.init_std)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        T = x.shape[1]
        h = self.embed(x) + self.pos[:T]
        for b in self.blocks:
            h = b(h)
        h = self.ln_f(h)
        return self.lm_head(h)
