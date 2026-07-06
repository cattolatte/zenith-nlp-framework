"""Decoder-only transformer language model, implemented from scratch.

Design Principles
-----------------
Zenith's core: a *causal* transformer that predicts the next token. The
architecture is configurable between two well-known recipes, both hand-written on
tensor primitives:

- **Llama-style (default):** RMSNorm, rotary position embeddings (RoPE), and a
  SwiGLU feed-forward network — the modern decoder recipe.
- **GPT-2-style:** LayerNorm, learned absolute position embeddings, and a GELU MLP.

Everything is pre-norm with weight-tied embeddings and GPT-2-style initialization
(residual projections scaled by ``1/sqrt(2·n_layers)``). Attention supports an
optional :class:`KVCache`; the cached and full-forward paths are numerically
equivalent (RoPE included) — see the cache-equivalence test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["DecoderConfig", "KVCache", "RMSNorm", "CausalSelfAttention", "DecoderBlock", "DecoderLM"]


@dataclass(frozen=True)
class DecoderConfig:
    """Shape of a :class:`DecoderLM`.

    Parameters
    ----------
    vocab_size, block_size, embed_dim, num_layers, num_heads, ff_dim, dropout
        The usual transformer dimensions.
    norm : {"rmsnorm", "layernorm"}
        Normalization layer (default RMSNorm, Llama-style).
    positional : {"rope", "learned"}
        Position encoding: rotary (RoPE) or learned absolute embeddings.
    ffn : {"swiglu", "gelu"}
        Feed-forward network: SwiGLU (Llama-style) or a GELU MLP (GPT-2-style).
    """

    vocab_size: int
    block_size: int = 256
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    norm: str = "rmsnorm"
    positional: str = "rope"
    ffn: str = "swiglu"


class KVCache:
    """Per-layer key/value cache for incremental autoregressive decoding."""

    def __init__(self, num_layers: int) -> None:
        self._k: list[torch.Tensor | None] = [None] * num_layers
        self._v: list[torch.Tensor | None] = [None] * num_layers
        self.length = 0

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        past_k, past_v = self._k[layer_idx], self._v[layer_idx]
        if past_k is None:
            self._k[layer_idx], self._v[layer_idx] = k, v
        else:
            self._k[layer_idx] = torch.cat([past_k, k], dim=2)
            self._v[layer_idx] = torch.cat([past_v, v], dim=2)
        return self._k[layer_idx], self._v[layer_idx]


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension, from scratch."""

    def __init__(self, dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(variance + self.eps) + self.bias


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Llama-style): no mean subtraction, no bias."""

    def __init__(self, dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


def _make_norm(kind: str, dim: int) -> nn.Module:
    return RMSNorm(dim) if kind == "rmsnorm" else LayerNorm(dim)


def _rope_tables(head_dim: int, max_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) of shape ``(max_len, head_dim)`` for RoPE."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_len).float()
    freqs = torch.outer(positions, inv_freq)  # (max_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch, heads, seq, head_dim); cos/sin: (seq, head_dim)
    return x * cos + _rotate_half(x) * sin


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))`` (Llama-style)."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


def _gelu_mlp(dim: int, hidden: int, dropout: float) -> nn.Sequential:
    """GPT-2-style feed-forward (kept as a Sequential so its keys match earlier
    checkpoints): Linear → GELU → Linear → Dropout."""
    return nn.Sequential(
        nn.Linear(dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, dim),
        nn.Dropout(dropout),
    )


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, optionally with rotary embeddings."""

    mask: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(
        self, *, embed_dim: int, num_heads: int, block_size: int, dropout: float = 0.0, rope: bool = True
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rope = rope

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        causal = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", causal)
        if rope:
            cos, sin = _rope_tables(self.head_dim, block_size)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)

    def forward(
        self, x: torch.Tensor, *, cache: KVCache | None = None, layer_idx: int = 0, offset: int = 0
    ) -> torch.Tensor:
        batch, seq, embed = x.shape
        q, k, v = self.qkv(x).split(embed, dim=2)
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope:
            cos = self.rope_cos[offset : offset + seq]
            sin = self.rope_sin[offset : offset + seq]
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        total = k.size(2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if cache is None:
            scores = scores.masked_fill(self.mask[:, :, :seq, :seq] == 0, float("-inf"))
        else:
            q_pos = torch.arange(offset, offset + seq, device=x.device).unsqueeze(1)
            k_pos = torch.arange(total, device=x.device).unsqueeze(0)
            scores = scores.masked_fill((k_pos > q_pos).view(1, 1, seq, total), float("-inf"))

        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(batch, seq, embed)
        return self.proj_dropout(self.proj(out))


class DecoderBlock(nn.Module):
    """A single pre-norm decoder block."""

    def __init__(self, *, config: DecoderConfig) -> None:
        super().__init__()
        self.norm1 = _make_norm(config.norm, config.embed_dim)
        self.attention = CausalSelfAttention(
            embed_dim=config.embed_dim, num_heads=config.num_heads,
            block_size=config.block_size, dropout=config.dropout, rope=config.positional == "rope",
        )
        self.norm2 = _make_norm(config.norm, config.embed_dim)
        if config.ffn == "swiglu":
            self.feed_forward: nn.Module = SwiGLU(config.embed_dim, config.ff_dim, config.dropout)
        else:
            self.feed_forward = _gelu_mlp(config.embed_dim, config.ff_dim, config.dropout)

    def forward(
        self, x: torch.Tensor, *, cache: KVCache | None = None, layer_idx: int = 0, offset: int = 0
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), cache=cache, layer_idx=layer_idx, offset=offset)
        x = x + self.feed_forward(self.norm2(x))
        return x


class DecoderLM(nn.Module):
    """Decoder-only transformer language model.

    Examples
    --------
    >>> import torch
    >>> model = DecoderLM(DecoderConfig(vocab_size=259, block_size=16, embed_dim=32,
    ...                                 num_layers=2, num_heads=2, ff_dim=64))
    >>> model(torch.zeros(1, 8, dtype=torch.long)).shape
    torch.Size([1, 8, 259])
    """

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.learned_positions = config.positional == "learned"
        if self.learned_positions:
            self.position_embedding = nn.Embedding(config.block_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(DecoderBlock(config=config) for _ in range(config.num_layers))
        self.norm = _make_norm(config.norm, config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying
        self._init_weights()

    def _init_weights(self) -> None:
        """GPT-2-style init: N(0, 0.02); residual output projections scaled down."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        residual_std = 0.02 / math.sqrt(2 * self.config.num_layers)
        for name, param in self.named_parameters():
            if name.endswith(("attention.proj.weight", "feed_forward.down.weight", "feed_forward.2.weight")):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    def forward(self, input_ids: torch.Tensor, *, cache: KVCache | None = None) -> torch.Tensor:
        """Return next-token logits of shape ``(batch, seq, vocab_size)``."""
        _, seq = input_ids.shape
        offset = cache.length if cache is not None else 0
        if offset + seq > self.config.block_size:
            raise ValueError(
                f"sequence position {offset + seq} exceeds block_size {self.config.block_size}"
            )
        x = self.token_embedding(input_ids)
        if self.learned_positions:
            positions = torch.arange(offset, offset + seq, device=input_ids.device)
            x = x + self.position_embedding(positions)
        x = self.dropout(x)
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, cache=cache, layer_idx=layer_idx, offset=offset)
        if cache is not None:
            cache.length += seq
        return self.lm_head(self.norm(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
