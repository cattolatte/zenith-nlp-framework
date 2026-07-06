"""Decoder-only transformer language model, implemented from scratch.

Design Principles
-----------------
This is Zenith's core and its reason to exist: a *causal* transformer that
predicts the next token, the counterpart to an encoder that classifies. Everything
that makes it a decoder rather than an encoder lives here — the causal mask that
forbids a position from attending to the future, and the language-model head that
scores the whole vocabulary at every position.

Built on tensor primitives (own attention, own layer norm, own blocks); PyTorch
supplies autograd, ``nn.Embedding``/``nn.Linear`` containers and optimizers, not
the architecture. Pre-norm blocks, GELU feed-forward, and weight-tied embeddings —
the small, well-understood decoder recipe, written to be read.

Attention supports an optional :class:`KVCache` for efficient autoregressive
decoding: without a cache (training / a full forward) it uses a precomputed
triangular mask; with a cache it appends this step's keys/values and applies a
positional mask derived from the running offset. The two paths are numerically
equivalent — see the cache-equivalence test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["DecoderConfig", "KVCache", "CausalSelfAttention", "DecoderBlock", "DecoderLM"]


@dataclass(frozen=True)
class DecoderConfig:
    """Shape of a :class:`DecoderLM`.

    Parameters
    ----------
    vocab_size : int
        Number of distinct token ids the model scores.
    block_size : int
        Maximum context length (positions) the model can attend over.
    embed_dim : int
        Model / embedding dimension.
    num_layers : int
        Number of stacked decoder blocks.
    num_heads : int
        Number of attention heads (must divide ``embed_dim``).
    ff_dim : int
        Hidden size of each block's feed-forward network.
    dropout : float
        Dropout probability used throughout.
    """

    vocab_size: int
    block_size: int = 256
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1


class KVCache:
    """Per-layer key/value cache for incremental autoregressive decoding.

    Holds the running keys and values for every layer so that generating the next
    token only requires a forward pass over the *new* token(s) rather than the
    whole sequence. ``length`` tracks how many positions are cached (updated once
    per model forward, by :class:`DecoderLM`).
    """

    def __init__(self, num_layers: int) -> None:
        self._k: list[torch.Tensor | None] = [None] * num_layers
        self._v: list[torch.Tensor | None] = [None] * num_layers
        self.length = 0

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this step's ``k``/``v`` for ``layer_idx`` and return the full pair."""
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


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    Each position may attend only to itself and earlier positions — the single
    property that makes this a *decoder*. Query/key/value projections are computed
    together; without a cache the mask is a cached lower-triangular buffer, with a
    cache the mask is derived from the running ``offset``.

    Parameters
    ----------
    embed_dim : int
        Model dimension.
    num_heads : int
        Number of heads; must divide ``embed_dim``.
    block_size : int
        Maximum sequence length (size of the cached causal mask).
    dropout : float, default 0.0
        Dropout on attention weights and the output projection.
    """

    mask: torch.Tensor

    def __init__(self, *, embed_dim: int, num_heads: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        causal = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", causal)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cache: KVCache | None = None,
        layer_idx: int = 0,
        offset: int = 0,
    ) -> torch.Tensor:
        """Attend over ``x`` of shape ``(batch, seq, embed)``.

        With ``cache`` set, ``x`` is the new token(s) only; keys/values are read
        from and written to ``cache[layer_idx]`` and the mask uses ``offset`` (the
        number of already-cached positions).
        """
        batch, seq, embed = x.shape
        q, k, v = self.qkv(x).split(embed, dim=2)
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        total = k.size(2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if cache is None:
            scores = scores.masked_fill(self.mask[:, :, :seq, :seq] == 0, float("-inf"))
        else:
            # Query i sits at absolute position offset+i and may see keys 0..offset+i.
            q_pos = torch.arange(offset, offset + seq, device=x.device).unsqueeze(1)
            k_pos = torch.arange(total, device=x.device).unsqueeze(0)
            disallowed = (k_pos > q_pos).view(1, 1, seq, total)
            scores = scores.masked_fill(disallowed, float("-inf"))

        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq, embed)
        return self.proj_dropout(self.proj(out))


class DecoderBlock(nn.Module):
    """A single pre-norm decoder block: causal attention + feed-forward."""

    def __init__(
        self, *, embed_dim: int, num_heads: int, ff_dim: int, block_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.attention = CausalSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, block_size=block_size, dropout=dropout
        )
        self.norm2 = LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        cache: KVCache | None = None,
        layer_idx: int = 0,
        offset: int = 0,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), cache=cache, layer_idx=layer_idx, offset=offset)
        x = x + self.feed_forward(self.norm2(x))
        return x


class DecoderLM(nn.Module):
    """Decoder-only transformer language model.

    Token embeddings + learned positional embeddings → stacked pre-norm decoder
    blocks → final norm → tied language-model head producing next-token logits.

    Parameters
    ----------
    config : DecoderConfig
        The model's shape.

    Examples
    --------
    >>> import torch
    >>> model = DecoderLM(DecoderConfig(vocab_size=259, block_size=16, embed_dim=32,
    ...                                 num_layers=2, num_heads=2, ff_dim=64))
    >>> logits = model(torch.zeros(1, 8, dtype=torch.long))
    >>> logits.shape
    torch.Size([1, 8, 259])
    """

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                ff_dim=config.ff_dim,
                block_size=config.block_size,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        )
        self.norm = LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        # Weight tying: the input embedding and output projection share weights.
        self.lm_head.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self) -> None:
        """GPT-2-style init: N(0, 0.02) weights, and residual output projections
        scaled by 1/sqrt(2 * num_layers) so the residual stream stays well-behaved
        with depth. This measurably improves convergence over default init."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        residual_std = 0.02 / math.sqrt(2 * self.config.num_layers)
        for name, param in self.named_parameters():
            if name.endswith("attention.proj.weight") or name.endswith("feed_forward.2.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    def forward(self, input_ids: torch.Tensor, *, cache: KVCache | None = None) -> torch.Tensor:
        """Return next-token logits of shape ``(batch, seq, vocab_size)``.

        With ``cache``, ``input_ids`` holds only the new token(s); positions are
        offset by the number of already-cached tokens.
        """
        _, seq = input_ids.shape
        offset = cache.length if cache is not None else 0
        if offset + seq > self.config.block_size:
            raise ValueError(
                f"sequence position {offset + seq} exceeds block_size {self.config.block_size}"
            )
        positions = torch.arange(offset, offset + seq, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, cache=cache, layer_idx=layer_idx, offset=offset)
        if cache is not None:
            cache.length += seq
        return self.lm_head(self.norm(x))

    def num_parameters(self) -> int:
        """Total parameter count (embedding counted once, since the head is tied)."""
        return sum(p.numel() for p in self.parameters())
