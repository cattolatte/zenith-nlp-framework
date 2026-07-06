"""Causal-LM data: turn a stream of token ids into next-token training pairs.

Design Principles
-----------------
Language-model training data is almost trivially simple and should stay that way:
a long 1-D sequence of token ids, sliced into fixed-length blocks, where the
target of each position is the *next* token. No task labels, no padding for the
Phase-1 char/byte setting — the label is the input shifted by one.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

__all__ = ["CausalLMDataset", "encode_corpus", "load_corpus_file", "train_val_split"]


class CausalLMDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Fixed-length ``(input, target)`` blocks over a 1-D id sequence.

    For a block starting at ``i``, the input is ``ids[i : i+block_size]`` and the
    target is the same window shifted one position: ``ids[i+1 : i+1+block_size]``.

    Parameters
    ----------
    token_ids : torch.Tensor
        A 1-D ``long`` tensor of token ids.
    block_size : int
        Context length of each training example.
    stride : int, default 1
        Step between consecutive window starts. ``stride=1`` yields every window
        (best for evaluation); ``stride=block_size`` yields non-overlapping blocks
        (far fewer, much faster passes — usual for training).
    """

    def __init__(self, token_ids: torch.Tensor, block_size: int, *, stride: int = 1) -> None:
        if token_ids.dim() != 1:
            raise ValueError(f"token_ids must be 1-D, got shape {tuple(token_ids.shape)}")
        if token_ids.numel() <= block_size:
            raise ValueError(
                f"corpus has only {token_ids.numel()} tokens after encoding, but block_size "
                f"is {block_size} (need more than {block_size + 1}). Use a larger corpus, a "
                f"smaller model.block_size, or — with BPE — a smaller tokenizer.vocab_size "
                f"(subword merges can compress a small corpus below block_size)."
            )
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.data = token_ids.long()
        self.block_size = block_size
        self.stride = stride

    def __len__(self) -> int:
        return (self.data.numel() - self.block_size - 1) // self.stride + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        chunk = self.data[start : start + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def encode_corpus(text: str, tokenizer: object) -> torch.Tensor:
    """Encode a whole corpus string into a 1-D ``long`` id tensor."""
    ids = tokenizer.encode(text)  # type: ignore[attr-defined]
    return torch.tensor(ids, dtype=torch.long)


def load_corpus_file(path: str | Path, tokenizer: object) -> torch.Tensor:
    """Read a UTF-8 text file and encode it into a 1-D id tensor."""
    text = Path(path).read_text(encoding="utf-8")
    return encode_corpus(text, tokenizer)


def train_val_split(
    token_ids: torch.Tensor, val_fraction: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a contiguous id sequence into train / val by position (no shuffling)."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    split = int(token_ids.numel() * (1.0 - val_fraction))
    return token_ids[:split], token_ids[split:]
