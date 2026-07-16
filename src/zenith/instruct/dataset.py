"""Instruction-tuning dataset with response-only loss masking.

The key idea of supervised fine-tuning: format each ``(instruction, response)`` pair
with the :class:`ChatTemplate`, but only compute loss on the **response** tokens
(and the final EOS). Prompt tokens and padding are labelled ``-100`` so they're
ignored by ``nn.CrossEntropyLoss`` — the model learns to *produce* answers, not to
re-predict the instruction it was given.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .template import ChatTemplate

__all__ = ["InstructionDataset", "load_instructions", "mask_prompt"]

IGNORE_INDEX = -100  # nn.CrossEntropyLoss default ignore_index


def mask_prompt(
    prompt_ids: list[int], response_ids: list[int], *, pad_id: int, max_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a fixed-length ``(input, target)`` pair supervising only the response.

    ``prompt_ids`` and trailing padding are labelled :data:`IGNORE_INDEX`; the
    ``response_ids`` (which already include any trailing EOS) are supervised. Shared
    by :class:`InstructionDataset` and the grounded dataset so the masking lives in
    one place.
    """
    ids = (prompt_ids + response_ids)[:max_length]
    n_prompt = min(len(prompt_ids), len(ids))
    content_len = len(ids)  # everything after this is padding
    full = ids + [pad_id] * (max_length - len(ids))
    # Label = the token, except prompt and pad positions which are ignored.
    labels = [tok if n_prompt <= i < content_len else IGNORE_INDEX for i, tok in enumerate(full)]
    # Shift for next-token prediction: input predicts the following label.
    return torch.tensor(full[:-1], dtype=torch.long), torch.tensor(labels[1:], dtype=torch.long)


class InstructionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Fixed-length ``(input, target)`` pairs with prompt/pad tokens masked out.

    Parameters
    ----------
    pairs : list of (instruction, response)
    tokenizer : a Zenith tokenizer (needs ``encode``/``eos_id``/``pad_id``).
    max_length : int
        Examples are right-padded (or truncated) to this many tokens.
    template : ChatTemplate, optional
    """

    def __init__(self, pairs, tokenizer, *, max_length: int, template: ChatTemplate | None = None):
        if max_length < 2:
            raise ValueError("max_length must be >= 2")
        self.template = template or ChatTemplate()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = [self._encode(instruction, response) for instruction, response in pairs]

    def _encode(self, instruction: str, response: str) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_ids = self.tokenizer.encode(self.template.format_prompt(instruction))
        response_ids = self.tokenizer.encode(response) + [self.tokenizer.eos_id]
        return mask_prompt(
            prompt_ids, response_ids, pad_id=self.tokenizer.pad_id, max_length=self.max_length
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


def load_instructions(path: str | Path) -> list[tuple[str, str]]:
    """Read a JSONL file of ``{"instruction": ..., "response": ...}`` objects."""
    pairs: list[tuple[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        pairs.append((obj["instruction"], obj["response"]))
    return pairs
