"""Instruction fine-tuning — turn a base decoder into a mini chat model."""

from __future__ import annotations

from .dataset import InstructionDataset, load_instructions
from .template import ChatTemplate

__all__ = ["ChatTemplate", "InstructionDataset", "load_instructions"]
