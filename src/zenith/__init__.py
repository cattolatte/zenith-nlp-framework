"""Zenith — a from-scratch generative NLP library.

Decoder-only transformer language models and text generation, built on PyTorch
tensor primitives. Zenith is a standalone project; it can *optionally* interoperate
with the ``polaris-nlp`` engine (an encoder-side sibling) but does not depend on it.

Public surface:

- :class:`DecoderLM` / :class:`DecoderConfig` — the model
- :class:`ByteTokenizer` — the built-in tokenizer
- :class:`CausalLMDataset` — next-token training data
- :class:`Generator` — text generation
- :class:`CausalLMTrainer` / :class:`TrainingConfig` — training
- :func:`save_checkpoint` / :func:`load_checkpoint` / :func:`load_pretrained`
"""

from __future__ import annotations

from .checkpoint import load_checkpoint, load_pretrained, save_checkpoint
from .data import CausalLMDataset
from .generation import Generator
from .models import DecoderConfig, DecoderLM
from .peft import LoraConfig, inject_lora
from .tokenizers import ByteTokenizer
from .training import CausalLMTrainer, TrainingConfig

__version__ = "0.5.0"

__all__ = [
    "__version__",
    "DecoderLM",
    "DecoderConfig",
    "ByteTokenizer",
    "CausalLMDataset",
    "Generator",
    "CausalLMTrainer",
    "TrainingConfig",
    "LoraConfig",
    "inject_lora",
    "save_checkpoint",
    "load_checkpoint",
    "load_pretrained",
]
