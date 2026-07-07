"""Save and load Zenith language-model checkpoints.

Design Principles
-----------------
A checkpoint is self-describing: it carries the model's ``DecoderConfig`` and the
tokenizer spec alongside the weights, so a model can be rebuilt and used for
generation from the file alone — no need to remember how it was constructed.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .generation import Generator
from .models import DecoderConfig, DecoderLM
from .tokenizers import BPETokenizer, ByteTokenizer

__all__ = ["save_checkpoint", "load_checkpoint", "load_pretrained"]

_FORMAT = "zenith-lm-v1"


def _tokenizer_spec(tokenizer: object | None) -> dict[str, Any]:
    if tokenizer is None or isinstance(tokenizer, ByteTokenizer):
        return {"type": "byte"}
    if isinstance(tokenizer, BPETokenizer):
        return tokenizer.to_dict()  # {"type": "bpe", "merges": [...]}
    return {"type": type(tokenizer).__name__}


def _build_tokenizer(spec: dict[str, Any]) -> ByteTokenizer | BPETokenizer:
    kind = spec.get("type", "byte")
    if kind == "byte":
        return ByteTokenizer()
    if kind == "bpe":
        return BPETokenizer.from_dict(spec)
    raise ValueError(f"unsupported tokenizer type in checkpoint: {kind!r}")


def save_checkpoint(
    model: DecoderLM,
    tokenizer: object | None,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
    lora: object | None = None,
) -> None:
    """Write ``model`` (+ tokenizer spec + optional ``extra``/``lora``) to ``path``.

    When ``lora`` (a ``LoraConfig``) is given, the model has LoRA layers injected;
    the config is recorded so :func:`load_checkpoint` can re-inject them before
    loading the weights.
    """
    payload: dict[str, Any] = {
        "format": _FORMAT,
        "model_config": asdict(model.config),
        "state_dict": model.state_dict(),
        "tokenizer": _tokenizer_spec(tokenizer),
    }
    if lora is not None:
        payload["lora"] = asdict(lora)
    if extra:
        payload["extra"] = extra
    torch.save(payload, str(path))


def load_checkpoint(
    path: str | Path, *, map_location: str = "cpu"
) -> tuple[DecoderLM, ByteTokenizer | BPETokenizer]:
    """Rebuild ``(model, tokenizer)`` from a checkpoint file."""
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    if payload.get("format") != _FORMAT:
        raise ValueError(f"not a {_FORMAT} checkpoint: {path}")
    model_config = dict(payload["model_config"])
    # Pre-v0.9 checkpoints predate configurable architecture — default to GPT-2-style.
    model_config.setdefault("norm", "layernorm")
    model_config.setdefault("positional", "learned")
    model_config.setdefault("ffn", "gelu")
    model_config.setdefault("attention", "eager")
    model = DecoderLM(DecoderConfig(**model_config))
    if "lora" in payload:  # re-inject adapters so the state-dict keys line up
        from .peft import LoraConfig, inject_lora

        inject_lora(model, LoraConfig(**payload["lora"]))
    model.load_state_dict(payload["state_dict"])
    tokenizer = _build_tokenizer(payload.get("tokenizer", {"type": "byte"}))
    model.tokenizer = tokenizer  # attach for ergonomic Generator(model) use
    return model, tokenizer


def load_pretrained(path: str | Path, *, map_location: str = "cpu") -> Generator:
    """Load a checkpoint straight into a ready-to-use :class:`Generator`."""
    model, tokenizer = load_checkpoint(path, map_location=map_location)
    return Generator(model, tokenizer)
