"""LoRA (Low-Rank Adaptation) — parameter-efficient fine-tuning for the decoder.

Design Principles
-----------------
Fine-tuning a language model by training a tiny low-rank adapter, with the base
weights frozen, is the workhorse of efficient adaptation. This module implements
it from scratch and generically: it walks a model's ``nn.Linear`` sub-modules and
replaces the targeted ones with a :class:`LoRALinear` that adds a trainable
``B @ A`` update while freezing the original weight.

It knows nothing about the decoder's internals beyond "it is made of
``nn.Linear`` layers with dotted names" — targeting is by name substring, so which
projections get adapted is a caller decision, not a hard-coded one. Use
:func:`find_linear_modules` to discover names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

__all__ = [
    "LoraConfig",
    "LoRALinear",
    "find_linear_modules",
    "inject_lora",
    "mark_only_lora_as_trainable",
    "lora_parameters",
    "lora_state_dict",
    "count_trainable_parameters",
]


@dataclass(frozen=True)
class LoraConfig:
    """Configuration for a LoRA injection pass.

    Parameters
    ----------
    rank : int
        Rank of the low-rank update (``r``).
    alpha : int
        Scaling numerator; the update is scaled by ``alpha / rank``.
    dropout : float
        Dropout applied to the LoRA input branch.
    target_modules : tuple[str, ...]
        Name substrings selecting which ``nn.Linear`` layers to adapt. Empty
        adapts every linear. Default targets attention projections (``qkv`` and
        ``proj`` both live under ``.attention.``).
    """

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("attention",)


class LoRALinear(nn.Module):
    """Wraps an ``nn.Linear``, freezing it and adding a trainable low-rank update.

    ``y = W0 x  +  (alpha / rank) * B (A x)`` with ``W0`` frozen, ``A``/``B``
    trainable and ``B`` zero-initialised (so training starts as a no-op).
    """

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: int, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base_layer = base_layer
        self.scaling = alpha / rank

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return self.base_layer(x) + self.scaling * update


def find_linear_modules(model: nn.Module) -> list[str]:
    """Dotted paths of every ``nn.Linear`` in ``model`` (to pick target_modules)."""
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def _matches(name: str, targets: tuple[str, ...]) -> bool:
    return True if not targets else any(t in name for t in targets)


def _parent_of(model: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parent = model
    *parents, child = dotted.split(".")
    for p in parents:
        parent = getattr(parent, p)
    return parent, child


def inject_lora(model: nn.Module, config: LoraConfig) -> nn.Module:
    """Replace matching ``nn.Linear`` layers with :class:`LoRALinear` in place.

    Also freezes all non-LoRA parameters. Returns the same model.
    """
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _matches(name, config.target_modules)
    ]
    if not targets:
        raise ValueError(
            f"inject_lora matched no nn.Linear modules for target_modules={config.target_modules!r}. "
            "Call zenith.peft.find_linear_modules(model) to see available names."
        )
    for name in targets:
        parent, child = _parent_of(model, name)
        base = getattr(parent, child)
        setattr(parent, child, LoRALinear(base, config.rank, config.alpha, config.dropout))
    mark_only_lora_as_trainable(model)
    return model


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Freeze every parameter except the injected LoRA ``A``/``B`` matrices."""
    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """The trainable LoRA parameters — pass these to the optimizer."""
    return [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Just the LoRA weights — a tiny, portable adapter checkpoint."""
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
