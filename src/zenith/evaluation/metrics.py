"""Language-model evaluation: token-level loss and perplexity.

Design Principles
-----------------
The right metric for a language model is per-token perplexity on held-out text —
``exp`` of the mean next-token cross-entropy. Computed with a summed loss over all
predicted tokens (not a per-batch mean), so batching does not change the number.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..models import DecoderLM
from ..utils import resolve_device

__all__ = ["evaluate", "perplexity"]


@torch.no_grad()
def evaluate(
    model: DecoderLM,
    dataset: Dataset,
    *,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, float]:
    """Return ``{"loss", "perplexity"}`` for ``model`` over ``dataset``."""
    model.eval()
    dev = torch.device(device) if device else resolve_device()
    model.to(dev)

    loader = DataLoader(dataset, batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(dev), targets.to(dev)
        logits = model(inputs)
        total_loss += loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item()
        total_tokens += int(targets.numel())

    mean_loss = total_loss / max(total_tokens, 1)
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20.0))}


def perplexity(
    model: DecoderLM, dataset: Dataset, *, batch_size: int = 32, device: str | None = None
) -> float:
    """Held-out perplexity of ``model`` over ``dataset``."""
    return evaluate(model, dataset, batch_size=batch_size, device=device)["perplexity"]
