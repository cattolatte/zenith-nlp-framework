"""Constrained decoding — optional hooks that reshape next-token logits.

A :class:`LogitsConstraint` is applied by the :class:`~zenith.generation.Generator`
to the next-token logits at every decoding step, *before* sampling. The default
(``None``) leaves decoding unchanged, so the hook is fully backward-compatible.

This is a **general mechanism**: Zenith supplies the machinery (masking the logits to
an allowed id set at chosen trigger positions); the caller supplies *which* ids are
valid — for example, restricting a citation slot to the ids of passages actually in
the prompt. Zenith holds no notion of citations, retrieval, or policy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

__all__ = ["LogitsConstraint", "AllowedTokens"]


@runtime_checkable
class LogitsConstraint(Protocol):
    """Callable that maps ``(step, generated_ids, logits) -> logits``.

    Parameters
    ----------
    step : int
        Index of the token about to be generated (0 for the first new token).
    generated_ids : torch.Tensor
        The sequence so far, shape ``(batch, seq)`` — includes the prompt.
    logits : torch.Tensor
        Next-token logits, shape ``(batch, vocab)``.

    Returns
    -------
    torch.Tensor
        Modified logits of the same shape (e.g. ``-inf`` outside an allowed set).
    """

    def __call__(
        self, step: int, generated_ids: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor: ...


class AllowedTokens:
    """Restrict the next token to ``allowed_ids`` whenever the last emitted token is
    in ``trigger_ids``.

    The canonical use: a token that *opens* a citation is a trigger, so the token
    that follows it must be one of a caller-supplied set of valid ids. Rows whose
    last token is not a trigger are left untouched. The set of trigger and allowed
    ids is entirely the caller's choice — this class encodes no domain policy.

    Parameters
    ----------
    trigger_ids : set of int
        When the previously generated token is one of these, the next token is
        constrained. An empty set never triggers (a no-op).
    allowed_ids : set of int
        The ids the next token may take at a triggered position. Must be non-empty.
    """

    def __init__(self, *, trigger_ids: set[int], allowed_ids: set[int]) -> None:
        if not allowed_ids:
            raise ValueError("allowed_ids must be non-empty")
        self.trigger_ids = frozenset(trigger_ids)
        self.allowed_ids = frozenset(allowed_ids)

    def __call__(
        self, step: int, generated_ids: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        if generated_ids.shape[1] == 0 or not self.trigger_ids:
            return logits
        device = logits.device
        trigger = torch.tensor(sorted(self.trigger_ids), device=device, dtype=torch.long)
        last = generated_ids[:, -1]  # (batch,)
        triggered = (last.unsqueeze(1) == trigger).any(dim=1)  # (batch,)
        if not bool(triggered.any()):
            return logits
        allowed = torch.tensor(sorted(self.allowed_ids), device=device, dtype=torch.long)
        mask = torch.full_like(logits, float("-inf"))
        mask[:, allowed] = 0.0  # allowed columns pass through, the rest go to -inf
        return torch.where(triggered.unsqueeze(1), logits + mask, logits)
