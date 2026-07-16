"""Grounded supervised fine-tuning — passages in the prompt, a cited answer (or an
abstention) in the target.

The instruction pipeline (:mod:`zenith.instruct.dataset`) formats generic
``(instruction, response)`` pairs. Grounded SFT differs only in the *shape* of an
example: the prompt carries retrieved ``(passage_id, text)`` passages and the target
is either the caller's answer (which may contain citations) or the reserved
``<abstain>`` token for an unanswerable question. The fixed-length, response-only
masking is reused unchanged.

This module is **format machinery only**. Zenith attaches no meaning to passage ids,
citation syntax, or what makes a question answerable — the caller builds the examples
and decides the citation policy (see the constrained-decoding hook in
:mod:`zenith.generation.constraints`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from .dataset import InstructionDataset, mask_prompt

__all__ = ["GroundedTemplate", "GroundedInstructionDataset", "GroundedExample"]

# (question, passages as (passage_id, text), answer | None-for-abstain)
GroundedExample = tuple[str, Sequence[tuple[str, str]], "str | None"]


class _Tokenizer(Protocol):
    """The tokenizer surface grounded SFT needs (byte or BPE both satisfy this)."""

    eos_id: int
    pad_id: int
    abstain_id: int

    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class GroundedTemplate:
    """Format ``(question, passages)`` into a prompt; the target is supplied separately.

    ``passages`` are ``(passage_id, text)`` — the ids are what a citation may
    reference, but their meaning is the caller's. This is plain text (no special
    vocabulary), so it works with the byte tokenizer unchanged.
    """

    context_header: str = "### Context:\n"
    question_header: str = "\n\n### Question:\n"
    answer_header: str = "\n\n### Answer:\n"

    def format_prompt(self, question: str, passages: Sequence[tuple[str, str]]) -> str:
        """The prompt half: the passages, then the question, up to the answer slot."""
        blocks = "\n".join(f"[{passage_id}] {text}" for passage_id, text in passages)
        return f"{self.context_header}{blocks}{self.question_header}{question}{self.answer_header}"

    def format_example(
        self, question: str, passages: Sequence[tuple[str, str]], answer: str
    ) -> str:
        """A full training string: prompt + answer (EOS is added by the dataset)."""
        return f"{self.format_prompt(question, passages)}{answer}"


class GroundedInstructionDataset(InstructionDataset):
    """``(question + passages) -> (cited answer, or <abstain>)``, prompt tokens masked.

    Reuses :class:`InstructionDataset`'s fixed-length, response-only masking
    (:func:`~zenith.instruct.dataset.mask_prompt`); only the grounded prompt
    formatting and the abstain target are new.

    Parameters
    ----------
    examples : sequence of (question, passages, answer)
        ``passages`` is a sequence of ``(passage_id, text)``. ``answer`` is the cited
        answer string, or ``None`` to train an **abstention** — the target becomes
        the tokenizer's abstain token.
    tokenizer : a Zenith tokenizer (needs ``encode``/``eos_id``/``pad_id``/``abstain_id``).
    max_length : int
        Examples are right-padded (or truncated) to this many tokens.
    template : GroundedTemplate, optional
    """

    def __init__(
        self,
        examples: Sequence[GroundedExample],
        tokenizer: _Tokenizer,
        *,
        max_length: int,
        template: GroundedTemplate | None = None,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be >= 2")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.grounded_template = template or GroundedTemplate()
        self.examples = [
            self._encode_grounded(q, passages, answer) for q, passages, answer in examples
        ]

    def _encode_grounded(
        self, question: str, passages: Sequence[tuple[str, str]], answer: str | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_ids = self.tokenizer.encode(self.grounded_template.format_prompt(question, passages))
        if answer is None:  # train an abstention
            response_ids = [self.tokenizer.abstain_id, self.tokenizer.eos_id]
        else:
            response_ids = self.tokenizer.encode(answer) + [self.tokenizer.eos_id]
        return mask_prompt(
            prompt_ids, response_ids, pad_id=self.tokenizer.pad_id, max_length=self.max_length
        )
