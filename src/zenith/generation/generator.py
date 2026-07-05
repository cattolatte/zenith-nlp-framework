"""Text generation — the capability Polaris will never have.

Design Principles
-----------------
Generation is Zenith's headline feature, so it gets its own module rather than
being buried as a model method. Phase 1 implements the two most fundamental
decoding strategies — greedy (``temperature == 0``) and temperature sampling —
autoregressively, one token at a time, always re-conditioning on the last
``block_size`` tokens.

Deliberately deferred to Phase 2 (kept out of here on purpose): top-k / nucleus
(top-p) sampling, beam search, and a KV-cache. This module is the correct floor.
"""

from __future__ import annotations

import torch

from ..models import DecoderLM
from ..tokenizers import ByteTokenizer

__all__ = ["Generator"]


class Generator:
    """Autoregressive sampler around a :class:`DecoderLM`.

    Parameters
    ----------
    model : DecoderLM
        A trained decoder language model.
    tokenizer : ByteTokenizer, optional
        Used by :meth:`generate` for text in/out. If omitted, falls back to a
        ``tokenizer`` attribute on the model, else a fresh :class:`ByteTokenizer`.
    """

    def __init__(self, model: DecoderLM, tokenizer: ByteTokenizer | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None) or ByteTokenizer()

    @torch.no_grad()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Extend ``input_ids`` (shape ``(batch, seq)``) by ``max_new_tokens``."""
        self.model.eval()
        device = next(self.model.parameters()).device
        block_size = self.model.config.block_size
        idx = input_ids.to(device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self.model(idx_cond)[:, -1, :]
            if temperature <= 0.0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def generate(
        self, prompt: str = "", *, max_new_tokens: int = 100, temperature: float = 1.0
    ) -> str:
        """Generate a continuation of ``prompt`` and return the new text only."""
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.tokenizer.bos_id]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        out = self.generate_ids(
            input_ids, max_new_tokens=max_new_tokens, temperature=temperature
        )
        new_ids = out[0, len(prompt_ids):].tolist()
        return self.tokenizer.decode(new_ids)
