"""Text generation — the capability Polaris will never have.

Design Principles
-----------------
Generation is Zenith's headline feature, so it gets its own module. This provides
the standard decoding toolkit, from scratch and readable:

- **Sampling**: temperature, top-k, nucleus (top-p), and a repetition penalty.
- **Greedy** decoding (``temperature == 0``).
- **Beam search** (deterministic, length-normalized).
- A **KV-cache** fast path for sampling (equivalent to the uncached path — see the
  cache-equivalence test).

Sampling stays within the model's ``block_size`` context; beam search recomputes
each step (no cache) for clarity.
"""

from __future__ import annotations

import torch

from ..models import DecoderLM, KVCache
from ..tokenizers import ByteTokenizer

__all__ = ["Generator"]


class Generator:
    """Autoregressive decoder around a :class:`DecoderLM`.

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

    # -- sampling ------------------------------------------------------------

    @torch.no_grad()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Extend ``input_ids`` (shape ``(batch, seq)``) by sampling.

        ``temperature <= 0`` selects greedy decoding. ``use_cache`` toggles the
        KV-cache fast path (numerically equivalent to the uncached path).
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        block_size = self.model.config.block_size
        idx = input_ids.to(device)

        if use_cache:
            cache: KVCache | None = KVCache(self.model.config.num_layers)
            logits = self.model(idx[:, -block_size:], cache=cache)[:, -1, :]
            for _ in range(max_new_tokens):
                next_id = self._select(logits, idx, temperature, top_k, top_p, repetition_penalty)
                idx = torch.cat([idx, next_id], dim=1)
                if idx.size(1) >= block_size:
                    break  # context window is full; cannot condition on more
                logits = self.model(next_id, cache=cache)[:, -1, :]
        else:
            for _ in range(max_new_tokens):
                logits = self.model(idx[:, -block_size:])[:, -1, :]
                next_id = self._select(logits, idx, temperature, top_k, top_p, repetition_penalty)
                idx = torch.cat([idx, next_id], dim=1)
        return idx

    def _select(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """Turn last-step logits ``(batch, vocab)`` into the next token id."""
        logits = logits.clone()
        if repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            logits = self._top_k(logits, top_k)
        if top_p is not None:
            logits = self._top_p(logits, top_p)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor, generated: torch.Tensor, penalty: float
    ) -> torch.Tensor:
        """Down-weight logits of already-generated tokens (CTRL-style penalty)."""
        score = torch.gather(logits, 1, generated)
        score = torch.where(score > 0, score / penalty, score * penalty)
        return logits.scatter(1, generated, score)

    @staticmethod
    def _top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
        k = min(max(k, 1), logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def _top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Remove tokens once the cumulative probability exceeds p, but always keep
        # the most probable token (shift the mask one position to the right).
        remove = cumulative > p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    # -- beam search ---------------------------------------------------------

    @torch.no_grad()
    def beam_search_ids(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 100,
        num_beams: int = 4,
        length_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Deterministic, length-normalized beam search (single sequence)."""
        if input_ids.size(0) != 1:
            raise ValueError("beam_search_ids operates on a single sequence (batch size 1)")
        self.model.eval()
        device = next(self.model.parameters()).device
        block_size = self.model.config.block_size
        prompt_len = input_ids.size(1)

        beams = input_ids.to(device).repeat(num_beams, 1)
        scores = torch.zeros(num_beams, device=device)
        scores[1:] = float("-inf")  # collapse duplicate initial beams

        for _ in range(max_new_tokens):
            logits = self.model(beams[:, -block_size:])[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            vocab = log_probs.size(-1)
            total = (scores[:, None] + log_probs).view(-1)
            top_scores, top_flat = torch.topk(total, num_beams)
            beam_id = torch.div(top_flat, vocab, rounding_mode="floor")
            token_id = top_flat % vocab
            beams = torch.cat([beams[beam_id], token_id[:, None]], dim=1)
            scores = top_scores

        gen_len = max(beams.size(1) - prompt_len, 1)
        normalized = scores / (gen_len**length_penalty)
        best = beams[torch.argmax(normalized)]
        return best[None, :]

    # -- text convenience ----------------------------------------------------

    def generate(
        self,
        prompt: str = "",
        *,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        num_beams: int | None = None,
    ) -> str:
        """Generate a continuation of ``prompt`` and return the new text only.

        Set ``num_beams`` > 1 for beam search; otherwise sampling is used.
        """
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.tokenizer.bos_id]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)

        if num_beams is not None and num_beams > 1:
            out = self.beam_search_ids(input_ids, max_new_tokens=max_new_tokens, num_beams=num_beams)
        else:
            out = self.generate_ids(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        new_ids = out[0, len(prompt_ids):].tolist()
        return self.tokenizer.decode(new_ids)
