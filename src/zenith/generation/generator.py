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

from dataclasses import dataclass
from typing import Iterable, Iterator

import torch

from ..models import DecoderLM, KVCache
from ..tokenizers import ByteTokenizer

__all__ = ["Generator", "SpeculativeStats"]


@dataclass
class SpeculativeStats:
    """Diagnostics from a speculative-decoding run.

    ``target_forwards`` is the count that matters: plain greedy decoding needs one
    target forward per generated token, so ``tokens / target_forwards`` is the
    speedup factor over greedy (assuming the draft is comparatively free).
    """

    tokens: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    proposed: int = 0
    accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def speedup(self) -> float:
        """Target forwards saved vs greedy: tokens per target forward."""
        return self.tokens / self.target_forwards if self.target_forwards else 0.0


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
        stop_ids: Iterable[int] | None = None,
    ) -> torch.Tensor:
        """Extend ``input_ids`` (shape ``(batch, seq)``) by sampling.

        ``temperature <= 0`` selects greedy decoding. ``use_cache`` toggles the
        KV-cache fast path (numerically equivalent to the uncached path).
        ``stop_ids`` halts a single-sequence run before a stop token is appended
        (e.g. EOS for instruction chat).
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        block_size = self.model.config.block_size
        idx = input_ids.to(device)
        stop = set(stop_ids) if stop_ids is not None else None

        if use_cache:
            cache: KVCache | None = KVCache(self.model.config.num_layers)
            logits = self.model(idx[:, -block_size:], cache=cache)[:, -1, :]
            for _ in range(max_new_tokens):
                next_id = self._select(logits, idx, temperature, top_k, top_p, repetition_penalty)
                if stop is not None and int(next_id[0, 0]) in stop:
                    break
                idx = torch.cat([idx, next_id], dim=1)
                if idx.size(1) >= block_size:
                    break  # context window is full; cannot condition on more
                logits = self.model(next_id, cache=cache)[:, -1, :]
        else:
            for _ in range(max_new_tokens):
                logits = self.model(idx[:, -block_size:])[:, -1, :]
                next_id = self._select(logits, idx, temperature, top_k, top_p, repetition_penalty)
                if stop is not None and int(next_id[0, 0]) in stop:
                    break
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

    # -- speculative decoding ------------------------------------------------

    @torch.no_grad()
    def speculative_generate_ids(
        self,
        input_ids: torch.Tensor,
        draft_model: DecoderLM,
        *,
        max_new_tokens: int = 100,
        lookahead: int = 4,
    ) -> tuple[torch.Tensor, SpeculativeStats]:
        """Greedy speculative decoding — provably identical output to greedy.

        A small ``draft_model`` proposes ``lookahead`` tokens each round; the target
        (``self.model``) verifies them in a single forward pass and accepts the
        longest prefix its own argmax agrees with, plus one correction/bonus token.
        The result is byte-for-byte the same as :meth:`generate_ids` with
        ``temperature=0`` on the target, but with fewer target forward passes.

        Both models must share a vocabulary and context length. Operates on a single
        sequence (batch size 1). Returns ``(ids, stats)``.
        """
        if input_ids.size(0) != 1:
            raise ValueError("speculative decoding operates on a single sequence (batch size 1)")
        if lookahead < 1:
            raise ValueError("lookahead must be >= 1")
        target = self.model
        target.eval()
        draft_model.eval()
        device = next(target.parameters()).device
        block_size = target.config.block_size
        idx = input_ids.to(device)

        t_cache = KVCache(target.config.num_layers)
        d_cache = KVCache(draft_model.config.num_layers)
        if idx.size(1) > 1:  # prime both caches on all but the current token
            target(idx[:, :-1], cache=t_cache)
            draft_model(idx[:, :-1], cache=d_cache)
        last = idx[:, -1:]  # current token; not yet in either cache

        stats = SpeculativeStats()
        new_tokens: list[int] = []
        while len(new_tokens) < max_new_tokens:
            base = t_cache.length  # positions committed before `last`
            # Room for k drafts + the correction, keeping the final position < block_size.
            room = block_size - base - 2
            k = min(lookahead, room, max_new_tokens - len(new_tokens))
            if k < 1:
                break

            # Draft proposes k tokens greedily, extending its own cache.
            draft_tokens: list[torch.Tensor] = []
            cur = last
            for _ in range(k):
                d_logits = draft_model(cur, cache=d_cache)[:, -1, :]
                stats.draft_forwards += 1
                cur = torch.argmax(d_logits, dim=-1, keepdim=True)
                draft_tokens.append(cur)

            # Target verifies [last, t_1..t_k] in one pass; argmax at each position.
            verify_in = torch.cat([last, *draft_tokens], dim=1)
            t_logits = target(verify_in, cache=t_cache)
            stats.target_forwards += 1
            target_arg = torch.argmax(t_logits[0], dim=-1)  # (k+1,)

            m = 0
            while m < k and int(draft_tokens[m].item()) == int(target_arg[m].item()):
                m += 1
            correction = target_arg[m].view(1, 1)  # first divergence, or bonus if m == k
            stats.proposed += k
            stats.accepted += m

            for i in range(m):
                new_tokens.append(int(draft_tokens[i].item()))
            new_tokens.append(int(correction.item()))

            # Roll both caches back to the committed prefix [.., last, t_1..t_m].
            t_cache.truncate(base + 1 + m)
            if m == k:  # draft never fed t_k to itself; sync it before truncating
                draft_model(draft_tokens[k - 1], cache=d_cache)
                stats.draft_forwards += 1
            else:
                d_cache.truncate(base + 1 + m)
            last = correction

        new_tokens = new_tokens[:max_new_tokens]  # a final round may overshoot by the correction
        stats.tokens = len(new_tokens)
        out = torch.cat([idx, torch.tensor([new_tokens], dtype=torch.long, device=device)], dim=1)
        return out, stats

    def speculative_generate(
        self,
        prompt: str,
        draft_model: DecoderLM,
        *,
        max_new_tokens: int = 100,
        lookahead: int = 4,
    ) -> tuple[str, SpeculativeStats]:
        """Text wrapper around :meth:`speculative_generate_ids`."""
        prompt_ids = self.tokenizer.encode(prompt) or [self.tokenizer.bos_id]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        out, stats = self.speculative_generate_ids(
            input_ids, draft_model, max_new_tokens=max_new_tokens, lookahead=lookahead
        )
        return self.tokenizer.decode(out[0, len(prompt_ids):].tolist()), stats

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

    def stream(
        self,
        prompt: str = "",
        *,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        stop_ids: Iterable[int] | None = None,
    ) -> Iterator[str]:
        """Yield decoded text incrementally, one UTF-8-complete chunk at a time.

        Bytes are buffered until they form a valid character, so a partial
        multibyte sequence is never yielded mid-way. Uses the KV-cache fast path.
        ``stop_ids`` ends the stream when a stop token (e.g. EOS) is produced.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        block_size = self.model.config.block_size
        stop = set(stop_ids) if stop_ids is not None else None

        prompt_ids = self.tokenizer.encode(prompt) or [self.tokenizer.bos_id]
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        cache = KVCache(self.model.config.num_layers)

        with torch.no_grad():
            logits = self.model(idx[:, -block_size:], cache=cache)[:, -1, :]

        buffer = bytearray()
        for _ in range(max_new_tokens):
            next_id = self._select(logits, idx, temperature, top_k, top_p, repetition_penalty)
            if stop is not None and int(next_id[0, 0]) in stop:
                break
            idx = torch.cat([idx, next_id], dim=1)

            buffer.extend(self.tokenizer.token_bytes(int(next_id.item())))
            try:
                yield buffer.decode("utf-8")
                buffer.clear()
            except UnicodeDecodeError:
                pass  # wait for the rest of the multibyte character

            if idx.size(1) >= block_size:
                break
            with torch.no_grad():
                logits = self.model(next_id, cache=cache)[:, -1, :]

        if buffer:
            yield buffer.decode("utf-8", errors="replace")
