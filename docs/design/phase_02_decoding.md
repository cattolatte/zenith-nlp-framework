# Phase 2 — Decoding strategies & KV-cache

## Goal
Make generation good and fast: the standard decoding toolkit, plus an efficient
autoregressive path — without changing training behaviour.

## Scope (what ships this phase)
- `generation/Generator` gains:
  - **Sampling controls**: `top_k`, `top_p` (nucleus), `repetition_penalty`,
    composed with temperature/greedy.
  - **Beam search**: `beam_search_ids`, deterministic and length-normalized;
    `generate(..., num_beams=N)` routes to it.
- `models/KVCache` + cache-aware `DecoderLM.forward(input_ids, cache=...)`:
  incremental decoding that appends per-step keys/values and masks by offset.
- Tests: cache-vs-full-forward logit equivalence; `top_k=1`, tiny `top_p`, and
  `num_beams=1` all reduce to greedy; validity/bounds; block-size stop.

## Key decisions
- **The cacheless path is untouched.** `cache=None` uses the original triangular
  mask, so v0.1.0 training/inference behaviour is byte-for-byte preserved. See
  ADR-0003.
- **Cache correctness is anchored by a test**, not just review: incremental logits
  must match a full recompute (`allclose`). This is essential because the change
  is subtle and easy to get wrong.
- **Beam search recomputes each step** (no cache) for clarity; caching beams is a
  later optimization if needed.
- Cached sampling is bounded by `block_size` (the model cannot attend beyond its
  context); generation stops there rather than silently sliding.

## Out of scope (later)
- KV-cache for beam search; sliding-window / cache eviction for >block_size.
- LoRA/QLoRA and distributed training (Phase 3).

## Definition of done
All decoding strategies work and are tested; cache equivalence holds; gates green;
tag `v0.2.0`.
