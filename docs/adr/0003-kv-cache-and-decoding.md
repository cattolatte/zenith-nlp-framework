# ADR-0003: KV-cache as an opt-in path; decoding strategies in the Generator

## Status
Accepted.

## Context
Phase 2 adds real decoding (top-k / nucleus / beam) and efficient autoregressive
generation (a key/value cache). The risk is that caching changes numerical results
or complicates the training path, which was clean and tested in v0.1.0.

## Decision
1. **The KV-cache is opt-in and isolated.** `DecoderLM.forward` takes an optional
   `cache`. When `cache is None` (training and full forwards) the original
   precomputed triangular mask is used — unchanged from v0.1.0. When a cache is
   passed, attention appends this step's keys/values and builds the causal mask
   from the running offset. The two paths are numerically equivalent.
2. **Correctness is enforced by a test**, not just review: incremental (cached)
   logits must match a full recompute within tolerance. The change is subtle, so
   the equivalence check is a first-class part of the feature.
3. **Decoding strategies live in the `Generator`**, operating on logits (temperature
   → repetition penalty → top-k → top-p → sample), with beam search as a separate
   deterministic method. Generation logic stays out of the model.

## Consequences
- v0.1.0 training/inference behaviour is preserved exactly.
- Cached sampling is bounded by `block_size`; beam search recomputes each step.
- Future work (cached beams, sliding-window contexts) can build on this without
  revisiting the training path.
