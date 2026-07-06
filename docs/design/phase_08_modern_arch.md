# Phase 8 — Modern (Llama-style) architecture

## Goal
Move Zenith's decoder from the GPT-2-era recipe (LayerNorm, learned positions,
GELU MLP) to the modern Llama-style stack — **RMSNorm, RoPE, SwiGLU** — written
from scratch, and *measure* whether it actually helps rather than assuming it does.

## Scope (what ships this phase)
- `zenith/models/decoder.py`:
  - `RMSNorm` — root-mean-square layer norm (no mean-subtraction, no bias).
  - **RoPE** — rotary position embeddings applied to q/k in attention. Registered
    as `cos`/`sin` buffers; the attention takes a position `offset` so it stays
    correct under the KV-cache (position i is rotated by i whether it arrives in a
    full prompt pass or one-token-at-a-time decoding).
  - `SwiGLU` — gated FFN (`(SiLU(xW_gate) * xW_up) W_down`), ~⅔ hidden to match
    param count of a GELU MLP.
  - All three are selectable via `DecoderConfig.{norm, positional, ffn}`; the
    default is Llama-style, GPT-2-style is still fully supported.
- Config plumbing: `model.{norm,positional,ffn}` in Hydra; `cli/train.py` wired.
- Backward-compatible checkpoint load: pre-v0.9 checkpoints lack the new fields, so
  the loader defaults them to `layernorm`/`learned`/`gelu`.
- Tests: RMSNorm unit-RMS property; the GPT-2-style variant builds/runs; the
  existing KV-cache-equivalence test now also guards RoPE + cache.

## Key decisions (ADR-0009)
- **Configurable, not a hard switch.** Both recipes live behind one config so the
  architecture ablation is a first-class, reproducible experiment — not a git diff.
- **Correctness before speed.** RoPE-under-cache equivalence is verified against a
  no-cache full forward; that invariant is the point, not micro-optimisation.
- **Measure the claim.** "Modern = better" is only asserted with a same-params,
  same-recipe ablation in BENCHMARKS.

## Notes / verification (real numbers, trained on MPS)
- Llama-style: **2.08 bits/char**, converged by ~epoch 10.
- GPT-2-style: **2.11 bits/char**, converged by ~epoch 17.
- Honest finding: the modern stack is slightly better *and* converges ~2× faster at
  fewer params, but both land near ~2.1 — at this scale the floor is set by data +
  model size, not architecture. Convergence speed is the real win here.

## Out of scope
- SDPA/flash-attention kernels (a later bonus slice); RoPE scaling / long-context
  tricks; GQA/MQA. This phase is the architecture *primitives*, not inference perf.

## Definition of done
RMSNorm/RoPE/SwiGLU land and are configurable; RoPE is correct under the KV-cache;
old checkpoints still load; the architecture ablation is in BENCHMARKS; gates green;
tag `v0.9.0`.
