# ADR-0009: Modern (Llama-style) architecture as the default, chosen by measurement

## Status
Accepted. Refines ADR-0007 (which, written before a GPU run was available, left
benchmarks as methodology-only): Zenith now trains real models and reports real,
earned numbers — never invented ones.

## Context
The original decoder was GPT-2-era: LayerNorm, learned absolute positions, a GELU
MLP. The modern open-LM stack (Llama and successors) uses RMSNorm, rotary position
embeddings (RoPE), and a gated SwiGLU FFN. These are widely assumed to be strictly
better. For a from-scratch learning project the interesting question is not "copy
Llama" but "implement these primitives correctly and measure whether they help at
our scale."

## Decision
1. **Implement RMSNorm, RoPE, and SwiGLU from scratch** and make each selectable via
   `DecoderConfig.{norm, positional, ffn}`. The Llama-style combination is the
   default; the GPT-2-style recipe remains fully supported.
2. **RoPE must be correct under the KV-cache.** Positions are rotated by absolute
   index via a position `offset`, so a token gets the same rotation whether it
   arrives in a full prompt pass or one step at a time. This equivalence is a test,
   not a hope.
3. **Backward compatibility is preserved.** Pre-v0.9 checkpoints omit the new config
   fields; the loader defaults them to `layernorm`/`learned`/`gelu`, so old models
   keep loading.
4. **Architecture claims are earned by ablation.** BENCHMARKS carries a same-params,
   same-recipe GPT-2-vs-Llama comparison, and a bits/char-vs-size scaling study
   (Phase 9). We report what we measured, including the null part of the result.

## Consequences
- One config spans both architectures, so the ablation is reproducible, not a diff.
- Measured outcome: Llama-style reaches 2.08 bits/char vs 2.11 for GPT-2-style, at
  fewer params and ~2× faster convergence — but both hit ~2.1, because at this
  scale the ceiling is data/size-bound, not architecture-bound. The honest headline
  is *convergence speed*, not a dramatic quality jump.
- ADR-0007's "benchmark table stays empty until someone runs it" no longer holds:
  the table is filled with real MPS runs, and the honesty rule now reads "report
  only numbers we actually produced."
