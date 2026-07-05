# ADR-0002: Byte-level tokenizer first; defer learned BPE

## Status
Accepted.

## Context
A language model needs a tokenizer, but Phase 1's goal is a runnable end-to-end
slice (train → generate) with the fewest moving parts and zero external
dependencies, so Zenith can stand alone.

## Decision
Ship a **byte-level tokenizer** (`ByteTokenizer`) as the built-in default: UTF-8
bytes → ids, plus a few reserved special tokens (BOS/EOS/PAD).

- No training step, no vocabulary files, no dependency.
- Every possible string round-trips losslessly.
- Fixed, tiny vocabulary (259).

A learned subword tokenizer (from-scratch BPE) and an optional adapter to Polaris'
tokenizers are deferred to a later slice, added when there is a concrete need
(longer contexts, efficiency) — not in advance.

## Consequences
- Phase 1 is dependency-free and always works on arbitrary input.
- Sequences are longer than with subwords (one id per byte); acceptable at the
  small scale of Phase 1 and revisited when scaling up.
