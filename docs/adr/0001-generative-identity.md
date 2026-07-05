# ADR-0001: Generative identity; Polaris is an optional sibling

## Status
Accepted.

## Context
Zenith began as a general, from-scratch NLP framework. It overlapped heavily with
Polaris — a separate, stable, encoder-focused NLP engine by the same author —
which risked the two projects being perceived as copies of each other. Zenith
needed a distinct reason to exist that is genuinely useful on its own, complements
Polaris without being built *for* it, and does not duplicate anything Polaris
already does.

## Decision
Zenith is a **from-scratch generative NLP library**: decoder-only transformer
language models and text generation.

- Polaris is **encoder / classification** (understanding text). Zenith is
  **decoder / generation** (producing text). No functional overlap.
- Polaris is an **optional interop extra** (`pip install zenith-nlp[polaris]`),
  never a required dependency. Zenith ships its own tokenizer and stands entirely
  on its own; if Polaris is present, Zenith may reuse its tokenizers for interop.
- Zenith mirrors Polaris' engineering discipline (from-scratch on tensor
  primitives, vertical slices, strict typing, offline tests, ADRs, semver) but is
  an independent codebase.

## Consequences
- The prior encoder-wrapper direction is abandoned; the earlier general-NLP code
  is frozen under `src/zenith/_incubating/` (see its README).
- "Helpful to Polaris, not for Polaris": Zenith becomes a real downstream consumer
  of Polaris' tokenizers when opted in, but its purpose is independent.
- Distributed training and PEFT are retained as roadmap items, wired in only when
  the training loop needs them — not built speculatively.
