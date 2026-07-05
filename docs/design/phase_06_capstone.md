# Phase 6 — Evaluation, a learned tokenizer & release-readiness

## Goal
Make the library measurable, better on real text, and ready to publish.

## Scope (what ships this phase)
- `tokenizers/BPETokenizer` — from-scratch **byte-level BPE**: trainable, lossless
  (byte fallback), interface-compatible with `ByteTokenizer` (adds `token_bytes`).
- `checkpoint` — serializes the tokenizer, so a **BPE model round-trips**.
- `generation` — streaming generalized to any tokenizer via `token_bytes`.
- `evaluation/` — held-out `evaluate()` / `perplexity()` (token-level).
- CLI: `tokenizer` config group (`byte`/`bpe`), BPE trained on the corpus before
  the model; `zenith eval -m <ckpt> -c <corpus>`.
- `BENCHMARKS.md` — methodology + reproduction (numbers left for a real run).
- `scripts/download_corpus.py` — fetch a public-domain corpus.
- `docs/modules.md` — module overview.
- `.github/workflows/release.yml` — PyPI **trusted publishing** on tag.
- Tests: BPE losslessness / merges / (de)serialization / checkpoint round-trip;
  perplexity; `token_bytes` reconstruction.

## Key decisions (ADR-0007)
- **BPE is byte-level** (starts from the 256 bytes) so it never produces an unknown
  token and always round-trips — losslessness is the invariant the tests pin.
- **No invented benchmark numbers.** This repo is developed without a GPU/torch
  runtime; the benchmark doc ships the *methodology* and a reproduction, with the
  results table populated from a real run. Publishing fake numbers would betray the
  project's honesty discipline.

## Out of scope (later)
- Sweep-result aggregation; regex pre-tokenization for BPE; larger benchmark
  sweeps.

## Definition of done
BPE + evaluation land and are tested; benchmark methodology + release workflow in
place; gates green; tag `v0.6.0`.
