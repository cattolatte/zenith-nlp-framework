# Changelog

All notable changes to Zenith are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] — Decoding strategies & KV-cache

Depth for generation: richer decoding and efficient autoregressive inference.

### Added
- **Decoding strategies** in `Generator`: top-k, nucleus (top-p), and a
  repetition penalty, alongside temperature and greedy.
- **Beam search** — deterministic, length-normalized (`beam_search_ids` /
  `generate(..., num_beams=N)`).
- **KV-cache** (`KVCache`) for incremental decoding: attention appends per-step
  keys/values and applies an offset-derived causal mask. Enabled by default for
  sampling (`use_cache=True`), numerically equivalent to a full recompute.

### Notes
- Cached sampling is bounded by the model's `block_size` context window.
- Training / full-forward behaviour is unchanged (the cacheless path is identical
  to v0.1.0).

## [0.1.0] — Core generative engine

First slice of Zenith's new identity: a from-scratch generative NLP library.

### Added
- `ByteTokenizer` — built-in, dependency-free byte-level tokenizer.
- `DecoderLM` / `DecoderConfig` — from-scratch decoder-only transformer (causal
  self-attention, pre-norm blocks, tied embeddings).
- `CausalLMDataset` and corpus/split helpers for next-token training.
- `Generator` — greedy and temperature text generation.
- `CausalLMTrainer` / `TrainingConfig` — single-device training with warmup/cosine
  schedule, gradient clipping, best-checkpoint saving, in-training samples, and
  optional MLflow tracking.
- Self-describing checkpoints: `save_checkpoint`, `load_checkpoint`,
  `load_pretrained`.
- Hydra config tree and CLI: `python -m zenith.cli.train`, `zenith generate`,
  `zenith info`.
- ADR-0001 (generative identity), ADR-0002 (byte-level tokenizer), Phase 1 design
  doc, and a full offline test suite.

### Changed
- Reframed the project from a general NLP framework into a generative
  (decoder-only) library. `polaris-nlp` is now an optional interop extra rather
  than a dependency.

### Notes
- The prior general-NLP code is frozen, unshipped, under
  `src/zenith/_incubating/`.
