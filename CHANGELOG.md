# Changelog

All notable changes to Zenith are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.0] — Serving & product surface

Talk to a trained model.

### Added
- `zenith.serving`: `create_app(model_path)` (FastAPI) + `serve()` with
  `GET /health`, `POST /generate` (full decoding controls) and `POST
  /generate/stream` (Server-Sent Events).
- `Generator.stream()` — incremental, UTF-8-complete text chunks (KV-cached).
- CLI: `zenith serve` and an interactive `zenith chat` REPL that streams to the
  terminal.
- Packaging: `serving` extra (fastapi/uvicorn/pydantic); `httpx` in `dev`.

### Notes
- Multi-model serving, metrics, batching and auth are deferred. See ADR-0006.

## [0.4.0] — Tracking & reproducibility

Legible, reproducible runs.

### Added
- `zenith.experiments`: `capture_environment()` (versions/seed/git commit) and
  `record_run()` (writes `config`/`metrics`/`samples`/`environment`/`run` files).
- `utils.set_deterministic()` — seeded, deterministic algorithms (best-effort).
- `CausalLMTrainer` now logs the full run config, per-epoch generated samples
  (MLflow artifacts) and the best checkpoint; optionally writes an on-disk run
  record (`record_dir`); supports a `deterministic` mode and a `run_config`.
- Config: `training.{log_samples,deterministic,record_dir}`; the CLI passes the
  resolved Hydra config through for logging/recording.

### Fixed
- `pyproject` version was left at 0.2.0 during 0.3.0; versions are back in sync.

### Notes
- Sweep-result aggregation and the MLflow Model Registry are deferred. See ADR-0005.

## [0.3.0] — Scaling & efficient fine-tuning

Train bigger and cheaper — all opt-in, with the simple path unchanged.

### Added
- **LoRA** (`zenith.peft`): `LoRALinear`, `inject_lora` (targets `nn.Linear` by
  name; defaults to attention `qkv`/`proj`), freeze/parameter helpers,
  `lora_state_dict`, and `find_linear_modules`.
- **Gradient accumulation** (`grad_accum_steps`) and **mixed precision** (`amp`,
  `amp_dtype`) in `CausalLMTrainer`.
- **Distributed data parallel** (`zenith.distributed`): `torchrun`-native helpers,
  wired into the trainer (sampler + main-process-only logging/checkpointing).
- LoRA-aware checkpoints: the `LoraConfig` is recorded and re-injected on load, so
  a LoRA checkpoint round-trips into a `Generator`.
- `peft` config group (`none`/`lora`) and `training.{grad_accum_steps,amp,amp_dtype}`.

### Notes
- All new features default off; the v0.2.0 single-device training path is unchanged.
- QLoRA (4-bit) and FSDP are deferred (GPU-only, not CI-testable). See ADR-0004.

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
