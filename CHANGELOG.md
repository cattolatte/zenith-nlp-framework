# Changelog

All notable changes to Zenith are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.9.2] — Benchmark figures

### Added
- `scripts/plot_benchmarks.py` — renders the benchmark figures (scaling curve,
  convergence curves, architecture ablation, text8 curve) into `assets/` from the
  measured numbers, on a white card so they read in GitHub light and dark themes.
- Figures embedded in the README (scaling hero) and BENCHMARKS (all four).

_Docs, assets, and tooling only — no library changes._

## [0.9.1] — Scaling study & a harder benchmark

### Added
- `scripts/scaling_study.py` — trains a sweep of Llama-style decoders (0.6M → 10.7M
  params, same recipe) and reports held-out bits/char.
- `scripts/download_text8.py` — fetches a subset of the standard **text8** char-LM
  corpus for an out-of-domain second benchmark.
- BENCHMARKS: a **bits/char-vs-model-size scaling curve** (diminishing returns into
  the data floor) and a **text8-subset result** (1.78 bpc), both with honest caveats
  (subset ≠ full text8; alphabet size affects bits/char).
- Design docs for phases 8 & 9, and ADR-0009 (modern architecture).

_No library API changes — benchmarks, scripts, and docs only._

## [0.9.0] — Modern (Llama-style) architecture

### Added
- **Configurable, from-scratch modern architecture** in `DecoderLM` /
  `DecoderConfig`:
  - `norm`: `rmsnorm` (default) or `layernorm` — from-scratch RMSNorm.
  - `positional`: `rope` (default) or `learned` — from-scratch rotary embeddings,
    correct under the KV-cache (verified by the cache-equivalence test).
  - `ffn`: `swiglu` (default) or `gelu` — from-scratch SwiGLU.
  Selectable via `model.{norm,positional,ffn}` in the Hydra config.

### Changed
- **Default architecture is now Llama-style** (RoPE + RMSNorm + SwiGLU). On
  tiny-shakespeare it converges ~2× faster than the GPT-2-style recipe and reaches
  **2.08 bits/char** (vs 2.11) at fewer params — see `BENCHMARKS.md` for the
  architecture ablation.

### Compatibility
- Pre-v0.9 (GPT-2-style) checkpoints still load: the loader defaults the new fields
  to `layernorm`/`learned`/`gelu` when they're absent.

## [0.8.0] — Interactive console & docs polish

### Added
- **`zenith console`** — an interactive REPL (`zenith.console`) in the spirit of the
  Polaris console: a colored ASCII banner, and a generation-first prompt where you
  just type text to continue it. Commands: `load`, `set <param> <value>` (temperature
  / top_k / top_p / repetition_penalty / num_beams / max_new_tokens), `show`,
  `generate`, `info`, `help`, `exit`. `load` a checkpoint with `-m` on launch.

### Changed
- README / module docs polished and brought fully up to date (console, evaluation,
  the nanoGPT-matching benchmark, corrected architecture tree).

## [0.7.2] — GPT-2 initialization & a nanoGPT-matching result

### Added
- **GPT-2-style weight initialization** in `DecoderLM`: N(0, 0.02) weights with
  residual output projections scaled by `1/sqrt(2 · n_layers)`. This substantially
  improves convergence (roughly halves epochs-to-converge) and is what lets a
  10.8M model reach **2.11 bits/char** on tiny-shakespeare — matching the nanoGPT
  baseline. `BENCHMARKS.md` updated with the headline result, ablation, curve, and
  sample.
- `[tool.mypy]` config that ignores missing stubs for optional, un-stubbed extras
  (`polaris`, `mlflow`, `hydra`, `omegaconf`, `uvicorn`).

### Changed
- README: refreshed the (stale) project-status section — the generative stack is
  complete and released — and added the benchmark result.

## [0.7.1] — Training efficiency, benchmarks & project docs

### Added
- `CausalLMDataset(..., stride=...)` — strided / non-overlapping training windows
  (default `stride=1` unchanged). `stride=block_size` gives far fewer passes, which
  makes training on real corpora practical; exposed as `data.stride`.
- **Real benchmarks** in `BENCHMARKS.md` — tiny-shakespeare, byte-level: 6.66
  held-out perplexity / 2.74 bits-per-char with a 10.8M model in ~15 min on Apple
  MPS, with the training curve and a sample.
- Community health files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`,
  and issue / pull-request templates.

## [0.7.0] — Polaris interop (sibling showcase)

### Added
- `zenith.interop.PolarisTokenizer` — adapts a Polaris tokenizer to Zenith's
  tokenizer interface, so a Zenith decoder can train and generate over a Polaris
  (encoder-side) vocabulary. `polaris-nlp` stays an optional extra, imported
  lazily.
- `examples/encode_and_generate.py` — the sibling demo (Polaris tokenizes, Zenith
  generates).
- Interop tests (skipped without the `[polaris]` extra; verified against Polaris
  1.1.0).

## [0.6.2] — UX

### Changed
- `CausalLMDataset` gives an actionable error when a corpus encodes to fewer
  tokens than `block_size`, pointing at corpus size / `block_size` /
  `tokenizer.vocab_size` (BPE can over-compress a small corpus).
- The trainer warns when `amp=true` on a device where autocast is a no-op
  (e.g. Apple MPS), instead of silently ignoring it.

## [0.6.1] — Fixes

### Fixed
- Distributed runs without CUDA now fall back to CPU (gloo) instead of MPS, since
  PyTorch collective ops aren't implemented for Apple MPS — `torchrun` no longer
  crashes on a Mac (it runs on CPU; DDP is really for multi-GPU CUDA).
- Use the modern `torch.amp.GradScaler` API, silencing the deprecation warning on
  recent PyTorch.

## [0.6.0] — Evaluation, a learned tokenizer & release-readiness

The capstone: measurable, better on real text, ready to publish.

### Added
- `BPETokenizer` — from-scratch byte-level BPE (trainable, lossless), a drop-in
  alternative to `ByteTokenizer`; checkpoints serialize it, and streaming
  generalizes to any tokenizer via `token_bytes`.
- `zenith.evaluation`: `evaluate()` / `perplexity()` (held-out, token-level).
- CLI: `tokenizer` config group (`byte`/`bpe`) and `zenith eval -m <ckpt> -c <corpus>`.
- `BENCHMARKS.md` (methodology + reproduction), `docs/modules.md`,
  `scripts/download_corpus.py`.
- `.github/workflows/release.yml` — PyPI trusted publishing on tag.

### Notes
- Benchmark numbers are intentionally left to a real run (no GPU/torch runtime in
  the dev environment) — see ADR-0007.

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
