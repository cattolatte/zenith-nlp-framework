# Phase 4 — Tracking & reproducibility

## Goal
Make runs legible and reproducible: log what the model actually generates, record
each run self-containedly, and support deterministic training.

## Scope (what ships this phase)
- `experiments/` — `capture_environment()` (versions / seed / git commit) and
  `record_run()` (writes `config.json`, `metrics.json`, `samples.txt`,
  `environment.json`, `run.json` to a run directory).
- `utils.set_deterministic()` — seed + deterministic algorithms (`warn_only`).
- `CausalLMTrainer`:
  - Logs the **full run config** to MLflow (flattened).
  - Collects the per-epoch **text sample** and logs each as an MLflow artifact
    (`samples/epoch_N.txt`) — generation is the thing worth watching.
  - Logs the best **checkpoint** as an artifact.
  - Optionally writes an on-disk **run record** (`record_dir`).
  - Optional **deterministic** mode.
  - Accepts a `run_config` (the resolved Hydra config) for logging/recording.
- Config: `training.{log_samples,deterministic,record_dir}`; the CLI passes the
  resolved config through as `run_config`.
- Tests: environment capture; run-record files + contents; deterministic seeding;
  the trainer writes a run record with metrics + samples + environment.

## Key decisions (ADR-0005)
- **On-disk run records complement MLflow**, not replace it: plain JSON/text that
  survives without a server and can be diffed (mirrors Polaris' `experiments`).
- **Determinism is a goal, not a gate** (`warn_only=True`) so a run isn't killed by
  one non-deterministic op.
- Sample logging is first-class: for a generative model, *what it writes* is the
  most informative signal, so it's logged every epoch.

## Out of scope (later)
- Sweep-result aggregation / best-run selection across an MLflow experiment
  (needs live runs to test meaningfully); MLflow Model Registry.

## Definition of done
Runs log config + metrics + samples + checkpoint; a run record is written and
tested; deterministic mode works; gates green; tag `v0.4.0`.
