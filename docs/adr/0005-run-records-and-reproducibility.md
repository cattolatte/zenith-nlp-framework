# ADR-0005: On-disk run records complement MLflow; determinism is best-effort

## Status
Accepted.

## Context
Phase 4 adds experiment tracking depth and reproducibility. Two questions: where
does run metadata live, and how strict is "deterministic"?

## Decision
1. **Records live in two places, by design.** Live metrics/params/artifacts go to
   MLflow (optional, server-backed). In parallel, each run can write a
   self-contained, human-readable folder (`config/metrics/samples/environment/run`
   JSON+text) via `zenith.experiments.record_run`. The on-disk record survives
   without a tracking server and can be diffed — the same philosophy as Polaris'
   `experiments` module.
2. **Generated samples are logged every epoch.** For a generative model, the most
   informative signal is what it produces; samples are collected and logged as
   MLflow artifacts and written into the run record.
3. **Determinism is best-effort** (`torch.use_deterministic_algorithms(True,
   warn_only=True)`): a run should not die because a single op lacks a
   deterministic kernel. Full seeding is always applied.

## Consequences
- Runs are reproducible-by-record without depending on an external service.
- The trainer takes an optional `run_config` (the resolved Hydra config) so the
  exact configuration is logged and recorded.
- Sweep-result aggregation and the MLflow Model Registry are deferred (they need
  live runs to be worth testing).
