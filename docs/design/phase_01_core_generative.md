# Phase 1 — Core generative engine

## Goal
A runnable, testable vertical slice: **train a decoder LM on text and generate
from it**, with zero required external NLP dependency.

## Scope (what ships this phase)
- `tokenizers/` — `ByteTokenizer` (byte-level, dependency-free). ADR-0002.
- `models/` — `DecoderLM` + `DecoderConfig`: from-scratch decoder-only transformer
  (causal self-attention, pre-norm blocks, tied embeddings).
- `data/` — `CausalLMDataset` (next-token blocks) + corpus/split helpers.
- `generation/` — `Generator`: greedy + temperature sampling.
- `training/` — `CausalLMTrainer` + `TrainingConfig`: single-device next-token
  training with warmup/cosine schedule, grad clipping, best-checkpoint saving,
  in-training text samples, optional MLflow logging.
- `checkpoint.py` — self-describing save/load + `load_pretrained` → `Generator`.
- `cli/` — Hydra `train` entrypoint; `zenith` CLI (`info`, `generate`).
- `configs/` — Hydra tree (model / training / data / tracking).
- `data/tiny_corpus.txt` — small original corpus for the demo slice.
- Tests for every module, incl. a learning-behaviour test (loss falls on a tiny
  fixture; checkpoint round-trips into a working generator).

## Explicitly out of scope (later phases)
- top-k / nucleus / beam search, KV-cache (Phase 2).
- LoRA/QLoRA, DDP, mixed precision (Phase 3).
- Sample-artifact logging, sweeps tooling (Phase 4).
- Generation serving API, streaming (Phase 5).
- Learned BPE tokenizer, Polaris tokenizer interop extra.

## Key decisions
- Decoder is written fresh (not lifted from `_incubating/`) to a clean, typed,
  tested standard in the house style.
- `polaris-nlp` demoted to an optional extra; Phase 1 has no Polaris dependency.

## Definition of done
`python -m zenith.cli.train` trains on the bundled corpus and prints improving
loss + text samples; `zenith generate -m <ckpt> "..."` produces text; all gates
(`ruff`, `black`, `pytest`) pass. Tag `v0.1.0`.
