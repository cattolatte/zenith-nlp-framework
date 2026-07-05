<div align="center">

# Zenith

### A from-scratch generative NLP library — decoder-only language models & text generation

[![Release](https://img.shields.io/github/v/release/cattolatte/zenith-nlp-framework?label=release&color=6E56CF)](https://github.com/cattolatte/zenith-nlp-framework/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active%20development-orange.svg)](#project-status)

</div>

---

Zenith is a clean, from-scratch library for **generative NLP**: decoder-only
transformer language models, causal-LM training, and **text generation** — built
on PyTorch tensor primitives. The architecture is hand-written (causal
self-attention, pre-norm blocks, weight-tied embeddings) and readable end to end;
PyTorch supplies only autograd, containers and optimizers.

Zenith is a standalone project. It is also the **generative counterpart** to
[Polaris](https://github.com/cattolatte/Polaris), a from-scratch engine focused on
*understanding* text (transformer encoders, classification). Polaris encodes;
Zenith generates. The two are complementary but independent — Zenith can
*optionally* reuse Polaris' tokenizers (`pip install zenith-nlp[polaris]`), but it
ships its own and does not depend on Polaris.

## What's here

- **Decoder-only transformer** (`DecoderLM`) — causal self-attention, pre-norm
  blocks, tied embeddings, written from scratch.
- **Byte-level tokenizer** (`ByteTokenizer`) — dependency-free, lossless on any
  text.
- **Text generation** (`Generator`) — greedy, temperature, top-k, nucleus (top-p),
  repetition penalty, and beam search, with a KV-cache for efficient decoding.
- **Causal-LM training** (`CausalLMTrainer`) — warmup/cosine schedule, gradient
  clipping, best-checkpoint saving, per-epoch samples, MLflow tracking, on-disk
  run records, and a deterministic mode.
- **Efficient fine-tuning & scaling** — LoRA adapters (`zenith.peft`), gradient
  accumulation, mixed precision (AMP), and `torchrun`-native distributed (DDP)
  training — all opt-in.
- **Hydra-configured** runs and sweeps; a small `zenith` CLI.

On the roadmap: a streaming generation service, plus QLoRA and FSDP for
larger-scale training.

## Install

```bash
git clone https://github.com/cattolatte/zenith-nlp-framework.git
cd zenith-nlp-framework
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"      # torch, hydra, omegaconf, mlflow, typer
```

## Usage

Train a language model on the bundled corpus (or point `data.corpus_path` at your
own text):

```bash
python -m zenith.cli.train                                  # defaults
python -m zenith.cli.train training.epochs=50 model.embed_dim=384
python -m zenith.cli.train peft=lora                        # LoRA fine-tuning
python -m zenith.cli.train training.amp=true training.grad_accum_steps=4
python -m zenith.cli.train -m training.learning_rate=1e-3,3e-4,1e-4   # sweep
torchrun --nproc_per_node=4 -m zenith.cli.train             # multi-GPU (DDP)
```

Generate text from a trained checkpoint:

```python
from zenith import load_pretrained

gen = load_pretrained("zenith-lm.pt")
print(gen.generate("Once upon a time", max_new_tokens=200, temperature=0.8))
```

Or from the CLI:

```bash
zenith generate -m zenith-lm.pt "Once upon a time" --temperature 0.8
```

## Architecture

```text
src/zenith/
├── models/          # decoder-only transformer (from scratch)
├── tokenizers/      # byte-level tokenizer
├── data/            # causal-LM datasets & corpus helpers
├── generation/      # sampling / decoding
├── training/        # causal-LM training loop
├── tracking/        # optional MLflow experiment tracking
├── cli/             # Hydra train entrypoint + `zenith` CLI
└── checkpoint.py    # self-describing save / load
```

## Project status

Zenith is under **active development**, mid-way through a redesign from an early,
general NLP framework into the focused generative library above. Phase 1 (the
generative core: model, tokenizer, training, generation) is in place; decoding
strategies, PEFT, distributed training and serving follow. Interfaces may change
until the first tagged release.

## License

MIT — see [LICENSE](LICENSE).

<div align="center">
by K Satya Sai Nischal
</div>
