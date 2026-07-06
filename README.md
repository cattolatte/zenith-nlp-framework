<div align="center">

# Zenith

### A from-scratch generative NLP library — decoder-only language models & text generation

[![Release](https://img.shields.io/github/v/release/cattolatte/zenith-nlp-framework?label=release&color=6E56CF)](https://github.com/cattolatte/zenith-nlp-framework/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-blue.svg)](#project-status)
[![tiny-shakespeare](https://img.shields.io/badge/tiny--shakespeare-2.11%20bits%2Fchar-6E56CF.svg)](BENCHMARKS.md)

</div>

---

Zenith is a clean, from-scratch library for **generative NLP**: decoder-only
transformer language models, causal-LM training, and **text generation** — built
on PyTorch tensor primitives. The architecture is hand-written (causal
self-attention, pre-norm blocks, weight-tied embeddings) and readable end to end;
PyTorch supplies only autograd, containers and optimizers.

> **It works:** a 10.8M from-scratch decoder trained in ~18 min on a MacBook (MPS)
> reaches **2.11 bits/char** on tiny-shakespeare — matching the well-known
> [nanoGPT](https://github.com/karpathy/nanoGPT) baseline. See [BENCHMARKS.md](BENCHMARKS.md).

Zenith is a standalone project. It is also the **generative counterpart** to
[Polaris](https://github.com/cattolatte/Polaris), a from-scratch engine focused on
*understanding* text (transformer encoders, classification). Polaris encodes;
Zenith generates. The two are complementary but independent — with the optional `[polaris]` extra,
`zenith.interop.PolarisTokenizer` lets a Zenith decoder generate over a Polaris
vocabulary (see `examples/encode_and_generate.py`), but Zenith ships its own
tokenizer and does not depend on Polaris.

## What's here

- **Decoder-only transformer** (`DecoderLM`) — causal self-attention, pre-norm
  blocks, tied embeddings, written from scratch.
- **Tokenizers** — a dependency-free byte-level tokenizer (`ByteTokenizer`) and a
  from-scratch, trainable byte-level BPE (`BPETokenizer`), both lossless.
- **Text generation** (`Generator`) — greedy, temperature, top-k, nucleus (top-p),
  repetition penalty, and beam search, with a KV-cache and streaming.
- **Causal-LM training** (`CausalLMTrainer`) — warmup/cosine schedule, gradient
  clipping, best-checkpoint saving, per-epoch samples, MLflow tracking, on-disk
  run records, and a deterministic mode.
- **Efficient fine-tuning & scaling** — LoRA adapters (`zenith.peft`), gradient
  accumulation, mixed precision (AMP), and `torchrun`-native distributed (DDP)
  training — all opt-in.
- **Evaluation** — held-out `perplexity` / `evaluate`, and a `zenith eval` command.
- **Serving** — a FastAPI service (`POST /generate`, SSE `POST /generate/stream`),
  a `zenith serve` command, and an interactive `zenith chat` REPL.
- **Hydra-configured** runs and sweeps; a small `zenith` CLI.

See [BENCHMARKS.md](BENCHMARKS.md) for the evaluation methodology and
[docs/modules.md](docs/modules.md) for a module overview. On the roadmap:
QLoRA/FSDP for larger-scale training, and sweep-result aggregation.

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
python -m zenith.cli.train tokenizer=bpe                    # from-scratch BPE
python -m zenith.cli.train peft=lora                        # LoRA fine-tuning
python -m zenith.cli.train training.amp=true training.grad_accum_steps=4
python -m zenith.cli.train -m training.learning_rate=1e-3,3e-4,1e-4   # sweep
torchrun --nproc_per_node=4 -m zenith.cli.train             # multi-GPU (DDP)
```

Evaluate held-out perplexity:

```bash
zenith eval -m zenith-lm.pt -c data/tiny_corpus.txt
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
zenith chat -m zenith-lm.pt          # interactive REPL, streams as it generates
```

Serve it over HTTP (blocking + streaming):

```bash
zenith serve -m zenith-lm.pt         # POST /generate, POST /generate/stream (SSE)
curl -s localhost:8000/generate -d '{"prompt":"Once","max_new_tokens":100}'
```

## Architecture

```text
src/zenith/
├── models/          # decoder-only transformer (from scratch)
├── tokenizers/      # byte-level tokenizer
├── data/            # causal-LM datasets & corpus helpers
├── generation/      # sampling / decoding (+ streaming)
├── training/        # causal-LM training loop
├── evaluation/      # held-out loss & perplexity
├── peft/            # LoRA adapters
├── distributed/     # DDP helpers
├── tracking/        # optional MLflow experiment tracking
├── experiments/     # environment capture & on-disk run records
├── serving/         # FastAPI generation service (+ SSE streaming)
├── interop/         # optional Polaris tokenizer adapter (sibling bridge)
├── cli/             # Hydra train entrypoint + `zenith` CLI (serve, chat, …)
└── checkpoint.py    # self-describing save / load
```

## Project status

The generative stack is **complete and released** (see the
[releases](https://github.com/cattolatte/zenith-nlp-framework/releases)): model,
decoding, training, scaling (LoRA / AMP / DDP), tracking, serving, evaluation, a
from-scratch BPE tokenizer, and optional Polaris interop — all covered by an
offline test suite and CI (Python 3.10–3.12). It trains real models and matches the
nanoGPT baseline on tiny-shakespeare (see [BENCHMARKS.md](BENCHMARKS.md)).

Still pre-1.0, so interfaces may change. Deferred (optional) work: QLoRA, FSDP,
vectorized BPE, sweep-result aggregation.

## License

MIT — see [LICENSE](LICENSE).

<div align="center">
by K Satya Sai Nischal
</div>
