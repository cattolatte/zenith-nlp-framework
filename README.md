<div align="center">

# Zenith

### A from-scratch generative NLP library — decoder-only language models & text generation

[![Release](https://img.shields.io/github/v/release/cattolatte/zenith-nlp-framework?label=release&color=6E56CF)](https://github.com/cattolatte/zenith-nlp-framework/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-blue.svg)](#project-status)
[![tiny-shakespeare](https://img.shields.io/badge/tiny--shakespeare-2.08%20bits%2Fchar-6E56CF.svg)](BENCHMARKS.md)

</div>

---

Zenith is a clean, from-scratch library for **generative NLP**: decoder-only
transformer language models, causal-LM training, and **text generation** — built
on PyTorch tensor primitives. The architecture is hand-written and modern
(Llama-style: **RoPE, RMSNorm, SwiGLU**, weight-tied embeddings), readable end to
end; PyTorch supplies only autograd, containers and optimizers.

> **It works:** a 10.7M from-scratch Llama-style decoder trained in ~10 min on a
> MacBook (MPS) reaches **2.08 bits/char** on tiny-shakespeare — matching the
> well-known [nanoGPT](https://github.com/karpathy/nanoGPT) baseline. See
> [BENCHMARKS.md](BENCHMARKS.md).

Zenith is a standalone project. It is also the **generative counterpart** to
[Polaris](https://github.com/cattolatte/Polaris), a from-scratch engine focused on
*understanding* text (transformer encoders, classification). Polaris encodes;
Zenith generates. The two are complementary but independent — with the optional `[polaris]` extra,
`zenith.interop.PolarisTokenizer` lets a Zenith decoder generate over a Polaris
vocabulary (see `examples/encode_and_generate.py`), but Zenith ships its own
tokenizer and does not depend on Polaris.

## What's here

- **Decoder-only transformer** (`DecoderLM`) — configurable **Llama-style** (RoPE,
  RMSNorm, SwiGLU) or GPT-2-style (LayerNorm, learned pos, GELU), from scratch, with
  an optional fused **SDPA** attention path (faster, numerically equivalent).
- **Tokenizers** — a dependency-free byte-level tokenizer (`ByteTokenizer`) and a
  from-scratch, trainable byte-level BPE (`BPETokenizer`), both lossless.
- **Text generation** (`Generator`) — greedy, temperature, top-k, nucleus (top-p),
  repetition penalty, and beam search, with a KV-cache and streaming; plus
  **greedy-exact speculative decoding** (a small draft model, output identical to
  greedy — 3×+ fewer target forward passes, see [BENCHMARKS.md](BENCHMARKS.md)).
- **Causal-LM training** (`CausalLMTrainer`) — warmup/cosine schedule, gradient
  clipping, best-checkpoint saving, per-epoch samples, MLflow tracking, on-disk
  run records, and a deterministic mode.
- **Efficient fine-tuning & scaling** — LoRA adapters (`zenith.peft`), gradient
  accumulation, mixed precision (AMP), and `torchrun`-native distributed (DDP)
  training — all opt-in.
- **Instruction tuning** (`zenith.instruct`) — a chat template + supervised
  fine-tuning with response-only loss masking turns a base model into a mini chat
  model (`zenith chat --instruct`). See the [model card](MODEL_CARD.md).
- **Evaluation** — held-out `perplexity` / `evaluate`, and a `zenith eval` command.
- **int8 quantization** (`zenith.quantize`) — weight-only int8 for ~4× smaller
  inference weights, output within quantization error (`zenith generate --int8`).
- **Serving & CLI** — a FastAPI service (`POST /generate`, SSE
  `POST /generate/stream`), plus `zenith serve`, a streaming `zenith chat`, and an
  interactive `zenith console` (a REPL with a banner and tunable decoding).
- **Hydra-configured** runs and hyperparameter sweeps.

See [BENCHMARKS.md](BENCHMARKS.md) for the evaluation methodology and
[docs/modules.md](docs/modules.md) for a module overview. On the roadmap:
QLoRA/FSDP for larger-scale training, and sweep-result aggregation.

## Benchmarks

A measured scaling study — same Llama-style recipe, only model size changes.
Bits/char falls with capacity and flattens into tiny-shakespeare's data floor:

![Scaling on tiny-shakespeare: bits per char versus model size, falling from 2.329 at 0.6M params to 2.066 at 10.7M and flattening into a data floor near 2.07](assets/scaling_curve.png)

Full write-up — architecture ablation, convergence curves, and a harder text8
benchmark — in [BENCHMARKS.md](BENCHMARKS.md). Figures regenerate from the measured
numbers via [`scripts/plot_benchmarks.py`](scripts/plot_benchmarks.py).

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
zenith chat -m zenith-lm.pt          # quick streaming REPL
zenith console -m zenith-lm.pt       # full REPL: load/set/show/generate + banner
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
├── tokenizers/      # byte-level + from-scratch BPE tokenizers
├── data/            # causal-LM datasets & corpus helpers
├── generation/      # sampling / decoding (+ streaming)
├── training/        # causal-LM training loop
├── evaluation/      # held-out loss & perplexity
├── peft/            # LoRA adapters
├── distributed/     # DDP helpers
├── tracking/        # optional MLflow experiment tracking
├── experiments/     # environment capture & on-disk run records
├── serving/         # FastAPI generation service (+ SSE streaming)
├── console/         # interactive `zenith console` REPL
├── interop/         # optional Polaris tokenizer adapter (sibling bridge)
├── cli/             # Hydra train entrypoint + `zenith` CLI (serve, chat, console, …)
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
