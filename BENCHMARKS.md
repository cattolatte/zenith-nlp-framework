# Benchmarks

Zenith is a small, from-scratch library, not a scale competitor. The value of this
document is a **reproducible methodology** and **honest numbers** on a modest,
laptop-scale setup — the same spirit as the sibling Polaris project: measure it,
explain it, don't cherry-pick.

The metric is **held-out perplexity** (`exp` of the mean next-token
cross-entropy), plus a qualitative generated sample.

## Setup

- **Task**: character/byte-level language modelling.
- **Corpus**: a single UTF-8 text file (`data.corpus_path`); 90/10 train/val split
  by position. The bundled `data/tiny_corpus.txt` is a smoke-test corpus; for a
  real run use a larger one (see `scripts/download_corpus.py`).
- **Model**: `configs/model/decoder.yaml` (defaults: 4 layers, 256-dim, 4 heads,
  128 context) — adjust to taste.
- **Tokenizer**: byte-level (`tokenizer=byte`) or from-scratch BPE
  (`tokenizer=bpe`).
- **Seed**: 0 (`training.deterministic=true` for a fully deterministic run).

## Reproduce

```bash
# 1. Get a real corpus (public-domain tiny-shakespeare)
python scripts/download_corpus.py            # writes data/tiny_shakespeare.txt

# 2. Train (byte-level, or add tokenizer=bpe)
python -m zenith.cli.train \
    data.corpus_path=data/tiny_shakespeare.txt \
    training.epochs=20 training.deterministic=true

# 3. Report held-out perplexity on a file
zenith eval -m zenith-lm.pt -c data/tiny_shakespeare.txt
```

Each run also writes a self-contained record (config / metrics / samples /
environment) under the Hydra run directory (`training.record_dir`).

## Results

> These rows are intentionally **left to be filled from an actual run** — this repo
> is developed in an environment without a GPU/torch runtime, and publishing
> invented numbers would defeat the purpose. Run the commands above and record what
> you get; the methodology is fixed so the comparison is fair.

| Tokenizer | Vocab | Params | Epochs | Val perplexity |
| :-------- | ----: | -----: | -----: | -------------: |
| byte      |   259 |      — |     20 |            _—_ |
| bpe       |  1024 |      — |     20 |            _—_ |

**What to expect / how to read it:** byte-level has the smallest vocab and longest
sequences; BPE shortens sequences and usually lowers perplexity at equal compute,
but adds a training step. At this scale both are dominated by corpus size and model
capacity — the honest takeaway is usually "more data and a bigger model move this
far more than tokenizer choice," which is itself the point worth demonstrating.
