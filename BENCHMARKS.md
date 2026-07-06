# Benchmarks

Zenith is a small, from-scratch library, not a scale competitor. These are honest,
reproducible numbers on a modest, laptop-scale setup (Apple M-series, MPS) — the
same spirit as the sibling Polaris project: measure it, explain it, don't
cherry-pick.

## Task & data

- **Task:** character/byte-level language modelling.
- **Corpus:** tiny-shakespeare (1,115,394 characters), 90/10 train/val split by
  position. Fetch it with `python scripts/download_corpus.py`.
- **Metric:** held-out perplexity, and **bits per character (bpc)** — the
  tokenizer-independent metric (so byte-level and BPE are comparable, since raw
  per-token perplexity is not).

## Model & training

| | |
|---|---|
| Architecture | decoder-only transformer, 6 layers, 384-dim, 6 heads, ff 1536 |
| Parameters | **10.8M** |
| Context (`block_size`) | 256 |
| Optimizer | AdamW, lr 1e-3, 100-step warmup → cosine decay, dropout 0.2 |
| Hardware / time | Apple M-series (MPS), ~15 min (byte), ~1,800 steps |

## Results

| Tokenizer | Vocab | Val perplexity (per token) | **Bits/char** | Training |
| :-------- | ----: | -------------------------: | ------------: | :------- |
| byte      |   259 |                       6.66 |     **2.74**  | 15 epochs (converged) |
| bpe       |  1024 |                       92.8 |     **2.65**  | ~9 epochs (undertrained) |

> **Only bits/char is comparable across tokenizers.** Per-token perplexity looks
> wildly different — byte scores each of 259 byte ids, BPE each of 1024 denser
> subwords — but bits/char normalizes to the underlying text. Notably, **BPE already
> edges out byte on bits/char despite far less training**: subwords pack more signal
> per step. The BPE run was cut short because Zenith's from-scratch BPE tokenizer is
> `O(merges × corpus)` and slow to train on a 1 MB corpus (a known limitation;
> vectorized BPE is future work). A full BPE run would widen the gap.

**Training curve (byte-level, val perplexity):**

```
epoch   1:  58.6      epoch  6:  11.96     epoch 11:   6.88
epoch   2:  20.76     epoch  7:  10.27     epoch 12:   6.77
epoch   3:  20.77     epoch  8:   8.24     epoch 13:   6.73
epoch   4:  14.25     epoch  9:   7.54     epoch 14:   6.69
epoch   5:  13.15     epoch 10:   7.02     epoch 15:   6.66
```

**Sample** (byte model, prompt `ROMEO:`, temperature 0.7):

```
ROMEO:
And all here is the wath the cause to word.

KING RICHARD II:
I well, the stuness will the come the so have will a she
Well from he shor quest my lose such well hen not the reed
And my sently didies in of the would not the would whome,
```

From raw bytes, the model has learned the *structure* of a play — speaker names in
caps, the `:` dialogue format, line breaks, and English-like morphology — even
though many words are invented. That is what ~15 minutes of laptop training buys at
this scale.

## Reproduce

```bash
python scripts/download_corpus.py
python -m zenith.cli.train \
    data.corpus_path=data/tiny_shakespeare.txt data.stride=128 \
    model.block_size=256 model.embed_dim=384 model.num_layers=6 \
    model.num_heads=6 model.ff_dim=1536 model.dropout=0.2 \
    training.epochs=15 training.batch_size=64 training.learning_rate=1e-3
zenith eval -m zenith-lm.pt -c data/tiny_shakespeare.txt
```

## Honest notes

- **2.74 bpc is decent, not state-of-the-art.** Well-trained char models reach
  ~1.4–1.5 bpc; the gap is training budget, not correctness — the run used ~1,800
  steps and the learning rate had fully decayed. More steps, a bigger model, or a
  smaller `stride` (more windows) lower it further.
- **`stride`** controls training cost: `stride=1` (every window) is thorough but
  makes an epoch huge; `stride=block_size` (non-overlapping) is far faster. The run
  above used `stride=128`.
- MPS is the bottleneck here (~2 steps/s at this size); a CUDA GPU is much faster.
