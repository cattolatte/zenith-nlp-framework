# Benchmarks

Honest, reproducible numbers on a modest, laptop-scale setup (Apple M-series, MPS).
The headline: **a from-scratch decoder that matches the well-known
[nanoGPT](https://github.com/karpathy/nanoGPT) baseline on tiny-shakespeare, trained
in ~18 minutes on a MacBook.**

## Headline result

| | |
|---|---|
| Task | character/byte-level language modelling, tiny-shakespeare (1.1M chars) |
| Model | decoder-only, 6 layers, 384-dim, 6 heads, **10.8M params**, block 256 |
| Held-out loss | **1.46** |
| Perplexity | **4.32** |
| **Bits/char** | **2.11** |
| Reference (nanoGPT) | ~1.47 loss / ~2.1 bits/char |

**Bits/char** (nats/char ÷ ln 2) is the metric that matters — it's tokenizer- and
model-independent, and it puts Zenith right on the nanoGPT reference.

## What moved the needle (ablation)

Getting from "mid" to "matches nanoGPT" was three concrete fixes, not brute force:

| Configuration | Bits/char |
| :------------ | --------: |
| block 256, 15 epochs, default init *(undertrained: LR decayed to 0 too early)* | 2.74 |
| block 128, converged, LR schedule matched to the run | 2.45 |
| block 256 + **GPT-2 initialization**, converged | **2.11** |

The biggest single lever was **initialization**: N(0, 0.02) weights with residual
projections scaled by `1/sqrt(2·n_layers)`. It dropped the *entire* training curve
and cut epochs-to-converge roughly in half.

## Training curve (headline run, val loss)

```
epoch  1: 2.463    epoch  8: 1.550    epoch 15: 1.475
epoch  2: 2.143    epoch  9: 1.522    epoch 17: 1.472  <- best (saved)
epoch  4: 1.755    epoch 11: 1.494    epoch 20: 1.486
epoch  6: 1.618    epoch 13: 1.480    epoch 26: 1.547  (overfitting)
```

Val bottoms out around epoch 15–17, then rises as the model overfits the small
corpus; the trainer keeps the best checkpoint automatically.

## Sample (headline model, prompt `KING RICHARD III:`, temperature 0.7)

```
KING RICHARD III:
So, then he may, and so so much a deed.

DUKE OF AUMERLE:
How now, my lord! disposition'd in the oracle!
So stands shall I show the morning tears;
And then see my descent report and my teeth.
```

Speaker names, dialogue format, grammar, and Shakespearean cadence — learned from
raw bytes.

## Tokenizer comparison (bits/char, the fair metric)

| Tokenizer | Vocab | Bits/char | Notes |
| :-------- | ----: | --------: | :---- |
| byte      |   259 |  **2.11** | headline (converged) |
| bpe       |  1024 |     ~2.65 | undertrained — Zenith's from-scratch BPE tokenizer is `O(merges × corpus)` and slow on 1 MB; vectorized BPE is future work |

## Reproduce

```bash
python scripts/download_corpus.py
python -m zenith.cli.train \
    data.corpus_path=data/tiny_shakespeare.txt data.stride=128 \
    model.block_size=256 model.embed_dim=384 model.num_layers=6 \
    model.num_heads=6 model.ff_dim=1536 model.dropout=0.2 \
    training.epochs=18 training.batch_size=64 training.learning_rate=1e-3
zenith eval -m zenith-lm.pt -c data/tiny_shakespeare.txt
zenith generate -m zenith-lm.pt "KING RICHARD III:" -n 320 -t 0.7
```

## Honest notes

- **2.11 bits/char matches, and marginally beats, the nanoGPT reference** — genuinely
  good for a from-scratch model at this scale. A bigger model or more data would go
  lower still (well-tuned char models on larger corpora reach ~1.0–1.5 bpc).
- The model **overfits after ~17 epochs** on this tiny corpus; best-checkpoint
  saving handles it. Early stopping is a natural future addition.
- MPS is the bottleneck (~2 steps/s at this size); a CUDA GPU is far faster.
