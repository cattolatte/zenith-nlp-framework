# Benchmarks

Honest, reproducible numbers on a modest, laptop-scale setup (Apple M-series, MPS).
The headline: **a from-scratch, Llama-style decoder (RoPE + RMSNorm + SwiGLU) that
matches the well-known [nanoGPT](https://github.com/karpathy/nanoGPT) baseline on
tiny-shakespeare, trained in ~10 minutes on a MacBook.**

## Headline result

| | |
|---|---|
| Task | character/byte-level language modelling, tiny-shakespeare (1.1M chars) |
| Model | Llama-style (RoPE, RMSNorm, SwiGLU), 6 layers, 384-dim, 6 heads, **10.7M params**, block 256 |
| Held-out loss | **1.44** |
| Perplexity | **4.22** |
| **Bits/char** | **2.08** |
| Reference (nanoGPT) | ~1.47 loss / ~2.1 bits/char |

**Bits/char** (nats/char ÷ ln 2) is the metric that matters — tokenizer- and
model-independent — and it puts Zenith right on the nanoGPT reference.

## Architecture ablation (GPT-2-style vs Llama-style)

Same recipe, same ~10.7M params, same data — only the architecture differs:

| Architecture | Params | Bits/char | Converged at |
| :----------- | -----: | --------: | :----------- |
| GPT-2-style — LayerNorm, learned pos, GELU | 10.8M | 2.11 | epoch 17 |
| **Llama-style — RMSNorm, RoPE, SwiGLU** | 10.7M | **2.08** | **epoch 10** |

The modern architecture is **slightly better *and* converges roughly twice as
fast** (best val at epoch 10 vs 17). The honest caveat: both land near ~2.1 bpc —
at this scale the floor is set by *data and model size*, not the architecture. The
architecture buys **convergence speed** (and a little quality); breaking the ~2.1
ceiling needs a bigger model or more data.

## What got us here (in order of impact)

1. **GPT-2 initialization** — N(0, 0.02) weights, residual projections scaled by
   `1/sqrt(2·n_layers)`. Roughly halved epochs-to-converge; the single biggest lever.
2. **Modern architecture** — RoPE + RMSNorm + SwiGLU: faster convergence, slightly
   lower loss, fewer params.
3. **Training to convergence** with an LR schedule matched to the run (and
   best-checkpoint saving, since it overfits this tiny corpus after ~10 epochs).

## Sample (Llama-style model, prompt `KING RICHARD III:`, temperature 0.7)

```
KING RICHARD III:
Brother, and on thee cry the children of your virgins,
By this the other heart of an intercept,
That live I see the commons of men cold days:
But in the house of York and Margaret,
When I have set thee from my heart to
```

Iambic cadence, real syntax, speaker format, and near-coherent meaning — learned
from raw bytes in ~10 minutes on a laptop.

## Tokenizer comparison (bits/char, the fair metric)

| Tokenizer | Vocab | Bits/char | Notes |
| :-------- | ----: | --------: | :---- |
| byte      |   259 |  **2.08** | headline (Llama-style, converged) |
| bpe       |  1024 |     ~2.65 | undertrained — Zenith's from-scratch BPE tokenizer is `O(merges × corpus)` and slow on 1 MB; vectorized BPE is future work |

## Reproduce

```bash
python scripts/download_corpus.py
python -m zenith.cli.train \
    data.corpus_path=data/tiny_shakespeare.txt data.stride=128 \
    model.block_size=256 model.embed_dim=384 model.num_layers=6 \
    model.num_heads=6 model.ff_dim=1024 model.dropout=0.2 \
    training.epochs=12 training.batch_size=64 training.learning_rate=1e-3
zenith eval -m zenith-lm.pt -c data/tiny_shakespeare.txt
```

Switch to the GPT-2-style architecture for comparison with
`model.norm=layernorm model.positional=learned model.ffn=gelu`.

## Honest notes

- **2.08 bits/char matches, and marginally beats, the nanoGPT reference** — genuinely
  good for a from-scratch model at this scale. A bigger model or more data would go
  lower (well-tuned char models on larger corpora reach ~1.0–1.5 bpc).
- The model **overfits after ~10 epochs** on this tiny corpus; best-checkpoint
  saving handles it. Early stopping is a natural future addition.
- MPS is the bottleneck (~2–3 steps/s at this size); a CUDA GPU is far faster.
