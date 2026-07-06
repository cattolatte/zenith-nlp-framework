# Phase 9 — Scaling study & a harder benchmark

## Goal
Turn "it matches nanoGPT at one size" into a *measured* story: how does quality
scale with model size, and how does the model do on a corpus that isn't
Shakespeare? Produce a real research-style figure and an honest second benchmark.

## Scope (what ships this phase)
- `scripts/scaling_study.py` — trains a sweep of Llama-style decoders (same recipe,
  only width/depth change) and reports held-out **bits/char** for each. Reproducible
  and committed, so the figure isn't a one-off.
- BENCHMARKS: a **bits/char-vs-params** table + a small ASCII curve, with the honest
  read (diminishing returns as the tiny-shakespeare data floor bites).
- A **harder benchmark**: train on a *subset* of **text8** (the standard char-LM
  corpus — lowercased Wikipedia, 27-symbol alphabet), reported explicitly as a
  subset (not the full 100 MB), out-of-domain vs Shakespeare.
- Ship as **v0.9.1** (benchmark + script; no library API change).

## Key decisions
- **Fixed recipe, vary only size.** Same block/optimizer/epochs across the sweep so
  the curve isolates capacity, not hyperparameters.
- **Honest about compute.** Full text8/enwik8 (100 MB) is a multi-hour job on MPS;
  we run a subset and *say so*, rather than publishing a number we didn't earn or
  faking a headline. (Consistent with ADR-0007's honesty stance, now that we do run
  real training — see ADR-0009.)
- **Bits/char is the axis.** Tokenizer-independent, comparable across corpora.

## Measured result (tiny-shakespeare, Llama-style, same recipe)
| Params | Bits/char |
| -----: | --------: |
| 0.60M  | 2.329 |
| 1.82M  | 2.137 |
| 5.06M  | 2.077 |
| 10.7M  | (headline ~2.08) |

Each doubling of parameters helps less — a clean diminishing-returns curve that
flattens into the data floor. The lesson (stated in BENCHMARKS): at this corpus
size, more capacity stops paying off; breaking ~2.0 needs more *data*, not a bigger
model — which motivates the text8 subset.

## Out of scope
- Full-scale text8/enwik8 SOTA runs (need a CUDA box); compute-optimal
  (Chinchilla-style) scaling laws; multi-seed error bars.

## Definition of done
Scaling script + figure in BENCHMARKS; a text8-subset number reported honestly;
gates green; tag `v0.9.1`.
