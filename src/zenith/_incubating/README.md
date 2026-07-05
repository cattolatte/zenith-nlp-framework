# `zenith._incubating` — frozen from-scratch generative engine

This package is a **frozen snapshot** of Zenith's original "from-scratch NLP
framework" code. It is **not part of Zenith's supported surface** and nothing on
the main path imports it. It exists so that no work is lost while Zenith's
identity changes.

## Why it's here

Zenith is now an **MLOps / scaling / orchestration platform on top of
[`polaris-nlp`](https://pypi.org/project/polaris-nlp/)** (see the repo root
`README.md` and `PolarisContext.md`). Polaris owns the model math: tokenizers,
attention, encoder architectures, single-GPU training, and metrics.

Most of Zenith's old encoder-side code was a direct duplicate of Polaris and was
**deleted**. But Polaris is **encoder / classification only** — it has no
decoder / causal-LM / seq2seq. The genuinely non-duplicative pieces below have no
Polaris equivalent to migrate onto, so they were archived here instead of being
thrown away:

| Archived | What it is |
| --- | --- |
| `models/gpt_model.py` | Decoder-only (GPT) transformer |
| `models/transformer_model.py` | Seq2seq encoder-decoder transformer |
| `models/hybrid_model.py` | CNN-RNN hybrid classifier |
| `models/bert_model.py` | From-scratch BERT encoder (superseded by Polaris) |
| `tasks/text_summarization.py` | Seq2seq summarization task script |
| `tasks/question_answering.py` | Span-QA task script |
| `tasks/ner.py` | Token-classification (NER) task script |
| `core/*`, `training/`, `evaluation/`, `data/`, `utils/` | Primitives the above depend on |

## Status: not wired up

This code is kept **importable in isolation** but is intentionally excluded from
Zenith's public API, CLI, tests, and CI gates. Reviving any of it is a
**deliberate future decision** (Polaris' own positioning notes suggest Zenith
could become the "generation-side sibling" — a decoder/seq2seq counterpart that
reuses Polaris tokenizers/collation). When that phase starts, this snapshot is
the starting point; expect to re-plumb it onto Polaris primitives rather than the
frozen copies bundled here.
