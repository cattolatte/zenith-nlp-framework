# Contributing to Zenith

Thanks for your interest! Zenith is a from-scratch generative NLP library, and it
values **clarity and engineering discipline** as much as features. This guide
covers how to set up, the standards a change must meet, and how work is organized.

## Development setup

Python 3.10+.

```bash
git clone https://github.com/cattolatte/zenith-nlp-framework.git
cd zenith-nlp-framework
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[all,dev]"
```

## The gates (all must pass)

Every change must pass, locally and in CI (Python 3.10 / 3.11 / 3.12):

```bash
ruff check src tests      # lint + import order
pytest -q                 # full offline test suite
```

`black` is the formatter (`black src tests`). Tests are **fully offline** — no
network, no downloads. Optional-dependency tests (e.g. Polaris interop) use
`pytest.importorskip` so they skip cleanly when the extra isn't installed.

## Standards

- **From scratch, on tensor primitives.** Model internals (attention, blocks,
  layernorm, tokenizers, sampling) are hand-written; PyTorch supplies autograd,
  containers, and optimizers — not the architecture.
- **Own the interface.** Heavy/optional deps (torch, mlflow, fastapi, polaris) are
  imported lazily and behind optional extras; `import zenith` stays cheap.
- **Readability over cleverness**, typing everywhere (`from __future__ import
  annotations`), NumPy-style docstrings with a "Design Principles" note.
- **Tests assert behavior**, not exact floats: shapes, invariants, and learning
  behavior on tiny fixtures.
- **No speculative infrastructure.** Build a capability the slice that first needs
  it.

## How work is organized

- **Branch → PR.** Never commit to `main` directly; open a pull request and let CI
  gate it.
- **Design docs & ADRs.** Significant, hard-to-reverse decisions get an ADR in
  `docs/adr/`; each development phase has a design doc in `docs/design/`.
- **Changelog & semver.** User-facing changes go in `CHANGELOG.md`
  (Keep-a-Changelog); releases are tagged `vX.Y.Z`.

## Reporting bugs / requesting features

Use the issue templates. For a bug, include the command, environment (OS, Python,
torch version — `zenith info`), and the traceback. For security issues, see
[SECURITY.md](SECURITY.md).
