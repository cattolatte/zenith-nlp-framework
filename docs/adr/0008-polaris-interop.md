# ADR-0008: Polaris interop is an optional adapter, not a dependency

## Status
Accepted.

## Context
Zenith and Polaris are independent siblings (decoder/generation vs
encoder/understanding). Phase 7 demonstrates them working together without
compromising Zenith's independence — the whole point of the project is that Zenith
does *not* depend on Polaris.

## Decision
1. **The bridge is an optional adapter in `zenith.interop`.** `PolarisTokenizer`
   wraps a Polaris tokenizer to Zenith's tokenizer interface so a Zenith decoder
   can generate over a Polaris vocabulary. `polaris-nlp` stays an optional extra
   (`pip install zenith-nlp[polaris]`), imported lazily; `import zenith` and
   `import zenith.interop` never require it.
2. **Built against the real Polaris API, verified by tests.** The adapter targets
   Polaris' actual public surface (`train_bpe`, `Encoding`, `Vocabulary`), checked
   with live tests — not prose assumptions. (An earlier, abandoned design coded
   against a *documented* Polaris API and every assumption was wrong; this one is
   verified.)
3. **Interop tests skip without the extra.** They use `pytest.importorskip` so CI
   (which doesn't install Polaris) stays green; they run locally when the extra is
   present.

## Consequences
- Zenith remains standalone; the sibling relationship is demonstrable but never
  load-bearing.
- The shared-tokenizer link is the clean proof; a Polaris-classifier-conditioned
  generation pipeline is left as future work (needs a trained Polaris bundle).
