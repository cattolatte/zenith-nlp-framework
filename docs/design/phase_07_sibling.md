# Phase 7 — Sibling showcase (Polaris interop)

## Goal
Demonstrate the sibling relationship concretely: Zenith (decoder / generation) and
Polaris (encoder / understanding) used together, linked through a **shared
tokenizer**, with neither depending on the other.

## Scope (what ships this phase)
- `zenith/interop/PolarisTokenizer` — adapts a Polaris tokenizer to Zenith's
  tokenizer interface (`encode`/`decode`/`token_bytes` + `bos_id`/`eos_id`/
  `pad_id`/`vocab_size`), so a Zenith decoder trains and generates over a Polaris
  vocabulary. Polaris is imported lazily; `import zenith.interop` never requires it.
- `examples/encode_and_generate.py` — trains a Polaris BPE tokenizer, then a Zenith
  decoder over it, then generates.
- Tests (`tests/test_interop/`) — skipped without the `[polaris]` extra
  (`pytest.importorskip`); verified locally against Polaris 1.1.0.

## Key decisions (ADR-0008)
- **Interop is optional and one-directional.** Zenith stays independent;
  `polaris-nlp` remains an optional extra. The bridge lives in `zenith.interop` and
  is never on the default path.
- **Adapter, not dependency.** The link is a small adapter over Polaris' public
  tokenizer API — the sibling relationship without coupling.

## Notes / verification
- Built and tested against the **real** Polaris API (not prose): `train_bpe`,
  `Encoding(ids, tokens)`, `Vocabulary` (`size`/`lookup_token`/specials).
- CI is CPU-only and doesn't install the extra, so interop tests **skip** there;
  they pass locally when `[polaris]` is installed.

## Out of scope
- A full "classify with Polaris → condition Zenith's generation" pipeline (needs a
  trained Polaris classifier bundle); the shared-tokenizer link is the clean, self-
  contained proof.

## Definition of done
`PolarisTokenizer` + example + tests land; a Zenith decoder generates over a
Polaris vocabulary; gates green; tag `v0.7.0`.
