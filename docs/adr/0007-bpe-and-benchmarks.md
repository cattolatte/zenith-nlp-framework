# ADR-0007: Byte-level BPE; honest benchmarks (no invented numbers)

## Status
Accepted.

## Context
The capstone adds a learned tokenizer and a benchmark story. Two decisions matter:
how the tokenizer guarantees correctness, and how benchmark numbers are reported
given the development constraints.

## Decision
1. **The BPE tokenizer is byte-level.** It starts from the 256 byte values and
   learns merges up to a target vocabulary. Consequences: no unknown token ever,
   any string round-trips losslessly, and correctness is independent of merge
   quality (merges are reversible byte concatenations). The round-trip test is the
   invariant, not the specific merges. It stays interface-compatible with
   `ByteTokenizer` (`encode`/`decode`/`token_bytes`/specials), so it drops into
   training, generation and serving unchanged; the checkpoint serializes it.
2. **Benchmarks ship methodology, not invented numbers.** This project is developed
   in an environment without a GPU/torch runtime, so the maintainer cannot run real
   training here. `BENCHMARKS.md` fixes the setup and reproduction commands and
   leaves the results table to be filled by an actual run. Publishing fabricated
   numbers would violate the project's honesty discipline (and mislead readers).

## Consequences
- BPE and byte tokenizers are interchangeable; checkpoints record which was used.
- Streaming works for any tokenizer via `token_bytes` (UTF-8-complete chunks).
- The benchmark table is empty until someone runs it — deliberately. The value is a
  fair, reproducible comparison, not a headline score.
