# ADR-0010: Speculative decoding is greedy-exact, with cache rollback

## Status
Accepted.

## Context
Speculative decoding speeds up autoregressive generation by letting a small draft
model propose several tokens that a large target model verifies in one forward pass.
There are two families: a **greedy/exact** version (output identical to greedy
decoding on the target) and a **sampled** version (accept/reject against a residual
distribution, matching the target's sampling distribution). We also need a way to
undo speculative tokens the target rejects.

## Decision
1. **Ship the greedy/exact version first.** It has a clean, testable invariant —
   speculative output must be byte-for-byte identical to greedy on the target — which
   fits the project's honesty and offline-testing discipline. The sampled variant is
   documented as future work rather than shipped half-verified.
2. **Add `KVCache.truncate(length)` and roll the cache back**, rather than recomputing
   context each round. The target verifies `k` tokens in one pass and may accept a
   shorter prefix; truncation removes the rejected positions so the cache reflects
   exactly the committed sequence. Keeping the cache is what makes each target step
   O(1) in context — the source of the speedup.
3. **Fold the correction into the next round.** Each round commits the accepted draft
   prefix plus one correction/bonus token (always ≥ 1), so progress is guaranteed and
   the correction is verified for free by the next round's forward.
4. **Report speedup as `tokens / target_forwards`.** Greedy needs one target forward
   per token, so this ratio is the honest, hardware-independent speedup. Wall-clock
   gains depend on the draft/target cost ratio and are left to the benchmark.

## Consequences
- `truncate` is a small, general cache primitive (also useful for future
  backtracking/search) — added to `KVCache`, covered by the exactness tests.
- The feature is opt-in and additive: the draft is a second `DecoderLM`; nothing on
  the existing sampling/beam paths changes.
- The exactness guarantee holds only for greedy. Sampled speculative decoding (with
  temperature) would need a separate accept/reject rule and its own distribution test
  — deferred.
