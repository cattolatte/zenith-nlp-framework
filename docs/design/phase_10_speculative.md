# Phase 10 — Speculative decoding

## Goal
Make greedy decoding faster without changing its output. A small, cheap **draft**
model proposes several tokens; the large **target** verifies them in one forward
pass and keeps the longest prefix it agrees with. The generated text is
byte-for-byte identical to greedy decoding on the target — speed is the only thing
that changes.

## Scope (what ships this phase)
- `KVCache.truncate(length)` — roll the cache back to a committed prefix. The target
  verifies `k` draft tokens in one pass but may accept fewer; the rejected positions
  must be dropped so the cache again reflects exactly the committed sequence.
- `Generator.speculative_generate_ids(input_ids, draft_model, *, lookahead, ...)` and
  a `speculative_generate(prompt, draft_model, ...)` text wrapper.
- `SpeculativeStats` — `tokens`, `target_forwards`, `draft_forwards`, `proposed`,
  `accepted`, plus `acceptance_rate` and `speedup` (= tokens / target forwards).
- Tests: exactness vs greedy for a self-draft (100 % acceptance), a weaker draft
  (real rejections), and several `lookahead` values; block-size and batch guards.

## The algorithm (greedy, exact)
Each round, with the caches holding everything before the current token `last`:
1. The draft proposes `t₁..t_k` greedily (k cheap forwards, its own cache).
2. The target runs one forward over `[last, t₁..t_k]`, giving its argmax at each
   position.
3. Accept `t_i` while it equals the target's argmax there; let `m` be the accepted
   count. Emit `t₁..t_m` plus one **correction** (the target's own token at the first
   disagreement) — or, if all `k` matched, a **bonus** token. So every round commits
   `m + 1` tokens from a single target forward, and always makes progress.
4. Truncate both caches back to the committed prefix; the correction becomes the next
   `last`.

Because accepted tokens equal the target's greedy choice and the correction *is* the
target's greedy choice, the emitted stream is exactly what greedy would produce.

## Key decisions (ADR-0010)
- **Greedy/exact, not sampled.** The exactness guarantee is the whole point and is
  directly testable; the probabilistic (accept/reject with residual) variant is
  noted as future work, not shipped.
- **Cache rollback over recompute.** A `truncate` primitive keeps the target's
  per-step cost O(1) in context, which is where the speedup actually comes from.
- **Speedup is reported as target forwards saved** (`tokens / target_forwards`) —
  hardware-independent and honest, since wall-clock depends on the draft/target size
  ratio and the machine.

## Notes / verification
- Speculative output is asserted **equal to greedy** on the target across drafts and
  lookaheads (the cache-equivalence property makes the batched verify numerically
  match one-at-a-time decoding).
- Real benchmark: a 0.6M draft + the 10.7M target on tiny-shakespeare — see
  BENCHMARKS for the measured acceptance rate and forward-pass reduction.

## Out of scope
- Sampled speculative decoding (temperature > 0 with the accept/reject rule);
  tree/Medusa-style multi-branch drafting; self-speculation via early-exit layers.

## Definition of done
`truncate` + `speculative_generate_ids` land; output proven identical to greedy in
tests; a real draft/target speedup measured in BENCHMARKS; gates green; tag `v0.10.0`.
