# ADR-0012: Grounded-generation hooks are mechanisms; policy stays with the caller

## Status
Accepted.

## Context
Downstream consumers (e.g. retrieval-augmented systems) need the generator to
(1) restrict what it may emit at certain positions — so a citation can only
reference a passage actually in the context, (2) refuse first-class — abstain
rather than hallucinate when the context can't support an answer, and (3) be
fine-tuned on grounded examples — passages in the prompt, a cited answer or a
refusal in the target. The design question is where the *policy* lives: does
Zenith know about citations, passages, and answerability, or only supply neutral
machinery?

## Decision
1. **Zenith ships mechanisms, never policy.** `AllowedTokens` masks next-token
   logits to a caller-supplied id set at caller-supplied trigger positions;
   `GroundedTemplate` formats `(passage_id, text)` pairs it does not interpret;
   the `<abstain>` token carries no built-in semantics beyond "reserved id,
   detectable, decodes to nothing". Which ids are valid citations, and what
   deserves abstention, are the consumer's decisions.
2. **The constraint hook is a protocol applied before sampling.**
   `LogitsConstraint: (step, generated_ids, logits) -> logits`, passed as an
   optional `logits_constraint=None` on `generate_ids`/`stream`. Default `None`
   is bit-for-bit identical to unconstrained decoding — additive under semver.
3. **`<abstain>` is appended after the existing specials** (`bos`/`eos`/`pad`) in
   both tokenizers, so no content or existing-special id is ever renumbered;
   pre-1.1 checkpoints and encodings remain valid. Detection is
   `Generator.abstained(ids)` (id-level, no string matching).
4. **Grounded SFT reuses the instruction machinery.** The response-only masking is
   extracted as `mask_prompt` and shared; `GroundedInstructionDataset` subclasses
   `InstructionDataset` but `GroundedTemplate` is standalone (its prompt takes
   `(question, passages)`, and overriding `ChatTemplate.format_prompt`'s signature
   would violate substitution under `mypy --strict`).
5. **Ship a PEP 561 `py.typed` marker.** A typed consumer contract is part of the
   release: downstream `mypy --strict` must be able to validate calls against
   Zenith's real signatures.

## Consequences
- Citation-constrained decoding, abstention, and grounded fine-tuning are possible
  without Zenith knowing what a citation is; the same hook serves any
  restricted-vocabulary decoding need.
- The constraint is per-step and last-token-triggered — cheap and simple. Grammar-
  or FSM-level constrained decoding would be a new, separate constraint type.
- `abstain_id` exists on fresh tokenizers even when unused; it decodes to nothing,
  like the other specials.
- Downstream type-checking is now supported (and expected) — future releases must
  keep public annotations accurate.
