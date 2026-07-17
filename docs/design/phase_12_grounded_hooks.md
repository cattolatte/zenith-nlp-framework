# Phase 12 — Grounded-generation hooks (v1.1.0)

## Goal
Give the generator the three primitives a grounded, citation-style consumer needs —
constrained decoding, a first-class refusal token, and a grounded SFT data format —
as **general mechanisms**, without baking any retrieval, citation-policy, or domain
logic into Zenith. Everything is additive: no existing public signature changes.

## Scope (what ships this phase)
- `zenith.generation.constraints` — a `LogitsConstraint` protocol
  (`(step, generated_ids, logits) -> logits`) plus an optional
  `logits_constraint=None` parameter on `Generator.generate_ids` and `stream`,
  applied to the next-token logits before sampling. `AllowedTokens` is the reusable
  implementation: when the last emitted token is a *trigger* id, the next token is
  masked to a caller-supplied *allowed* set (`-inf` elsewhere).
- **`<abstain>`** — `ByteTokenizer` and `BPETokenizer` reserve a stable
  `abstain_id`, appended after `bos`/`eos`/`pad` so no content or existing-special
  id is renumbered. `Generator.abstained(ids)` detects a refusal without decoding;
  the id also works as a `stop_ids` entry.
- `zenith.instruct.grounded` — `GroundedTemplate` formats `(question, passages)`
  prompts (passages as `(passage_id, text)`), and `GroundedInstructionDataset`
  builds `(prompt → cited answer | <abstain>)` examples, reusing the response-only
  masking extracted from `InstructionDataset` as `instruct.mask_prompt`.
- A PEP 561 `py.typed` marker, so downstream projects type-check against Zenith.

## Key decisions (ADR-0012)
- **Mechanism, not policy.** Zenith masks logits and formats prompts; *which* ids
  are valid citations, what a passage id means, and when to abstain are entirely
  the caller's decisions.
- **Additive-only.** New optional parameters and new types; `logits_constraint=None`
  reproduces unconstrained decoding bit-for-bit (tested).
- **Specials are appended, never renumbered.** `abstain_id` sits after the existing
  specials in both tokenizers, so pre-1.1 id assignments are untouched.
- **`GroundedTemplate` is standalone**, not a `ChatTemplate` subclass — its
  `format_prompt(question, passages)` is a different shape, and overriding an
  inherited signature would violate substitution under `mypy --strict`.

## Notes / verification
- Constraint honored on both the cached and uncached decode paths (they agree
  exactly); streaming honors it too.
- The consumer usage sketch (grounded LoRA SFT → citation-constrained,
  abstain-aware decoding) is a test that both runs offline and type-checks under
  `mypy --strict` — the acceptance criterion for the release.

## Out of scope
- Retrieval, RAG orchestration, citation syntax/policy, answerability judgments —
  downstream concerns. Grammar/FSM-constrained decoding beyond the trigger→allowed
  mechanism is future work.

## Definition of done
Hook + `AllowedTokens` + `<abstain>` + grounded SFT land, additive and gated
(pytest, ruff, black on new code, `mypy --strict` with no new errors); CHANGELOG +
version bump; the consumer sketch type-checks. Tag `v1.1.0`.
