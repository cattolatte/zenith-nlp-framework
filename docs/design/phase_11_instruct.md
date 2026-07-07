# Phase 11 — Instruction fine-tuning (mini chat)

## Goal
Turn a pretrained base decoder into a small instruction-following chat model with
the standard supervised fine-tuning (SFT) recipe — a prompt template, response-only
loss masking, and an interactive chat loop — and be honest about what a ~10M
byte-level model can and cannot do.

## Scope (what ships this phase)
- `zenith.instruct.ChatTemplate` — one explicit template shared by training and
  inference, so the model is fine-tuned on exactly the string it later sees. Plain
  text (`### Instruction: … ### Response: …`), so it needs no special vocabulary and
  works with the byte tokenizer; the response ends at the tokenizer's EOS id.
- `zenith.instruct.InstructionDataset` — formats `(instruction, response)` pairs and
  emits fixed-length `(input, target)` with **prompt and padding masked to `-100`**.
  Only the response (and its EOS) is supervised, so the model learns to *produce*
  answers, not re-predict the prompt. Loss masking is free: `nn.CrossEntropyLoss`
  already ignores `-100`.
- `zenith.instruct.load_instructions` + a bundled `data/instructions.jsonl` (small,
  curated, with paraphrase clusters so a tiny model can actually learn).
- `scripts/finetune_instruct.py` — SFT a base checkpoint into a chat model.
- `zenith chat --instruct` — wraps input in the template and stops at EOS; enabled by
  a new `stop_ids` argument on `Generator.generate_ids` / `stream`.
- `MODEL_CARD.md` documenting data, method, and (honestly) the limits.
- Tests: template formatting, masking correctness (prompt/pad = `-100`, response +
  EOS supervised), the JSONL loader, and EOS stopping.

## Key decisions (ADR-0011)
- **Response-only masking via the `-100` convention.** No trainer changes needed —
  the dataset carries the mask and the existing cross-entropy honours it.
- **One template, two call sites.** The same `ChatTemplate` builds training examples
  and inference prompts, so train/serve can't drift.
- **Full fine-tune of a small model** (LoRA is available but unnecessary at this
  size); SFT from the pretrained base so the model keeps its English fluency.
- **Honesty about scale.** The dataset is tiny and the model small, so this
  memorises more than it generalises. The model card says so plainly; the value is a
  correct, end-to-end SFT pipeline, not a capable assistant.

## Notes / verification
- Masking is unit-tested: exactly the response tokens plus EOS are supervised;
  prompt and padding are `-100`.
- After SFT the model follows the template and stops cleanly at EOS — see the model
  card for sample exchanges.

## Out of scope
- RLHF / DPO / preference tuning; multi-turn conversation state; a large or
  general-purpose instruction corpus. This is single-turn SFT.

## Definition of done
Template + masked dataset + fine-tune script land; `zenith chat --instruct` works and
stops at EOS; masking is tested; a model card documents the result honestly; gates
green; tag `v0.11.0`.
