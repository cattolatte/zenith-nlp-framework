# ADR-0011: Instruction tuning is single-turn SFT with response-only masking

## Status
Accepted.

## Context
The final phase turns a pretrained base decoder into a small chat model. The design
questions: how to format prompts, what loss to optimise, whether to change the
trainer, and how to set expectations for a ~10M byte-level model.

## Decision
1. **Supervised fine-tuning with response-only loss masking.** Each
   `(instruction, response)` pair is formatted with a single `ChatTemplate`; the
   prompt tokens and padding are labelled `-100` so only the response (and its EOS)
   contributes to the loss. The model learns to *produce* answers rather than to
   re-predict the instruction.
2. **Reuse the existing trainer unchanged.** `nn.CrossEntropyLoss` already ignores
   `-100`, so the dataset carries the mask and no training-loop change is needed.
3. **One template shared by training and inference.** `ChatTemplate` builds both the
   SFT examples and the chat-time prompt, so the two can never drift. It is plain
   text (no special tokens), so the byte tokenizer is unchanged; EOS marks the end of
   a response and stops generation (`stop_ids` on the generator).
4. **Full fine-tune from the pretrained base**, keeping its English fluency. LoRA
   exists and works but is unnecessary at this size.
5. **Be explicit about scale.** The bundled dataset is small and the model tiny, so
   the result memorises more than it generalises. The model card states this plainly;
   the deliverable is a correct end-to-end SFT pipeline, not a capable assistant.

## Consequences
- Instruction tuning is additive: a new `zenith.instruct` module, a bundled dataset,
  a fine-tune script, and a `--instruct` chat flag — nothing on the pretraining path
  changes.
- The `-100` masking convention is now load-bearing; it is unit-tested (exactly the
  response + EOS is supervised).
- Multi-turn chat, preference tuning (RLHF/DPO), and a large instruction corpus are
  out of scope and left as future work.
