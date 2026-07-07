# Model card — Zenith mini chat (instruction-tuned)

A tiny, instruction-tuned demonstration model produced by Phase 11. It exists to
show the **supervised fine-tuning (SFT) pipeline end to end**, not to be a capable
assistant. Read the limitations before expecting anything of it.

## Overview
| | |
|---|---|
| Base | 10.7M Llama-style decoder (RoPE, RMSNorm, SwiGLU), byte-level, pretrained on tiny-shakespeare |
| Method | full supervised fine-tuning with response-only loss masking |
| Data | `data/instructions.jsonl` — 84 curated single-turn instruction/response pairs |
| Tokenizer | byte-level (259 ids); EOS marks the end of a response |
| Context | 256 tokens |

## Intended use
A teaching/demo artifact: how a base language model is turned into an
instruction-follower (prompt template, response-only masking, EOS-terminated chat).
Not for production, not a knowledge source, not safety-tuned.

## Prompt format
The same `ChatTemplate` is used for training and inference:

```
### Instruction:
{your message}

### Response:
```

The model generates the response and stops at EOS. Run it with:

```bash
zenith chat --instruct -m instruct-shakespeare.pt
```

## Example exchanges (real, greedy decoding)

**Trained instructions — works well:**

| Instruction | Response |
|---|---|
| Who are you? | I am Zenith, a small language model trained from scratch. |
| Count to five. | One, two, three, four, five. |
| What is two plus two? | Two plus two is four. |
| What is the capital of France? | The capital of France is Paris. |
| Tell me a joke. | Why did the programmer quit? Because they did not get arrays. |

**Held-out phrasings — mixed, honestly:**

| Instruction | Response | Verdict |
|---|---|---|
| Introduce yourself please. | I am Zenith, a small language model trained from scratch. | ✓ generalised (paraphrase) |
| What is three plus three? | Sunlight scatters off the air, and blue light scatters… | ✗ wrong content |
| What is the opposite of fast? | The opposite of big is small. | ✗ right format, wrong content |

## Limitations
- **Memorises more than it generalises.** With 84 examples and ~10M parameters, it
  reproduces trained answers (and close paraphrases) well, but for genuinely unseen
  instructions it emits a plausibly-formatted but often wrong response.
- **No knowledge.** Any "facts" are memorised strings from the tiny dataset; it
  cannot reason or look anything up.
- **Archaic base.** Pretraining was on Shakespeare, so its English can drift.
- **Single-turn only.** No conversation memory; no system prompt.
- **Not safety-tuned.** No preference tuning, no guardrails.

## Reproduce
```bash
python scripts/download_corpus.py
# ... pretrain a base (see BENCHMARKS.md), or use an existing checkpoint ...
python scripts/finetune_instruct.py --base shakespeare-llama.pt \
    --data data/instructions.jsonl --out instruct-shakespeare.pt --epochs 60
zenith chat --instruct -m instruct-shakespeare.pt
```

The value here is a correct, legible SFT pipeline — prompt template, response-only
masking, EOS-terminated generation — that would scale to a real model given a real
base and a real instruction corpus.
