# Phase 3 — Scaling & efficient fine-tuning

## Goal
Train bigger, cheaper, faster — without complicating the simple single-device
loop. Everything this phase adds is opt-in and defaults off.

## Scope (what ships this phase)
- `peft/` — **LoRA**: `LoRALinear`, `inject_lora` (targets `nn.Linear` by name
  substring; defaults to attention `qkv`/`proj`), freeze/parameter helpers, and a
  portable `lora_state_dict`.
- `distributed/` — **DDP** helpers (`torchrun`-native): rank/world/main-process,
  `wrap_model`, `make_sampler`, `resolve_device`; all no-ops single-process.
- `training/CausalLMTrainer` gains, all opt-in:
  - `use_lora` — train only the adapter, base frozen.
  - `grad_accum_steps` — larger effective batch on limited memory.
  - `amp` (+ `amp_dtype`) — `torch.autocast` (+ GradScaler for CUDA fp16).
  - DDP — auto-detected from the environment; sampler + main-process-only
    logging/sampling/checkpointing.
- `checkpoint.py` — LoRA-aware save/load (records the `LoraConfig`, re-injects on
  load so a LoRA checkpoint round-trips into a `Generator`).
- Config: `peft` group (`none`/`lora`), `training.{grad_accum_steps,amp,amp_dtype}`.
- Tests: LoRA unit tests; distributed single-process helpers; trainer learns with
  gradient accumulation; LoRA training freezes the base and moves the adapter, and
  its checkpoint round-trips.

## Key decisions (ADR-0004)
- **Opt-in, default-off.** The v0.2.0 single-device full-fine-tune path is
  byte-for-byte unchanged (`GradScaler(enabled=False)` and disabled autocast are
  pass-throughs; `grad_accum_steps=1` steps every batch).
- **DDP is environment-driven** (`torchrun`), not a config toggle; the helper layer
  makes the single- and multi-process code paths identical.
- **QLoRA and FSDP are deferred**: they need GPU-only libraries (bitsandbytes) that
  can't be exercised in CI. Adapter *merging* is also deferred.

## Out of scope (later)
- QLoRA (4-bit), FSDP, adapter merge-and-unload, sample-artifact logging.

## Definition of done
LoRA / accumulation / AMP / DDP all wired and (single-process) tested; the simple
path is unchanged; gates green; tag `v0.3.0`.
