# ADR-0004: Scaling features are opt-in; QLoRA/FSDP deferred

## Status
Accepted.

## Context
Phase 3 adds LoRA, gradient accumulation, mixed precision, and distributed
training. Each interacts with the training loop, and the loop was clean and tested.
The risk is turning one readable loop into a tangle of branches, or shipping
GPU-only code paths that CI cannot exercise.

## Decision
1. **Everything is opt-in and defaults off.** The single-device, full-fine-tune
   path is unchanged from v0.2.0: `GradScaler(enabled=False)` and disabled
   autocast are pass-throughs, and `grad_accum_steps=1` steps every batch. Existing
   behaviour and tests are preserved.
2. **DDP is driven by the environment (`torchrun`), not a config flag.** The
   `zenith.distributed` helpers degrade to no-ops on a single process, so the loop
   is written once and the single-process path (which CI runs) exercises most of it.
3. **LoRA lives in `peft/` and operates generically** on `nn.Linear` by name; the
   trainer only decides *whether* to inject and *which* parameters to optimize.
4. **QLoRA (4-bit) and FSDP are deferred.** They depend on GPU-only libraries
   (e.g. bitsandbytes) that cannot be installed or tested in CI; shipping untested
   GPU code would violate the project's "everything is tested" discipline. They
   return when there is a way to validate them.

## Consequences
- The simple path stays simple and fully tested; scaling is available when needed.
- Multi-GPU (DDP) correctness is validated on hardware, not CI; the single-process
  path is covered by tests.
- A LoRA checkpoint records its `LoraConfig` and re-injects on load, so it
  round-trips like any other checkpoint.
