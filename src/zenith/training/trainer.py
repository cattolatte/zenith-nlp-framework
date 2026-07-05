"""Causal-LM training — Zenith's next-token training loop.

Design Principles
-----------------
One readable loop that scales up on demand. The default path (single device, full
fine-tuning) is unchanged from earlier releases; four opt-in capabilities layer on
top without complicating that path:

- **LoRA** — train a low-rank adapter with the base frozen (``use_lora``).
- **Gradient accumulation** — larger effective batch on limited memory
  (``grad_accum_steps``).
- **Mixed precision** — ``torch.autocast`` (+ a GradScaler for CUDA fp16)
  (``amp``).
- **Distributed data parallel** — multi-GPU via ``torchrun`` (auto-detected from
  the environment; ``zenith.distributed`` degrades to single-process).

Everything defaults off, so ``CausalLMTrainer(model).fit(dataset)`` behaves exactly
as before. QLoRA and FSDP are intentionally deferred (they need GPU-only libraries
that can't be exercised in CI).
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from .. import distributed as dist
from ..checkpoint import save_checkpoint
from ..experiments import capture_environment, record_run
from ..generation import Generator
from ..models import DecoderLM
from ..peft import LoraConfig, count_trainable_parameters, inject_lora, lora_parameters
from ..tokenizers import ByteTokenizer
from ..utils import get_logger, set_deterministic, set_seed

__all__ = ["TrainingConfig", "CausalLMTrainer"]


@dataclass
class TrainingConfig:
    """Hyperparameters and options for a training run."""

    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip: float = 1.0
    seed: int = 0
    device: str | None = None
    num_workers: int = 0
    # scaling
    grad_accum_steps: int = 1
    amp: bool = False
    amp_dtype: str = "bf16"  # "bf16" or "fp16"
    # parameter-efficient fine-tuning
    use_lora: bool = False
    lora: LoraConfig = field(default_factory=LoraConfig)
    # experiment tracking
    tracking_enabled: bool = False
    experiment: str = "zenith"
    tracking_uri: str | None = None
    run_name: str | None = None
    # in-training samples
    sample_prompt: str = ""
    sample_tokens: int = 120
    sample_temperature: float = 0.8
    log_samples: bool = True
    # reproducibility
    deterministic: bool = False
    record_dir: str | None = None
    # checkpointing
    save_path: str = "zenith-lm.pt"


def _warmup_cosine(step: int, warmup: int, total: int) -> float:
    """Linear warmup to 1.0, then cosine decay toward 0."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


class CausalLMTrainer:
    """Train a :class:`DecoderLM` on a causal-LM dataset.

    Parameters
    ----------
    model : DecoderLM
        The model to train.
    tokenizer : ByteTokenizer, optional
        Used for in-training text samples and saved with the checkpoint.
    config : TrainingConfig, optional
        Run hyperparameters; defaults are sensible for a tiny model.
    run_config : dict, optional
        The full run configuration (e.g. a resolved Hydra config) to log to MLflow
        and record on disk. If omitted, the ``TrainingConfig`` itself is used.
    """

    def __init__(
        self,
        model: DecoderLM,
        tokenizer: ByteTokenizer | None = None,
        config: TrainingConfig | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer or ByteTokenizer()
        self.config = config or TrainingConfig()
        self.run_config = run_config
        self._log = get_logger("zenith.training")

    def fit(
        self, train_dataset: Dataset, val_dataset: Dataset | None = None
    ) -> dict[str, object]:
        """Run training and return a history dict; writes the best checkpoint."""
        cfg = self.config
        if cfg.deterministic:
            set_deterministic(cfg.seed)
        else:
            set_seed(cfg.seed)
        environment = capture_environment(cfg.seed)
        device = torch.device(cfg.device) if cfg.device else dist.resolve_device()

        with dist.distributed_context():
            if cfg.use_lora:
                inject_lora(self.model, cfg.lora)

            self.model.to(device)
            model: nn.Module = dist.wrap_model(self.model)
            raw = dist.unwrap_model(model)

            sampler = dist.make_sampler(train_dataset, shuffle=True)
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                sampler=sampler,
                shuffle=sampler is None,
                num_workers=cfg.num_workers,
            )
            val_loader = (
                DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
                if val_dataset is not None
                else None
            )

            params = lora_parameters(raw) if cfg.use_lora else list(raw.parameters())
            optimizer = torch.optim.AdamW(
                params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
            accum = max(cfg.grad_accum_steps, 1)
            total_steps = cfg.epochs * max(len(train_loader) // accum, 1)
            scheduler = LambdaLR(
                optimizer, lambda step: _warmup_cosine(step, cfg.warmup_steps, total_steps)
            )
            loss_fn = nn.CrossEntropyLoss()

            amp_dtype = torch.float16 if cfg.amp_dtype == "fp16" else torch.bfloat16
            use_autocast = cfg.amp and device.type in ("cuda", "cpu")
            scaler = torch.cuda.amp.GradScaler(
                enabled=cfg.amp and device.type == "cuda" and amp_dtype is torch.float16
            )

            main = dist.is_main_process()
            tracker = self._make_tracker() if main else None
            history: list[dict[str, float]] = []
            samples: list[str] = []
            best = math.inf

            run_ctx = tracker.run(cfg.run_name) if tracker is not None else nullcontext()
            with run_ctx:
                if tracker is not None:
                    trainable, total = count_trainable_parameters(raw)
                    tracker.log_config(self._effective_config())
                    tracker.log_params(
                        {
                            "world_size": dist.world_size(),
                            "trainable_params": trainable,
                            "total_params": total,
                        }
                    )

                for epoch in range(cfg.epochs):
                    if sampler is not None:
                        sampler.set_epoch(epoch)
                    metrics = self._train_epoch(
                        model, train_loader, optimizer, scheduler, scaler, loss_fn,
                        device, use_autocast, amp_dtype, accum, params, cfg.grad_clip,
                    )
                    if val_loader is not None:
                        val_loss = self._evaluate(
                            model, val_loader, loss_fn, device, use_autocast, amp_dtype
                        )
                        metrics["val_loss"] = val_loss
                        metrics["val_perplexity"] = math.exp(min(val_loss, 20.0))

                    history.append(metrics)
                    if main:
                        if tracker is not None:
                            tracker.log_metrics(metrics, step=epoch)
                        self._log.info("epoch %d | %s", epoch + 1, _fmt(metrics))
                        if cfg.log_samples:
                            sample = self._generate_sample(raw)
                            samples.append(sample)
                            self._log.info("sample: %s", sample.replace("\n", " ")[:200])
                            if tracker is not None:
                                tracker.log_text(sample, f"samples/epoch_{epoch + 1}.txt")
                        score = metrics.get("val_loss", metrics["train_loss"])
                        if score < best:
                            best = score
                            save_checkpoint(
                                raw, self.tokenizer, cfg.save_path,
                                lora=cfg.lora if cfg.use_lora else None,
                            )

                if main:
                    self._record(tracker, history, samples, environment)

        return {"history": history, "best_loss": best, "checkpoint": cfg.save_path}

    def _record(self, tracker, history, samples, environment) -> None:
        """Log the checkpoint artifact and write an on-disk run record."""
        cfg = self.config
        from pathlib import Path

        if tracker is not None and Path(cfg.save_path).exists():
            tracker.log_artifact(cfg.save_path)
        if cfg.record_dir is not None:
            path = record_run(
                cfg.record_dir,
                config=self._effective_config(),
                history=history,
                samples=samples,
                environment=environment,
            )
            self._log.info("recorded run to %s", path)

    def _effective_config(self) -> dict[str, Any]:
        """Full run config to log/record — the provided one, or the TrainingConfig."""
        return dict(self.run_config) if self.run_config is not None else asdict(self.config)

    def _train_epoch(
        self, model, train_loader, optimizer, scheduler, scaler, loss_fn,
        device, use_autocast, amp_dtype, accum, params, grad_clip,
    ) -> dict[str, float]:
        model.train()
        optimizer.zero_grad()
        running, batches = 0.0, 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            with self._autocast(device, use_autocast, amp_dtype):
                logits = model(inputs)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0:
                self._optimizer_step(scaler, optimizer, params, grad_clip)
                scheduler.step()
            running += loss.item()
            batches += 1
        if batches % accum != 0:  # flush a partial final accumulation group
            self._optimizer_step(scaler, optimizer, params, grad_clip)
            scheduler.step()
        return {"train_loss": running / max(batches, 1), "lr": scheduler.get_last_lr()[0]}

    @staticmethod
    def _autocast(device, use_autocast, amp_dtype):
        """Autocast context when AMP is on, else a no-op (avoids MPS device issues)."""
        if use_autocast:
            return torch.autocast(device_type=device.type, dtype=amp_dtype)
        return nullcontext()

    @staticmethod
    def _optimizer_step(scaler, optimizer, params, grad_clip) -> None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    @torch.no_grad()
    def _evaluate(
        self, model, loader, loss_fn, device, use_autocast, amp_dtype
    ) -> float:
        model.eval()
        total, steps = 0.0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with self._autocast(device, use_autocast, amp_dtype):
                logits = model(inputs)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total += loss.item()
            steps += 1
        return total / max(steps, 1)

    def _generate_sample(self, model: DecoderLM) -> str:
        return Generator(model, self.tokenizer).generate(
            self.config.sample_prompt,
            max_new_tokens=self.config.sample_tokens,
            temperature=self.config.sample_temperature,
        )

    def _make_tracker(self) -> object | None:
        if not self.config.tracking_enabled:
            return None
        from ..tracking import MlflowTracker

        return MlflowTracker(
            experiment=self.config.experiment, tracking_uri=self.config.tracking_uri
        )


def _fmt(metrics: dict[str, float]) -> str:
    return " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
