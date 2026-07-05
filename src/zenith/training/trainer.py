"""Causal-LM training — Zenith's next-token training loop.

Design Principles
-----------------
Single-device, readable, and complete: next-token cross-entropy, AdamW, a
linear-warmup-then-cosine-decay schedule, gradient clipping, best-checkpoint
tracking, and optional MLflow logging. It samples text from the model at the end
of each epoch so training is legible, not just a falling number.

Distributed (DDP) training and parameter-efficient fine-tuning (LoRA) are
deliberately *not* here — they arrive in a later phase, wired in when the loop
actually needs them, rather than built speculatively now.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from ..checkpoint import save_checkpoint
from ..generation import Generator
from ..models import DecoderLM
from ..tokenizers import ByteTokenizer
from ..utils import get_logger, resolve_device, set_seed

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
    # experiment tracking (optional)
    tracking_enabled: bool = False
    experiment: str = "zenith"
    tracking_uri: str | None = None
    run_name: str | None = None
    # sampling during training (legibility)
    sample_prompt: str = ""
    sample_tokens: int = 120
    sample_temperature: float = 0.8
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
    """

    def __init__(
        self,
        model: DecoderLM,
        tokenizer: ByteTokenizer | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer or ByteTokenizer()
        self.config = config or TrainingConfig()
        self._log = get_logger("zenith.training")

    def fit(
        self, train_dataset: Dataset, val_dataset: Dataset | None = None
    ) -> dict[str, object]:
        """Run training and return a history dict; writes the best checkpoint."""
        cfg = self.config
        set_seed(cfg.seed)
        device = resolve_device(cfg.device)
        model = self.model.to(device)

        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
        )
        val_loader = (
            DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
            if val_dataset is not None
            else None
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        total_steps = cfg.epochs * max(len(train_loader), 1)
        scheduler = LambdaLR(
            optimizer, lambda step: _warmup_cosine(step, cfg.warmup_steps, total_steps)
        )
        loss_fn = nn.CrossEntropyLoss()

        tracker = self._make_tracker()
        history: list[dict[str, float]] = []
        best = math.inf

        run_ctx = tracker.run(cfg.run_name) if tracker is not None else nullcontext()
        with run_ctx:
            if tracker is not None:
                tracker.log_params(
                    {
                        "epochs": cfg.epochs,
                        "batch_size": cfg.batch_size,
                        "learning_rate": cfg.learning_rate,
                        "parameters": model.num_parameters(),
                        "block_size": model.config.block_size,
                    }
                )

            for epoch in range(cfg.epochs):
                model.train()
                running, steps = 0.0, 0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    logits = model(inputs)
                    loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    running += loss.item()
                    steps += 1

                metrics: dict[str, float] = {
                    "train_loss": running / max(steps, 1),
                    "lr": scheduler.get_last_lr()[0],
                }
                if val_loader is not None:
                    val_loss = self._evaluate(model, val_loader, loss_fn, device)
                    metrics["val_loss"] = val_loss
                    metrics["val_perplexity"] = math.exp(min(val_loss, 20.0))

                history.append(metrics)
                if tracker is not None:
                    tracker.log_metrics(metrics, step=epoch)
                self._log.info("epoch %d | %s", epoch + 1, _fmt(metrics))
                self._log_sample(model)

                score = metrics.get("val_loss", metrics["train_loss"])
                if score < best:
                    best = score
                    save_checkpoint(model, self.tokenizer, cfg.save_path)

        return {"history": history, "best_loss": best, "checkpoint": cfg.save_path}

    @torch.no_grad()
    def _evaluate(
        self, model: DecoderLM, loader: DataLoader, loss_fn: nn.Module, device: torch.device
    ) -> float:
        model.eval()
        total, steps = 0.0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            total += loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item()
            steps += 1
        return total / max(steps, 1)

    def _log_sample(self, model: DecoderLM) -> None:
        """Print a short generated sample so training is legible."""
        sample = Generator(model, self.tokenizer).generate(
            self.config.sample_prompt,
            max_new_tokens=self.config.sample_tokens,
            temperature=self.config.sample_temperature,
        )
        self._log.info("sample: %s", sample.replace("\n", " ")[:200])

    def _make_tracker(self) -> object | None:
        if not self.config.tracking_enabled:
            return None
        from ..tracking import MlflowTracker

        return MlflowTracker(
            experiment=self.config.experiment, tracking_uri=self.config.tracking_uri
        )


def _fmt(metrics: dict[str, float]) -> str:
    return " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
