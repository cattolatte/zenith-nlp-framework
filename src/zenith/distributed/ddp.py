"""Distributed Data Parallel (DDP) helpers for multi-GPU training.

Design Principles
-----------------
Thin, well-behaved wrappers over ``torch.distributed`` that read the standard
``torchrun`` environment (``RANK``/``WORLD_SIZE``/``LOCAL_RANK``). Every helper
degrades gracefully on a single process: ``is_distributed()`` is ``False``,
``wrap_model`` returns the model untouched, ``make_sampler`` returns ``None``. So
the trainer can be written once and run either way.

Launch a distributed job with:

    torchrun --nproc_per_node=<N> -m zenith.cli.train distributed=ddp
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

__all__ = [
    "world_size",
    "rank",
    "local_rank",
    "is_distributed",
    "is_main_process",
    "resolve_device",
    "setup",
    "cleanup",
    "wrap_model",
    "unwrap_model",
    "make_sampler",
    "distributed_context",
]


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return world_size() > 1


def is_main_process() -> bool:
    """The single rank that should log, sample, and checkpoint."""
    return rank() == 0


def resolve_device() -> torch.device:
    """The device this process trains on.

    In distributed runs without CUDA we fall back to CPU (gloo): PyTorch's
    collective ops are not implemented for Apple MPS, so DDP on MPS would crash.
    Single-process runs still use MPS when available.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank()}")
    if is_distributed():
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup() -> None:
    """Initialise the process group (no-op when single-process / already up)."""
    if not is_distributed() or dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank(), world_size=world_size())
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Wrap in DDP when distributed; otherwise return the model unchanged."""
    if not is_distributed():
        return model
    device_ids = [local_rank()] if torch.cuda.is_available() else None
    return DDP(model, device_ids=device_ids)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module (unwrapping DDP if present)."""
    return model.module if isinstance(model, DDP) else model


def make_sampler(dataset: Dataset, *, shuffle: bool = True) -> DistributedSampler | None:
    """A ``DistributedSampler`` when distributed, else ``None`` (plain shuffle)."""
    if not is_distributed():
        return None
    return DistributedSampler(dataset, shuffle=shuffle)


@contextmanager
def distributed_context() -> Iterator[None]:
    """Set up the process group on entry, tear it down on exit."""
    setup()
    try:
        yield
    finally:
        cleanup()
