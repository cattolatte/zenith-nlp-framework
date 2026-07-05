"""Small, self-contained utilities shared across Zenith.

Design Principles
-----------------
Zenith owns its own primitives; this module deliberately depends only on the
standard library and ``torch`` so that importing it drags in nothing heavy or
optional. Device/seed/logging helpers live here rather than being borrowed from a
sibling project — Zenith stands on its own.
"""

from __future__ import annotations

import logging
import random

import torch

__all__ = ["set_seed", "set_deterministic", "resolve_device", "get_logger"]


def set_seed(seed: int) -> None:
    """Seed Python and torch RNGs for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_deterministic(seed: int) -> None:
    """Seed RNGs and request deterministic algorithms (best-effort).

    ``warn_only=True`` keeps a training run alive if some op lacks a deterministic
    implementation, rather than raising — determinism is a goal, not a hard gate.
    """
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device (MPS → CUDA → CPU), or honour ``prefer``."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str = "zenith") -> logging.Logger:
    """A module logger that configures a sane default handler once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
