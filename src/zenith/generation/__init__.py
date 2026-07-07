"""Text generation — sampling and (later) search over a decoder model."""

from __future__ import annotations

from .generator import Generator, SpeculativeStats

__all__ = ["Generator", "SpeculativeStats"]
