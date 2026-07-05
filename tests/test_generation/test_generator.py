"""Tests for text generation."""

import torch

from zenith.generation import Generator
from zenith.models import DecoderConfig, DecoderLM
from zenith.tokenizers import ByteTokenizer


def _gen() -> Generator:
    model = DecoderLM(
        DecoderConfig(vocab_size=259, block_size=16, embed_dim=32, num_layers=2, num_heads=2, ff_dim=64)
    )
    return Generator(model, ByteTokenizer())


def test_generate_ids_extends_by_max_new_tokens():
    gen = _gen()
    out = gen.generate_ids(torch.zeros(1, 3, dtype=torch.long), max_new_tokens=7)
    assert out.shape == (1, 10)


def test_generate_returns_string():
    text = _gen().generate("hello", max_new_tokens=5)
    assert isinstance(text, str)


def test_greedy_is_deterministic():
    gen = _gen()
    x = torch.zeros(1, 2, dtype=torch.long)
    a = gen.generate_ids(x, max_new_tokens=8, temperature=0.0)
    b = gen.generate_ids(x, max_new_tokens=8, temperature=0.0)
    assert torch.equal(a, b)


def test_empty_prompt_still_generates():
    text = _gen().generate("", max_new_tokens=5)
    assert isinstance(text, str)
