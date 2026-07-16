"""Constrained decoding: the optional logits hook and the AllowedTokens constraint."""

import pytest
import torch

from zenith.generation import AllowedTokens, Generator, LogitsConstraint
from zenith.models import DecoderConfig, DecoderLM
from zenith.tokenizers import ByteTokenizer


def _gen(seed: int = 0) -> Generator:
    torch.manual_seed(seed)
    model = DecoderLM(
        DecoderConfig(
            vocab_size=259,
            block_size=64,
            embed_dim=32,
            num_layers=2,
            num_heads=2,
            ff_dim=64,
            dropout=0.0,
        )
    ).eval()
    return Generator(model, ByteTokenizer())


def _prompt() -> torch.Tensor:
    return torch.tensor([[1, 40, 41, 42, 43]], dtype=torch.long)


# --- backward compatibility ----------------------------------------------


def test_none_constraint_matches_unconstrained_bit_for_bit():
    gen = _gen()
    prompt = _prompt()
    plain = gen.generate_ids(prompt, max_new_tokens=25, temperature=0.0)
    with_none = gen.generate_ids(prompt, max_new_tokens=25, temperature=0.0, logits_constraint=None)
    assert torch.equal(plain, with_none)


def test_allowed_tokens_satisfies_the_protocol():
    assert isinstance(AllowedTokens(trigger_ids={1}, allowed_ids={2}), LogitsConstraint)


# --- the masking guarantee -----------------------------------------------


def test_constrained_generation_only_emits_allowed_ids():
    gen = _gen()
    prompt = _prompt()
    allowed = {10, 11, 12}
    # Trigger on the allowed ids and the last prompt token, so *every* step is
    # constrained → every generated token must land in the allowed set.
    constraint = AllowedTokens(trigger_ids=allowed | {int(prompt[0, -1])}, allowed_ids=allowed)
    out = gen.generate_ids(prompt, max_new_tokens=20, temperature=0.0, logits_constraint=constraint)
    generated = out[0, prompt.size(1) :].tolist()
    assert generated  # produced something
    assert set(generated) <= allowed


def test_constraint_agrees_with_and_without_cache():
    gen = _gen()
    prompt = _prompt()
    allowed = {5, 6, 7, 8}
    constraint = AllowedTokens(trigger_ids=allowed | {int(prompt[0, -1])}, allowed_ids=allowed)
    cached = gen.generate_ids(
        prompt, max_new_tokens=18, temperature=0.0, logits_constraint=constraint, use_cache=True
    )
    uncached = gen.generate_ids(
        prompt, max_new_tokens=18, temperature=0.0, logits_constraint=constraint, use_cache=False
    )
    assert torch.equal(cached, uncached)
    assert set(cached[0, prompt.size(1) :].tolist()) <= allowed


def test_empty_trigger_is_a_noop():
    gen = _gen()
    prompt = _prompt()
    plain = gen.generate_ids(prompt, max_new_tokens=20, temperature=0.0)
    noop = gen.generate_ids(
        prompt,
        max_new_tokens=20,
        temperature=0.0,
        logits_constraint=AllowedTokens(trigger_ids=set(), allowed_ids={1}),
    )
    assert torch.equal(plain, noop)


def test_stream_respects_the_constraint():
    gen = _gen()
    # Force the first byte to be a printable letter from a small allowed set.
    allowed = {ord("a"), ord("b"), ord("c")}
    constraint = AllowedTokens(trigger_ids={ord(">")}, allowed_ids=allowed)
    text = "".join(gen.stream(">", max_new_tokens=1, temperature=0.0, logits_constraint=constraint))
    assert text and all(ord(ch) in allowed for ch in text)


def test_allowed_ids_must_be_non_empty():
    with pytest.raises(ValueError):
        AllowedTokens(trigger_ids={1}, allowed_ids=set())
