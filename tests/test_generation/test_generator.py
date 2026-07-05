"""Tests for text generation: sampling strategies, beam search, and the KV-cache."""

import torch

from zenith.generation import Generator
from zenith.models import DecoderConfig, DecoderLM
from zenith.tokenizers import ByteTokenizer


def _gen() -> Generator:
    model = DecoderLM(
        DecoderConfig(vocab_size=259, block_size=16, embed_dim=32, num_layers=2,
                      num_heads=2, ff_dim=64, dropout=0.0)
    )
    return Generator(model, ByteTokenizer())


# --- basics ---------------------------------------------------------------


def test_generate_ids_extends_by_max_new_tokens():
    out = _gen().generate_ids(torch.zeros(1, 3, dtype=torch.long), max_new_tokens=7)
    assert out.shape == (1, 10)


def test_generate_returns_string():
    assert isinstance(_gen().generate("hello", max_new_tokens=5), str)


def test_empty_prompt_still_generates():
    assert isinstance(_gen().generate("", max_new_tokens=5), str)


def test_greedy_is_deterministic():
    gen = _gen()
    x = torch.zeros(1, 2, dtype=torch.long)
    a = gen.generate_ids(x, max_new_tokens=8, temperature=0.0)
    b = gen.generate_ids(x, max_new_tokens=8, temperature=0.0)
    assert torch.equal(a, b)


# --- the KV-cache must not change results ---------------------------------


def test_kv_cache_matches_full_forward():
    """Incremental (cached) logits must match a full recompute at the last position."""
    from zenith.models import KVCache

    gen = _gen()
    model = gen.model.eval()
    x = torch.randint(0, 259, (1, 6))

    full_logits = model(x)[:, -1, :]

    cache = KVCache(model.config.num_layers)
    model(x[:, :-1], cache=cache)                 # prefill all but the last token
    cached_logits = model(x[:, -1:], cache=cache)[:, -1, :]

    assert torch.allclose(full_logits, cached_logits, atol=1e-4)


# --- decoding strategies reduce to greedy in their degenerate limits ------


def test_top_k_1_equals_greedy():
    gen = _gen()
    x = torch.zeros(1, 2, dtype=torch.long)
    greedy = gen.generate_ids(x, max_new_tokens=8, temperature=0.0)
    top1 = gen.generate_ids(x, max_new_tokens=8, temperature=1.0, top_k=1)
    assert torch.equal(greedy, top1)


def test_tiny_top_p_equals_greedy():
    gen = _gen()
    x = torch.zeros(1, 2, dtype=torch.long)
    greedy = gen.generate_ids(x, max_new_tokens=8, temperature=0.0)
    nucleus = gen.generate_ids(x, max_new_tokens=8, temperature=1.0, top_p=1e-6)
    assert torch.equal(greedy, nucleus)


def test_num_beams_1_equals_greedy():
    gen = _gen()
    x = torch.zeros(1, 2, dtype=torch.long)
    # Both use the full-forward path so the comparison is exact (no cache float drift).
    greedy = gen.generate_ids(x, max_new_tokens=8, temperature=0.0, use_cache=False)
    beam = gen.beam_search_ids(x, max_new_tokens=8, num_beams=1)
    assert torch.equal(greedy, beam)


# --- validity -------------------------------------------------------------


def test_sampled_tokens_are_in_vocab():
    torch.manual_seed(0)
    gen = _gen()
    out = gen.generate_ids(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=12,
                           temperature=1.0, top_k=40, top_p=0.9)
    assert out.min() >= 0 and out.max() < 259


def test_repetition_penalty_runs_and_stays_valid():
    gen = _gen()
    out = gen.generate_ids(torch.zeros(1, 3, dtype=torch.long), max_new_tokens=8,
                           temperature=0.0, repetition_penalty=1.3)
    assert out.shape == (1, 11) and out.max() < 259


def test_beam_search_shape_and_batch_guard():
    import pytest

    gen = _gen()
    out = gen.beam_search_ids(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=6, num_beams=3)
    assert out.shape == (1, 8)
    with pytest.raises(ValueError):
        gen.beam_search_ids(torch.zeros(2, 2, dtype=torch.long), max_new_tokens=3, num_beams=3)


def test_generation_stops_at_block_size():
    """Cached sampling cannot exceed the model's context window."""
    gen = _gen()  # block_size=16
    out = gen.generate_ids(torch.zeros(1, 4, dtype=torch.long), max_new_tokens=100, temperature=0.0)
    assert out.size(1) <= 16
