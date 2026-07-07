"""Speculative decoding: it must produce byte-identical output to greedy."""

import torch
import pytest

from zenith.generation import Generator, SpeculativeStats
from zenith.models import DecoderConfig, DecoderLM
from zenith.tokenizers import ByteTokenizer


def _model(seed: int, *, embed_dim: int = 32, layers: int = 2) -> DecoderLM:
    torch.manual_seed(seed)
    return DecoderLM(
        DecoderConfig(vocab_size=259, block_size=64, embed_dim=embed_dim, num_layers=layers,
                      num_heads=2, ff_dim=embed_dim * 2, dropout=0.0)
    ).eval()


def _prompt() -> torch.Tensor:
    return torch.tensor([[1, 40, 41, 42, 43]], dtype=torch.long)


# --- exactness: the core guarantee ---------------------------------------


def test_speculative_matches_greedy_with_self_draft():
    """Draft == target ⇒ every proposal is accepted and output == greedy."""
    target = _model(0)
    gen = Generator(target, ByteTokenizer())
    prompt = _prompt()
    greedy = gen.generate_ids(prompt, max_new_tokens=30, temperature=0.0)
    spec, stats = gen.speculative_generate_ids(prompt, target, max_new_tokens=30, lookahead=4)
    assert torch.equal(greedy, spec)
    assert stats.acceptance_rate == 1.0  # self-draft is always right
    assert stats.tokens == 30


def test_speculative_matches_greedy_with_weaker_draft():
    """A different (weaker) draft causes rejections but output is still identical."""
    target = _model(0)
    draft = _model(1, embed_dim=16, layers=1)  # different weights ⇒ real disagreements
    gen = Generator(target, ByteTokenizer())
    prompt = _prompt()
    greedy = gen.generate_ids(prompt, max_new_tokens=30, temperature=0.0)
    spec, stats = gen.speculative_generate_ids(prompt, draft, max_new_tokens=30, lookahead=4)
    assert torch.equal(greedy, spec)
    assert 0.0 <= stats.acceptance_rate <= 1.0
    assert stats.target_forwards <= stats.tokens  # never slower than greedy in forwards


def test_speculative_varying_lookahead_all_match_greedy():
    target = _model(2)
    draft = _model(3, embed_dim=16, layers=1)
    gen = Generator(target, ByteTokenizer())
    prompt = _prompt()
    greedy = gen.generate_ids(prompt, max_new_tokens=24, temperature=0.0)
    for k in (1, 2, 3, 8):
        spec, _ = gen.speculative_generate_ids(prompt, draft, max_new_tokens=24, lookahead=k)
        assert torch.equal(greedy, spec), f"mismatch at lookahead={k}"


# --- guards & stats ------------------------------------------------------


def test_speculative_rejects_batch():
    target = _model(0)
    gen = Generator(target, ByteTokenizer())
    with pytest.raises(ValueError):
        gen.speculative_generate_ids(torch.zeros(2, 4, dtype=torch.long), target)


def test_speculative_rejects_bad_lookahead():
    target = _model(0)
    gen = Generator(target, ByteTokenizer())
    with pytest.raises(ValueError):
        gen.speculative_generate_ids(_prompt(), target, lookahead=0)


def test_speculative_stops_at_block_size():
    target = _model(0)
    gen = Generator(target, ByteTokenizer())
    # block_size is 64; ask for far more than fits and confirm it stays in-bounds.
    out, stats = gen.speculative_generate_ids(_prompt(), target, max_new_tokens=500, lookahead=4)
    assert out.size(1) <= target.config.block_size
    assert stats.speedup >= 1.0  # self-draft: many tokens per target forward


def test_stats_speedup_and_acceptance():
    s = SpeculativeStats(tokens=20, target_forwards=5, proposed=16, accepted=12)
    assert s.speedup == 4.0
    assert s.acceptance_rate == 0.75
