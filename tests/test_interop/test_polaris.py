"""Tests for the optional Polaris tokenizer interop (skipped without the extra)."""

import pytest

pytest.importorskip("polaris", reason="requires the optional [polaris] extra")

from zenith.generation import Generator  # noqa: E402
from zenith.interop import PolarisTokenizer  # noqa: E402
from zenith.models import DecoderConfig, DecoderLM  # noqa: E402


def _tokenizer() -> PolarisTokenizer:
    return PolarisTokenizer.train(["the quick brown fox " * 20], vocab_size=300)


def test_adapter_exposes_zenith_interface():
    tok = _tokenizer()
    assert tok.vocab_size > 0
    assert isinstance(tok.encode("the quick"), list)
    assert isinstance(tok.bos_id, int) and isinstance(tok.eos_id, int)


def test_roundtrip_on_trained_text():
    tok = _tokenizer()
    assert tok.decode(tok.encode("the quick brown fox")) == "the quick brown fox"


def test_zenith_decoder_generates_over_polaris_vocab():
    tok = _tokenizer()
    model = DecoderLM(
        DecoderConfig(vocab_size=tok.vocab_size, block_size=16, embed_dim=32, num_layers=2,
                      num_heads=2, ff_dim=64, dropout=0.0)
    )
    generator = Generator(model, tok)
    assert isinstance(generator.generate("the", max_new_tokens=8), str)
    assert isinstance("".join(generator.stream("the", max_new_tokens=8)), str)
