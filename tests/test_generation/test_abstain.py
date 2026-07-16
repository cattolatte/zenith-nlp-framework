"""The reserved <abstain> token: reservation, stability, round-trip, detection."""

import torch

from zenith.generation import Generator
from zenith.models import DecoderConfig, DecoderLM
from zenith.tokenizers import BPETokenizer, ByteTokenizer


def test_byte_abstain_id_is_reserved_and_stable():
    tok = ByteTokenizer()
    assert tok.abstain_id == 259
    assert tok.vocab_size == 260
    assert tok.abstain_id not in {tok.bos_id, tok.eos_id, tok.pad_id}
    assert tok.abstain_id >= 256  # never collides with a byte (content) id


def test_bpe_abstain_id_reserved_above_content():
    tok = BPETokenizer().train(["the quick brown fox " * 20], vocab_size=300)
    assert tok.abstain_id == tok.vocab_size - 1
    assert tok.abstain_id not in {tok.bos_id, tok.eos_id, tok.pad_id}
    assert tok.abstain_id >= 256


def test_bpe_abstain_id_survives_serialization():
    tok = BPETokenizer().train(["hello world " * 20], vocab_size=290)
    rebuilt = BPETokenizer.from_dict(tok.to_dict())
    assert rebuilt.abstain_id == tok.abstain_id
    assert rebuilt.vocab_size == tok.vocab_size


def test_abstain_roundtrips_without_corrupting_text():
    tok = ByteTokenizer()
    ids = [*tok.encode("hi"), tok.abstain_id]
    assert tok.decode(ids) == "hi"  # abstain drops out like the other specials


def test_generator_detects_abstention():
    torch.manual_seed(0)
    model = DecoderLM(
        DecoderConfig(
            vocab_size=260,
            block_size=16,
            embed_dim=32,
            num_layers=2,
            num_heads=2,
            ff_dim=64,
            dropout=0.0,
        )
    ).eval()
    gen = Generator(model, ByteTokenizer())
    assert gen.abstained(torch.tensor([[10, gen.tokenizer.abstain_id, 11]]))
    assert not gen.abstained(torch.tensor([[10, 11, 12]]))
