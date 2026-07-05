"""Tests for the from-scratch BPE tokenizer."""

from zenith.checkpoint import load_pretrained, save_checkpoint
from zenith.models import DecoderConfig, DecoderLM
from zenith.tokenizers import BPETokenizer


def _trained() -> BPETokenizer:
    return BPETokenizer().train(["the quick brown fox " * 20], vocab_size=320)


def test_roundtrip_is_lossless():
    tok = _trained()
    for text in ["the quick brown fox", "hello world", "café — 日本語 — 🌙", ""]:
        assert tok.decode(tok.encode(text)) == text


def test_training_learns_merges_and_grows_vocab():
    tok = _trained()
    assert tok.vocab_size > 256 + 3  # some merges were learned
    assert tok.vocab_size <= 320


def test_untrained_is_pure_bytes():
    tok = BPETokenizer()
    assert tok.vocab_size == 256 + 3
    assert tok.encode("hi") == [ord("h"), ord("i")]


def test_special_tokens_wrap_and_drop():
    tok = _trained()
    ids = tok.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == "hi"


def test_to_from_dict_roundtrip():
    tok = _trained()
    rebuilt = BPETokenizer.from_dict(tok.to_dict())
    assert rebuilt.vocab_size == tok.vocab_size
    assert rebuilt.encode("the quick brown fox") == tok.encode("the quick brown fox")


def test_token_bytes_reconstructs_input():
    tok = _trained()
    text = "the quick"
    ids = tok.encode(text)
    assert b"".join(tok.token_bytes(i) for i in ids) == text.encode("utf-8")


def test_bpe_checkpoint_roundtrips(tmp_path):
    tok = _trained()
    model = DecoderLM(
        DecoderConfig(vocab_size=tok.vocab_size, block_size=32, embed_dim=32, num_layers=2,
                      num_heads=2, ff_dim=64, dropout=0.0)
    )
    path = tmp_path / "lm.pt"
    save_checkpoint(model, tok, str(path))

    generator = load_pretrained(str(path))
    assert isinstance(generator.tokenizer, BPETokenizer)
    assert generator.tokenizer.vocab_size == tok.vocab_size
    assert isinstance(generator.generate("the", max_new_tokens=5), str)
