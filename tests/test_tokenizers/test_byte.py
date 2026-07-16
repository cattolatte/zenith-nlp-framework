"""Tests for the built-in byte-level tokenizer."""

from zenith.tokenizers import ByteTokenizer


def test_roundtrip_is_lossless_for_ascii():
    tok = ByteTokenizer()
    assert tok.decode(tok.encode("hello world")) == "hello world"


def test_roundtrip_is_lossless_for_unicode():
    tok = ByteTokenizer()
    text = "café — 日本語 — 🌙"
    assert tok.decode(tok.encode(text)) == text


def test_vocab_size_is_bytes_plus_specials():
    assert ByteTokenizer().vocab_size == 260  # 256 bytes + bos/eos/pad/abstain


def test_bos_eos_wrapping():
    tok = ByteTokenizer()
    ids = tok.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id


def test_decode_drops_special_tokens():
    tok = ByteTokenizer()
    assert tok.decode([tok.bos_id, *tok.encode("ok"), tok.eos_id]) == "ok"
