"""Byte-level tokenizer — Zenith's built-in, dependency-free text encoder.

Design Principles
-----------------
Phase 1 needs a tokenizer that *always works*, on any text, with no training step
and no external dependency — so Zenith can stand entirely on its own. A byte-level
vocabulary (UTF-8 bytes → ids) is the simplest thing that is also complete: every
possible string round-trips losslessly, the vocabulary is a fixed 256 plus a few
reserved special tokens, and there is nothing to fit.

A learned subword tokenizer (BPE), and an optional adapter to Polaris'
tokenizers, are deliberately deferred to a later slice — this one is the floor,
not the ceiling.
"""

from __future__ import annotations

__all__ = ["ByteTokenizer"]

# Reserved ids sit above the 256 byte values.
_BOS = 256  # beginning of sequence
_EOS = 257  # end of sequence
_PAD = 258  # padding


class ByteTokenizer:
    """Encode text as UTF-8 byte ids, plus a handful of special tokens.

    Attributes
    ----------
    vocab_size : int
        Number of distinct ids (256 bytes + special tokens).

    Examples
    --------
    >>> tok = ByteTokenizer()
    >>> tok.decode(tok.encode("hi"))
    'hi'
    >>> tok.vocab_size
    259
    """

    bos_id = _BOS
    eos_id = _EOS
    pad_id = _PAD

    def __init__(self) -> None:
        self.vocab_size = 259

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode ``text`` into a list of byte ids, optionally BOS/EOS-wrapped."""
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode ids back to text, dropping special tokens.

        Invalid UTF-8 byte sequences are replaced rather than raising, so partial
        generations always render.
        """
        byte_values = bytes(i for i in ids if i < 256)
        return byte_values.decode("utf-8", errors="replace")

    def token_bytes(self, token: int) -> bytes:
        """The raw bytes a single token expands to (for streaming)."""
        return bytes([token]) if token < 256 else b""
