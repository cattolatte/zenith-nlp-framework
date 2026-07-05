"""Byte-level Byte-Pair Encoding (BPE) tokenizer, implemented from scratch.

Design Principles
-----------------
A learned subword tokenizer that is also *complete*: it starts from the 256 byte
values (so any string round-trips losslessly, never an unknown token) and learns
merges of frequent adjacent pairs up to a target vocabulary size — the GPT-2-style
byte-level BPE, written plainly.

The algorithm is deliberately the obvious one (count pairs, merge the most
frequent, repeat); it is O(merges × corpus), which is fine at the scale this
library targets. Losslessness holds regardless of merge quality — merges are just
reversible byte concatenations — which is exactly what the round-trip test checks.
"""

from __future__ import annotations

from typing import Any

__all__ = ["BPETokenizer"]


class BPETokenizer:
    """A trainable byte-level BPE tokenizer.

    Interface-compatible with :class:`ByteTokenizer` (``encode``/``decode``/
    ``token_bytes`` and ``bos_id``/``eos_id``/``pad_id``/``vocab_size``), so it is a
    drop-in for training, generation and serving.

    Examples
    --------
    >>> tok = BPETokenizer().train(["banana " * 10], vocab_size=300)
    >>> tok.decode(tok.encode("banana")) == "banana"
    True
    """

    def __init__(self) -> None:
        self._merges: dict[tuple[int, int], int] = {}
        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._finalize()

    # -- special tokens sit above the base bytes + learned merges ------------

    def _finalize(self) -> None:
        base = 256 + len(self._merges)
        self.bos_id = base
        self.eos_id = base + 1
        self.pad_id = base + 2
        for special in (self.bos_id, self.eos_id, self.pad_id):
            self._vocab[special] = b""
        self.vocab_size = base + 3

    # -- training ------------------------------------------------------------

    def train(self, texts: list[str], vocab_size: int) -> "BPETokenizer":
        """Learn merges from ``texts`` up to ``vocab_size`` total tokens."""
        if vocab_size < 256 + 3:
            raise ValueError(f"vocab_size must be at least {256 + 3}, got {vocab_size}")
        num_merges = vocab_size - 256 - 3

        sequences = [list(text.encode("utf-8")) for text in texts]
        self._merges = {}
        self._vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        for _ in range(num_merges):
            pairs = self._count_pairs(sequences)
            if not pairs:
                break
            best = max(pairs, key=lambda p: pairs[p])
            sequences = [self._merge(seq, best, next_id) for seq in sequences]
            self._merges[best] = next_id
            self._vocab[next_id] = self._vocab[best[0]] + self._vocab[best[1]]
            next_id += 1

        self._finalize()
        return self

    @staticmethod
    def _count_pairs(sequences: list[list[int]]) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for seq in sequences:
            for pair in zip(seq, seq[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    # -- encode / decode -----------------------------------------------------

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode ``text`` into token ids by replaying the learned merges."""
        ids = list(text.encode("utf-8"))
        for pair, new_id in self._merges.items():
            ids = self._merge(ids, pair, new_id)
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text (special tokens drop out)."""
        data = b"".join(self._vocab.get(i, b"") for i in ids)
        return data.decode("utf-8", errors="replace")

    def token_bytes(self, token: int) -> bytes:
        """The raw bytes a single token expands to (for streaming)."""
        return self._vocab.get(token, b"")

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "bpe",
            "merges": [[a, b, new_id] for (a, b), new_id in self._merges.items()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BPETokenizer":
        tok = cls()
        tok._merges = {}
        tok._vocab = {i: bytes([i]) for i in range(256)}
        for a, b, new_id in data["merges"]:
            tok._merges[(a, b)] = new_id
            tok._vocab[new_id] = tok._vocab[a] + tok._vocab[b]
        tok._finalize()
        return tok
