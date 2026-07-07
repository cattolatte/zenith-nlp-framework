"""Byte-level Byte-Pair Encoding (BPE) tokenizer, implemented from scratch.

Design Principles
-----------------
A learned subword tokenizer that is also *complete*: it starts from the 256 byte
values (so any string round-trips losslessly, never an unknown token) and learns
merges of frequent adjacent pairs up to a target vocabulary size — the GPT-2-style
byte-level BPE, written plainly.

The algorithm is the classic one (count adjacent pairs, merge the most frequent,
repeat). Training is **vectorized with numpy** — each round counts all pairs and
applies the chosen merge over the whole corpus with array operations rather than
Python loops, which is dramatically faster on a real corpus (numpy is imported
lazily inside ``train`` only, so encode/decode stay dependency-light). Losslessness
holds regardless of merge quality — merges are just reversible byte concatenations —
which is exactly what the round-trip test checks.
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

    _PAIR_SHIFT = 1 << 21  # encode a (left, right) id pair as left * SHIFT + right

    def train(self, texts: list[str], vocab_size: int) -> "BPETokenizer":
        """Learn merges from ``texts`` up to ``vocab_size`` total tokens (vectorized)."""
        if vocab_size < 256 + 3:
            raise ValueError(f"vocab_size must be at least {256 + 3}, got {vocab_size}")
        import numpy as np  # lazy: only training needs it, keeps encode/decode light

        num_merges = vocab_size - 256 - 3
        self._merges = {}
        self._vocab = {i: bytes([i]) for i in range(256)}

        # One flat array of all byte ids, with -1 sentinels between texts so no merge
        # ever spans a boundary. Merges rewrite this array in place, vectorized.
        flat: list[int] = []
        for text in texts:
            flat.extend(text.encode("utf-8"))
            flat.append(-1)
        arr = np.array(flat or [-1], dtype=np.int64)
        shift = self._PAIR_SHIFT
        next_id = 256

        for _ in range(num_merges):
            left, right = arr[:-1], arr[1:]
            valid = (left >= 0) & (right >= 0)
            if not valid.any():
                break
            keys = left[valid] * shift + right[valid]
            uniq, counts = np.unique(keys, return_counts=True)
            best = int(uniq[counts.argmax()])
            a, b = best // shift, best % shift

            # Non-overlapping left-to-right merge of the (a, b) pair.
            matches = np.nonzero((arr[:-1] == a) & (arr[1:] == b))[0]
            kept, prev = [], -10
            for m in matches.tolist():
                if m > prev + 1:  # previous merge consumed m-1's slot
                    kept.append(m)
                    prev = m
            kept_arr = np.array(kept, dtype=np.int64)
            arr[kept_arr] = next_id
            drop = np.zeros(arr.shape[0], dtype=bool)
            drop[kept_arr + 1] = True  # the second token of each merged pair
            arr = arr[~drop]

            self._merges[(a, b)] = next_id
            self._vocab[next_id] = self._vocab[a] + self._vocab[b]
            next_id += 1

        self._finalize()
        return self

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
