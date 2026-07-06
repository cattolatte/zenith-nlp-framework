"""Optional interoperability with the Polaris NLP engine.

Design Principles
-----------------
Zenith and Polaris are independent siblings — Polaris encodes/classifies text,
Zenith generates it — and neither depends on the other. This module is the *bridge*
you reach for only if you want to use both: it adapts a Polaris tokenizer to
Zenith's tokenizer interface, so a Zenith decoder can train and generate over a
Polaris (encoder-side) vocabulary.

Polaris is imported lazily and only when you actually build a tokenizer, so
``import zenith`` — and even ``import zenith.interop`` — never require it. Install
it with the optional extra: ``pip install zenith-nlp[polaris]``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PolarisTokenizer"]


class PolarisTokenizer:
    """Adapt a Polaris tokenizer to Zenith's tokenizer interface.

    Exposes ``encode`` / ``decode`` / ``token_bytes`` and
    ``bos_id`` / ``eos_id`` / ``pad_id`` / ``vocab_size`` — everything a
    :class:`~zenith.generation.Generator` and :class:`~zenith.training.CausalLMTrainer`
    need — so a Zenith decoder is a drop-in consumer of a Polaris vocabulary.

    Construct from an existing Polaris tokenizer, or train one with
    :meth:`train`.

    Examples
    --------
    >>> from zenith.interop import PolarisTokenizer   # doctest: +SKIP
    >>> tok = PolarisTokenizer.train(["the quick brown fox " * 20], vocab_size=300)
    >>> tok.decode(tok.encode("the quick brown fox"))
    'the quick brown fox'
    """

    def __init__(self, polaris_tokenizer: Any) -> None:
        self._tok = polaris_tokenizer
        vocab = polaris_tokenizer.vocabulary
        self._vocab = vocab
        self.vocab_size: int = vocab.size
        # Zenith needs valid special-token ids; reuse Polaris' if present, else 0.
        self.pad_id: int = vocab.pad_id if vocab.pad_id is not None else 0
        self.bos_id: int = self.pad_id
        self.eos_id: int = vocab.unk_id if vocab.unk_id is not None else 0

    @classmethod
    def train(cls, texts: list[str], vocab_size: int, **kwargs: Any) -> "PolarisTokenizer":
        """Train a Polaris byte-pair tokenizer on ``texts`` and wrap it.

        Polaris' ``train_bpe`` expects token *sequences*; each text is fed as its
        sequence of characters (a from-scratch char-level BPE).
        """
        try:
            from polaris.tokenizers import train_bpe
        except ImportError as exc:  # pragma: no cover - exercised without the extra
            raise ImportError(
                "PolarisTokenizer requires the optional 'polaris' extra: "
                "pip install zenith-nlp[polaris]"
            ) from exc
        tokenizer = train_bpe([list(text) for text in texts], vocab_size=vocab_size, **kwargs)
        return cls(tokenizer)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self._tok.encode(text).ids)
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode([int(i) for i in ids])

    def token_bytes(self, token: int) -> bytes:
        """Best-effort raw bytes for one token (for streaming).

        Polaris marks word boundaries with an end-of-word token; we render that as
        a space. The authoritative reconstruction is :meth:`decode`.
        """
        try:
            text = self._vocab.lookup_token(int(token))
        except Exception:
            return b""
        return text.replace("</w>", " ").encode("utf-8")
