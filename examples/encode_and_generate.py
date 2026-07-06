"""Sibling demo: Zenith generates over a Polaris tokenizer.

Polaris (the encoder-side sibling) owns tokenization; Zenith owns generation.
Neither depends on the other — this script just uses both, linked through a shared
vocabulary. Requires the optional extra:

    pip install zenith-nlp[polaris]
    python examples/encode_and_generate.py
"""

from __future__ import annotations

from pathlib import Path

from zenith.data import CausalLMDataset, encode_corpus, train_val_split
from zenith.generation import Generator
from zenith.interop import PolarisTokenizer
from zenith.models import DecoderConfig, DecoderLM
from zenith.training import CausalLMTrainer, TrainingConfig


def main() -> None:
    corpus = Path("data/tiny_corpus.txt").read_text(encoding="utf-8")

    # 1. Polaris builds the vocabulary (a from-scratch char-level BPE).
    tokenizer = PolarisTokenizer.train([corpus], vocab_size=512)
    print(f"Polaris tokenizer: vocab_size={tokenizer.vocab_size}")

    # 2. Zenith trains a decoder over that shared vocabulary.
    ids = encode_corpus(corpus, tokenizer)
    train_ids, val_ids = train_val_split(ids, 0.1)
    block_size = 48
    train_ds = CausalLMDataset(train_ids, block_size)
    val_ds = CausalLMDataset(val_ids, block_size) if val_ids.numel() > block_size + 1 else None

    model = DecoderLM(
        DecoderConfig(vocab_size=tokenizer.vocab_size, block_size=block_size,
                      embed_dim=128, num_layers=3, num_heads=4, ff_dim=512, dropout=0.1)
    )
    config = TrainingConfig(epochs=30, batch_size=16, learning_rate=3e-3,
                            log_samples=False, save_path="polaris-zenith-lm.pt")
    CausalLMTrainer(model, tokenizer, config).fit(train_ds, val_ds)

    # 3. Zenith generates.
    print("\n--- generated ---")
    print(Generator(model, tokenizer).generate("The", max_new_tokens=200, temperature=0.8))


if __name__ == "__main__":
    main()
