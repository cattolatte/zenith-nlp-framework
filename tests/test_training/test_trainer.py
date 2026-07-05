"""Tests for the causal-LM trainer — assert it actually learns on a tiny fixture."""

import torch

from zenith.checkpoint import load_pretrained
from zenith.data import CausalLMDataset, encode_corpus
from zenith.models import DecoderConfig, DecoderLM
from zenith.tokenizers import ByteTokenizer
from zenith.training import CausalLMTrainer, TrainingConfig


def test_training_reduces_loss_and_saves_loadable_checkpoint(tmp_path):
    tokenizer = ByteTokenizer()
    # A short, highly repetitive corpus a tiny model can memorise quickly.
    ids = encode_corpus("the cat sat on the mat. " * 40, tokenizer)
    dataset = CausalLMDataset(ids, block_size=16)

    model = DecoderLM(
        DecoderConfig(vocab_size=tokenizer.vocab_size, block_size=16, embed_dim=32,
                      num_layers=2, num_heads=2, ff_dim=64, dropout=0.0)
    )
    save_path = tmp_path / "lm.pt"
    config = TrainingConfig(
        epochs=5, batch_size=16, learning_rate=3e-3, warmup_steps=5, save_path=str(save_path)
    )

    result = CausalLMTrainer(model, tokenizer, config).fit(dataset)
    history = result["history"]

    # Learning behaviour: loss should fall meaningfully over training.
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert save_path.exists()

    # The checkpoint round-trips into a working generator.
    generator = load_pretrained(str(save_path))
    assert isinstance(generator.generate("the ", max_new_tokens=10), str)
