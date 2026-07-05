"""Tests for language-model evaluation (loss & perplexity)."""

import math

import pytest

from zenith.data import CausalLMDataset, encode_corpus
from zenith.evaluation import evaluate, perplexity
from zenith.models import DecoderConfig, DecoderLM
from zenith.tokenizers import ByteTokenizer


def _model_and_dataset():
    tokenizer = ByteTokenizer()
    ids = encode_corpus("the cat sat on the mat. " * 20, tokenizer)
    dataset = CausalLMDataset(ids, block_size=16)
    model = DecoderLM(
        DecoderConfig(vocab_size=tokenizer.vocab_size, block_size=16, embed_dim=32,
                      num_layers=2, num_heads=2, ff_dim=64, dropout=0.0)
    )
    return model, dataset


def test_evaluate_returns_loss_and_perplexity():
    model, dataset = _model_and_dataset()
    metrics = evaluate(model, dataset)
    assert metrics["loss"] > 0 and math.isfinite(metrics["loss"])
    # perplexity = exp(loss), with an overflow guard capping the exponent at 20.
    assert metrics["perplexity"] == pytest.approx(math.exp(min(metrics["loss"], 20.0)), rel=1e-4)
    assert metrics["perplexity"] >= 1.0  # perplexity is always at least 1


def test_perplexity_is_positive_and_finite():
    model, dataset = _model_and_dataset()
    ppl = perplexity(model, dataset)
    assert ppl > 0 and math.isfinite(ppl)
