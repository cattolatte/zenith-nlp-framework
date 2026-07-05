"""Tests for the causal-LM trainer — assert it actually learns on a tiny fixture."""

import json

import torch

from zenith.checkpoint import load_pretrained
from zenith.data import CausalLMDataset, encode_corpus
from zenith.models import DecoderConfig, DecoderLM
from zenith.peft import LoraConfig
from zenith.tokenizers import ByteTokenizer
from zenith.training import CausalLMTrainer, TrainingConfig


def _fixture(dropout: float = 0.0):
    tokenizer = ByteTokenizer()
    ids = encode_corpus("the cat sat on the mat. " * 40, tokenizer)
    dataset = CausalLMDataset(ids, block_size=16)
    model = DecoderLM(
        DecoderConfig(vocab_size=tokenizer.vocab_size, block_size=16, embed_dim=32,
                      num_layers=2, num_heads=2, ff_dim=64, dropout=dropout)
    )
    return tokenizer, dataset, model


def test_training_reduces_loss_and_saves_loadable_checkpoint(tmp_path):
    tokenizer, dataset, model = _fixture()
    save_path = tmp_path / "lm.pt"
    config = TrainingConfig(
        epochs=5, batch_size=16, learning_rate=3e-3, warmup_steps=5, save_path=str(save_path)
    )
    result = CausalLMTrainer(model, tokenizer, config).fit(dataset)
    history = result["history"]

    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert save_path.exists()
    generator = load_pretrained(str(save_path))
    assert isinstance(generator.generate("the ", max_new_tokens=10), str)


def test_gradient_accumulation_still_learns(tmp_path):
    tokenizer, dataset, model = _fixture()
    config = TrainingConfig(
        epochs=5, batch_size=8, learning_rate=3e-3, warmup_steps=5,
        grad_accum_steps=2, save_path=str(tmp_path / "lm.pt"),
    )
    history = CausalLMTrainer(model, tokenizer, config).fit(dataset)["history"]
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_lora_training_freezes_base_and_updates_adapter(tmp_path):
    tokenizer, dataset, model = _fixture()
    base_before = model.blocks[0].attention.qkv.weight.detach().clone()

    save_path = tmp_path / "lora.pt"
    config = TrainingConfig(
        epochs=4, batch_size=16, learning_rate=5e-3, warmup_steps=3,
        use_lora=True, lora=LoraConfig(rank=4, alpha=8), save_path=str(save_path),
    )
    CausalLMTrainer(model, tokenizer, config).fit(dataset)

    # Base weight is frozen; adapter moved away from its zero init.
    qkv = model.blocks[0].attention.qkv
    assert torch.allclose(base_before, qkv.base_layer.weight)
    assert qkv.lora_B.abs().sum() > 0

    # The LoRA checkpoint round-trips into a working generator.
    generator = load_pretrained(str(save_path))
    assert isinstance(generator.generate("the ", max_new_tokens=10), str)


def test_fit_writes_a_run_record(tmp_path):
    tokenizer, dataset, model = _fixture()
    run_dir = tmp_path / "runs"
    config = TrainingConfig(
        epochs=2, batch_size=16, warmup_steps=2,
        record_dir=str(run_dir), save_path=str(tmp_path / "lm.pt"),
    )
    CausalLMTrainer(model, tokenizer, config).fit(dataset)

    assert (run_dir / "run.json").exists()
    assert (run_dir / "samples.txt").exists()  # log_samples defaults to True
    run = json.loads((run_dir / "run.json").read_text())
    assert len(run["metrics"]["history"]) == 2
    assert run["environment"]["seed"] == 0
