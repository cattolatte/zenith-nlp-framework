"""Tests for LoRA parameter-efficient fine-tuning."""

import torch
import torch.nn as nn
import pytest

from zenith.models import DecoderConfig, DecoderLM
from zenith.peft import (
    LoraConfig,
    LoRALinear,
    count_trainable_parameters,
    find_linear_modules,
    inject_lora,
    lora_state_dict,
)


def _tiny_model() -> DecoderLM:
    return DecoderLM(
        DecoderConfig(vocab_size=259, block_size=16, embed_dim=32, num_layers=2,
                      num_heads=2, ff_dim=64, dropout=0.0)
    )


# --- LoRALinear -----------------------------------------------------------


def test_lora_linear_starts_as_no_op():
    base = nn.Linear(8, 4)
    wrapped = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(3, 8)
    assert torch.allclose(wrapped(x), base(x))


def test_lora_linear_perturbs_output_once_b_is_nonzero():
    wrapped = LoRALinear(nn.Linear(8, 4), rank=2, alpha=4)
    nn.init.normal_(wrapped.lora_B)
    x = torch.randn(3, 8)
    assert not torch.allclose(wrapped(x), wrapped.base_layer(x))


def test_lora_rejects_non_positive_rank():
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 4), rank=0, alpha=1)


# --- injection into the decoder ------------------------------------------


def test_inject_targets_attention_projections_by_default():
    model = _tiny_model()
    inject_lora(model, LoraConfig(rank=4, alpha=8))
    assert isinstance(model.blocks[0].attention.qkv, LoRALinear)
    assert isinstance(model.blocks[0].attention.proj, LoRALinear)
    # Feed-forward linears are not targeted by default.
    assert not isinstance(model.blocks[0].feed_forward[0], LoRALinear)


def test_inject_freezes_all_but_lora():
    model = _tiny_model()
    inject_lora(model, LoraConfig(rank=4, alpha=8))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_A" in n or "lora_B" in n for n in trainable)


def test_inject_reduces_trainable_parameter_count():
    model = _tiny_model()
    _, total = count_trainable_parameters(model)
    inject_lora(model, LoraConfig(rank=4, alpha=8))
    trainable, _ = count_trainable_parameters(model)
    assert 0 < trainable < total


def test_inject_raises_when_nothing_matches():
    with pytest.raises(ValueError):
        inject_lora(_tiny_model(), LoraConfig(target_modules=("does_not_exist",)))


def test_lora_state_dict_has_only_adapter_weights():
    model = _tiny_model()
    inject_lora(model, LoraConfig(rank=4, alpha=8))
    sd = lora_state_dict(model)
    assert sd and all("lora_A" in k or "lora_B" in k for k in sd)


def test_find_linear_modules_lists_decoder_projections():
    names = find_linear_modules(_tiny_model())
    assert "blocks.0.attention.qkv" in names
    assert "blocks.0.attention.proj" in names
