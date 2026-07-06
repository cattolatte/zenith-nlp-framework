"""Tests for the decoder-only language model."""

import torch
import pytest

from zenith.models import DecoderConfig, DecoderLM


def _tiny() -> DecoderLM:
    return DecoderLM(
        DecoderConfig(vocab_size=259, block_size=16, embed_dim=32, num_layers=2, num_heads=2, ff_dim=64)
    )


def test_forward_output_shape():
    model = _tiny()
    logits = model(torch.zeros(4, 10, dtype=torch.long))
    assert logits.shape == (4, 10, 259)


def test_embedding_and_head_weights_are_tied():
    model = _tiny()
    assert model.lm_head.weight is model.token_embedding.weight


def test_attention_is_causal():
    """Changing the last token must not affect earlier positions' logits."""
    model = _tiny().eval()
    x = torch.randint(0, 259, (1, 12))
    with torch.no_grad():
        out1 = model(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % 259
        out2 = model(x2)
    assert torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-5)


def test_rejects_sequence_longer_than_block_size():
    model = _tiny()
    with pytest.raises(ValueError):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_num_parameters_positive():
    assert _tiny().num_parameters() > 0


def test_gpt2_init_scales_residual_projections():
    """Weights init ~N(0,0.02); residual output projections are scaled down."""
    model = DecoderLM(
        DecoderConfig(vocab_size=259, block_size=16, embed_dim=64, num_layers=4,
                      num_heads=4, ff_dim=128)
    )
    qkv_std = model.blocks[0].attention.qkv.weight.std().item()
    proj_std = model.blocks[0].attention.proj.weight.std().item()
    assert 0.01 < qkv_std < 0.03  # standard 0.02 init
    assert proj_std < qkv_std  # residual proj scaled by 1/sqrt(2*n_layers)
