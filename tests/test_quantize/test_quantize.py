"""Weight-only int8 quantization: ~4x smaller, output close to full precision."""

import copy

import torch
from torch import nn

from zenith.generation import Generator
from zenith.models import DecoderConfig, DecoderLM
from zenith.quantize import QuantizedLinear, quantize_int8
from zenith.tokenizers import ByteTokenizer


def _model() -> DecoderLM:
    torch.manual_seed(0)
    return DecoderLM(
        DecoderConfig(vocab_size=259, block_size=32, embed_dim=48, num_layers=3,
                      num_heads=4, ff_dim=96, dropout=0.0)
    ).eval()


def test_quantized_linear_is_int8_and_close():
    torch.manual_seed(1)
    linear = nn.Linear(64, 32)
    q = QuantizedLinear.from_linear(linear)
    assert q.weight_int8.dtype == torch.int8
    x = torch.randn(5, 64)
    with torch.no_grad():
        ref, got = linear(x), q(x)
    cos = torch.cosine_similarity(ref.flatten(), got.flatten(), dim=0)
    assert cos > 0.999  # per-row int8 is a very close approximation


def test_quantize_replaces_every_linear():
    model = quantize_int8(_model())
    assert not any(isinstance(m, nn.Linear) for m in model.modules())
    assert any(isinstance(m, QuantizedLinear) for m in model.modules())


def test_quantize_shrinks_weight_memory():
    fp32 = _model()
    q = quantize_int8(copy.deepcopy(fp32))

    def linear_bytes(m):
        total = 0
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                total += mod.weight.numel() * 4
            if isinstance(mod, QuantizedLinear):
                total += mod.weight_int8.numel() + mod.scale.numel() * 4
        return total

    assert linear_bytes(q) < linear_bytes(fp32) / 3  # ~4x, comfortably under 3x


def test_quantized_model_still_generates():
    fp32 = _model()
    q = quantize_int8(copy.deepcopy(fp32)).eval()
    x = torch.randint(0, 259, (1, 12))
    with torch.no_grad():
        cos = torch.cosine_similarity(fp32(x).flatten(), q(x).flatten(), dim=0)
    assert cos > 0.9  # error compounds across layers but stays close

    out = Generator(q, ByteTokenizer()).generate_ids(x, max_new_tokens=10, temperature=0.0)
    assert out.size(1) == x.size(1) + 10
