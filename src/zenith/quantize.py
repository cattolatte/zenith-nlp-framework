"""Weight-only int8 quantization for smaller inference, from scratch.

Transformer weight is almost all in the linear layers, so storing those as int8
(one byte) instead of fp32 (four) shrinks the model ~4×. Each linear's weight is
quantized **per output channel** with a symmetric scale — ``w ≈ w_int8 * scale`` —
and dequantized on the fly in the forward pass. Output matches full precision to
within quantization error.

This is an inference-only transform: the returned model is not trainable, and any
weight tying is materialized into the quantized copies. On CPU/MPS it mainly saves
memory (there is no int8 matmul kernel here — the weight is dequantized before a
normal matmul), which is the honest scope.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["QuantizedLinear", "quantize_int8"]


class QuantizedLinear(nn.Module):
    """A ``nn.Linear`` with int8 weights + per-output-channel fp32 scales."""

    weight_int8: torch.Tensor
    scale: torch.Tensor

    def __init__(self, weight_int8: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinear":
        w = linear.weight.detach()
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0  # per output row
        w_int8 = torch.round(w / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        return cls(w_int8, scale, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype).unsqueeze(1)
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        out, inp = self.weight_int8.shape
        return f"in_features={inp}, out_features={out}, dtype=int8"


def quantize_int8(model: nn.Module) -> nn.Module:
    """Replace every ``nn.Linear`` in ``model`` with an int8 :class:`QuantizedLinear`.

    Mutates ``model`` in place and returns it. Inference only — the model is no longer
    trainable, and tied weights are materialized into the quantized copies.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            setattr(model, name, QuantizedLinear.from_linear(child))
        else:
            quantize_int8(child)
    return model
