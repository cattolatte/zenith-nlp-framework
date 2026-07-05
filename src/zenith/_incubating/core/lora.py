"""Frozen copy of the original LoRA layer used by the archived generative models.

The canonical, maintained LoRA adapter now lives in ``zenith.peft``. This copy is
kept only so the frozen ``_incubating`` models remain importable in isolation.
"""

import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    A LoRA (Low-Rank Adaptation) layer that wraps a linear layer.
    """
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.original_layer = original_layer

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Freeze the original layer
        self.original_layer.weight.requires_grad = False

        # Create the low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.Tensor(rank, in_features))
        self.lora_B = nn.Parameter(torch.Tensor(out_features, rank))

        self.scaling = alpha / rank

        # Initialize the new matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        # Original forward pass
        original_output = self.original_layer(x)

        # LoRA adaptation
        lora_output = (self.lora_B @ self.lora_A) * self.scaling

        return original_output + (x @ lora_output.T)
