"""
Optimizer implementation for MVP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Create AdamW optimizer with parameter groups."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    # Separate parameters with and without weight decay
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No decay for biases and layer norms
        lower_name = name.lower()
        if "bias" in lower_name or "ln_" in lower_name or "layer_norm" in lower_name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {
            'params': decay_params,
            'weight_decay': weight_decay,
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    return optimizer
