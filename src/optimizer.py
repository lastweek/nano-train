"""
Optimizer implementation for MVP.
"""

import torch
from torch.optim import AdamW


def create_optimizer(model, learning_rate, weight_decay):
    """Create AdamW optimizer with parameter groups."""
    # Separate parameters with and without weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No decay for biases and layer norms
        if 'bias' in name or 'ln_' in name or 'layer_norm' in name:
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
