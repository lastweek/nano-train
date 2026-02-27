"""
MLP implementation for MVP.

Supports tensor parallelism via TPConfig parameter.
Phase 7 will add MoE support.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.config import TPConfig
from src.layers import ColumnParallelLinear
from src.layers import Dropout
from src.layers import GELU
from src.layers import Linear
from src.layers import RowParallelLinear


class MLP(nn.Module):
    """
    Feed-forward network with optional tensor parallelism.

    If tp_config.enabled is True:
        - fc1 uses ColumnParallelLinear (split intermediate dimension)
        - fc2 uses RowParallelLinear (split intermediate dimension)
        Communication: 1 all-reduce per forward pass

    If tp_config.enabled is False (default):
        - Uses standard Linear layers
        - No communication
    """

    def __init__(self, config: ModelConfig, tp_config: Optional[TPConfig] = None) -> None:
        super().__init__()
        tp_config = tp_config or TPConfig()

        self.tp_enabled = tp_config.enabled
        self.tp_rank = tp_config.rank
        self.tp_size = tp_config.size
        self.tp_group = tp_config.group

        if self.tp_enabled:
            # For TP: each GPU gets intermediate_size // tp_size
            if config.intermediate_size % self.tp_size != 0:
                raise ValueError(
                    f"intermediate_size ({config.intermediate_size}) must be divisible by "
                    f"tp_size ({self.tp_size})"
                )

            # ColumnParallelLinear expects GLOBAL out_features and returns local shard.
            self.fc1 = ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                tp_group=self.tp_group,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
            )

            # RowParallelLinear expects GLOBAL in_features and consumes local shard.
            self.fc2 = RowParallelLinear(
                config.intermediate_size,
                config.hidden_size,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                tp_group=self.tp_group,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
            )
        else:
            # Standard Linear layers
            self.fc1 = Linear(
                config.hidden_size,
                config.intermediate_size,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
            )
            self.fc2 = Linear(
                config.intermediate_size,
                config.hidden_size,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
            )

        self.dropout = Dropout(config.dropout)
        self.act = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_size)

        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        if x.dim() != 3:
            raise ValueError("x must have shape [batch_size, seq_len, hidden_size]")

        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)  # All-reduce inside if TP mode
        x = self.dropout(x)
        return x
