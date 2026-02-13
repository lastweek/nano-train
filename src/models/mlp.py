"""
MLP implementation for MVP.

Phase 7 will add MoE support.
"""

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.layers import Linear, Dropout, GELU


class MLP(nn.Module):
    """Standard feed-forward network (FFN)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        # NATIVE: Our Linear in src/layers.py
        # ORIGINAL: torch.nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc1 = Linear(config.hidden_size, config.intermediate_size)

        # NATIVE: Our Linear in src/layers.py
        # ORIGINAL: torch.nn.Linear(config.intermediate_size, config.hidden_size)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size)

        # NATIVE: Our Dropout in src/layers.py
        # ORIGINAL: torch.nn.Dropout(config.dropout)
        self.dropout = Dropout(config.dropout)

        # NATIVE: Our GELU in src/layers.py
        # ORIGINAL: torch.nn.GELU()
        self.act = GELU()

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)

        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        # Project to intermediate size for non-linear mixing.
        x = self.fc1(x)  # (B, S, intermediate)
        x = self.act(x)
        x = self.dropout(x)
        # Project back to hidden size.
        x = self.fc2(x)  # (B, S, hidden)
        x = self.dropout(x)
        return x
