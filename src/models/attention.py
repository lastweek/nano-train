"""
Multi-head attention implementation for MVP.

Phase 5 will add RoPE, GQA, MQA.
Phase 2 will integrate Flash Attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.core.config import ModelConfig


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention (MHA)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        assert self.head_dim * self.num_heads == self.hidden_size, \
            "hidden_size must be divisible by num_attention_heads"

        # QKV projections
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)

        # Output projection
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # Scale factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, 1, seq_len, seq_len) or None

        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Project QKV
        qkv = self.qkv_proj(x)  # (B, S, 3H)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, S, dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # attn_scores: (B, heads, S, S)

        # Apply mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)  # (B, heads, S, dim)
        attn_output = attn_output.permute(0, 2, 1, 3)  # (B, S, heads, dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.out_proj(attn_output)

        return output
