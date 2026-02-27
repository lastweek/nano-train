"""
Multi-head attention implementation for MVP.

Supports tensor parallelism via TPConfig parameter.
Phase 5 will add RoPE, GQA, MQA.
Phase 2 will integrate Flash Attention.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.config import TPConfig
from src.layers import ColumnParallelLinear
from src.layers import Dropout
from src.layers import Linear
from src.layers import RowParallelLinear


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional tensor parallelism.

    If tp_config.enabled is True:
        - QKV projection uses ColumnParallelLinear (split by heads)
        - Output projection uses RowParallelLinear (split by heads)
        Communication: 1 all-reduce per forward pass

    If tp_config.enabled is False (default):
        - Uses standard Linear layers
        - No communication
    """

    def __init__(self, config: ModelConfig, tp_config: Optional[TPConfig] = None) -> None:
        super().__init__()
        tp_config = tp_config or TPConfig()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # TP configuration
        self.tp_enabled = tp_config.enabled
        self.tp_rank = tp_config.rank
        self.tp_size = tp_config.size
        self.tp_group = tp_config.group

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        if self.tp_enabled:
            # For TP: each GPU gets num_heads // tp_size heads
            if self.num_heads % self.tp_size != 0:
                raise ValueError(
                    f"num_heads ({self.num_heads}) must be divisible by tp_size "
                    f"({self.tp_size})"
                )

            self.tp_num_heads = self.num_heads // self.tp_size

            # ColumnParallelLinear expects the GLOBAL output size and returns a local shard.
            self.qkv_proj = ColumnParallelLinear(
                config.hidden_size,
                3 * config.hidden_size,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                tp_group=self.tp_group,
                bias=False,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
            )

            # RowParallelLinear expects the GLOBAL input size and consumes a local shard.
            self.out_proj = RowParallelLinear(
                config.hidden_size,
                config.hidden_size,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                tp_group=self.tp_group,
                bias=False,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
            )
        else:
            # Standard Linear layers
            self.qkv_proj = Linear(
                config.hidden_size,
                3 * config.hidden_size,
                bias=False,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
            )
            self.out_proj = Linear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
            )

        self.dropout = Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Monitoring (opt-in; enabled/disabled by Trainer).
        self._monitor_enabled: bool = False
        self._monitor_tau: float = 100.0
        self._monitor_stats: dict[str, float] | None = None

    def set_monitoring(self, enabled: bool, *, tau: float = 100.0) -> None:
        """Enable/disable internal attention monitoring and set tau for comparisons."""
        self._monitor_enabled = bool(enabled)
        self._monitor_tau = float(tau)
        if not self._monitor_enabled:
            self._monitor_stats = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, 1, seq_len, seq_len) or None

        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        if x.dim() != 3:
            raise ValueError("x must have shape [batch_size, seq_len, hidden_size]")

        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)

        if self.tp_enabled:
            # TP mode: reshape for tp_num_heads
            qkv = qkv.reshape(batch_size, seq_len, 3, self.tp_num_heads, self.head_dim)
            num_heads = self.tp_num_heads
        else:
            # Standard mode: reshape for num_heads
            qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            num_heads = self.num_heads

        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, S, dim)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)

        if self._monitor_enabled:
            with torch.no_grad():
                max_attn_logit = float(attn_scores.max().item())

                probs = attn_probs
                if probs.dtype not in (torch.float32, torch.float64):
                    probs = probs.float()
                probs_safe = probs.clamp_min(1e-12)
                entropy = -(probs_safe * probs_safe.log()).sum(dim=-1)
                attn_entropy = float(entropy.mean().item())
                attn_entropy_norm = float(attn_entropy / math.log(seq_len)) if seq_len > 1 else 0.0

                logits_gt_tau = attn_scores > float(self._monitor_tau)
                frac_logits_gt_tau = float(logits_gt_tau.float().mean().item())

                self._monitor_stats = {
                    "max_attn_logit": max_attn_logit,
                    "attn_entropy": attn_entropy,
                    "attn_entropy_norm": attn_entropy_norm,
                    "frac_logits_gt_tau": frac_logits_gt_tau,
                }

        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.permute(0, 2, 1, 3)

        if self.tp_enabled:
            attn_output = attn_output.reshape(
                batch_size,
                seq_len,
                self.tp_num_heads * self.head_dim,
            )
        else:
            attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        # Output projection (all-reduce inside if TP mode)
        output = self.out_proj(attn_output)

        return output
