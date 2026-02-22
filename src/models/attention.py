"""
Multi-head attention implementation for MVP.

Phase 5 will add RoPE, GQA, MQA.
Phase 2 will integrate Flash Attention.
"""

import math
import torch
import torch.nn as nn

from src.config import ModelConfig
from src.layers import Linear, Dropout


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
        # NATIVE: Our Linear in src/layers.py
        # ORIGINAL: torch.nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        # Single matrix multiply for efficiency: x @ [W_q, W_k, W_v].T
        self.qkv_proj = Linear(config.hidden_size, 3 * config.hidden_size, bias=False)

        # Output projection
        # NATIVE: Our Linear in src/layers.py
        # ORIGINAL: torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = Linear(config.hidden_size, config.hidden_size, bias=False)

        # NATIVE: Our Dropout in src/layers.py
        # ORIGINAL: torch.nn.Dropout(config.dropout)
        self.dropout = Dropout(config.dropout)

        # Scale factor
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

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, 1, seq_len, seq_len) or None

        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        # INPUT: x (batch_size, seq_len, hidden_size)
        batch_size, seq_len, _ = x.shape

        # Project QKV from hidden size to 3 * hidden size for a single matmul.
        qkv = self.qkv_proj(x)  # (B, S, 3H) - single matmul for efficiency
        # OUTPUT: qkv (B, S, 3H)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, S, dim)

        # Extract Q, K, V from combined projections
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute scaled dot-product attention scores.
        # INPUT: q (B, heads, S, D), k (B, heads, S, D)
        # OUTPUT: attn_scores (B, heads, S, S)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if attention_mask is not None:
            # Mask uses -inf to block future tokens in softmax.
            attn_scores = attn_scores + attention_mask

        # Softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)  # (B, heads, S, S)

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
        # INPUT: attn_probs (B, heads, S, S), v (B, heads, S, D)
        # OUTPUT: attn_output (B, heads, S, D)
        attn_output = torch.matmul(attn_probs, v) # (B, heads, S, dim)
        attn_output = attn_output.permute(0, 2, 1, 3)  # (B, S, heads, dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)  # (B, S, H)

        # Output projection back to hidden size.
        output = self.out_proj(attn_output) # (B, S, H)

        return output
