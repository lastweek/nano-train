"""
Unit tests for attention-internal monitoring vitals.
"""

from __future__ import annotations

import math

import torch

from src.config import ModelConfig
from src.models.attention import MultiHeadAttention


def test_attention_monitoring_stats_exist_and_are_bounded() -> None:
    config = ModelConfig(
        param_dtype=torch.float32,
        param_device=None,
        hidden_size=16,
        num_attention_heads=4,
        dropout=0.0,
    )
    attn = MultiHeadAttention(config)
    attn.eval()

    x = torch.randn(2, 5, 16)

    attn.set_monitoring(True, tau=100.0)
    _ = attn(x)
    stats = attn._monitor_stats
    assert stats is not None

    assert math.isfinite(float(stats["max_attn_logit"]))
    assert math.isfinite(float(stats["attn_entropy"]))
    assert math.isfinite(float(stats["attn_entropy_norm"]))
    assert 0.0 <= float(stats["frac_logits_gt_tau"]) <= 1.0

    max_entropy = math.log(x.size(1))
    assert 0.0 <= float(stats["attn_entropy"]) <= max_entropy + 1e-5
    assert 0.0 <= float(stats["attn_entropy_norm"]) <= 1.0 + 1e-5

    attn.set_monitoring(False, tau=100.0)
    _ = attn(x)
    assert attn._monitor_stats is None
