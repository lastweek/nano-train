"""Tests for local routed MoE behavior."""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.moe import LocalRoutedMoE
from src.runtime.contracts import PrecisionConfig
from src.runtime.mixed_precision import build_module_precision_resolver


def test_local_moe_forward_backward_and_stats() -> None:
    """Local routed MoE runs forward/backward and exposes routing stats."""
    torch.manual_seed(3)

    moe = LocalRoutedMoE(
        hidden_size=16,
        expert_intermediate_size=32,
        num_experts=4,
        top_k=2,
        param_dtype=torch.float32,
        param_device=None,
        precision_resolver=build_module_precision_resolver(PrecisionConfig(mode="fp32")),
        module_prefix="moe_local",
        dropout=0.0,
        n_shared_experts=1,
        scoring_func="sigmoid",
        n_group=2,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        capacity_factor=0.5,
    )

    x = torch.randn(3, 8, 16)
    out = moe(x)
    loss = out.pow(2).mean() + 0.01 * moe.last_aux_loss
    loss.backward()

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(moe.last_aux_loss)
    assert 0.0 <= moe.last_dropped_fraction <= 1.0

    # Assignment count cannot exceed total routed assignments.
    total_capacity_assignments = int(moe.last_token_counts.sum().item())
    assert total_capacity_assignments <= (x.shape[0] * x.shape[1] * moe.top_k)

    router_has_grad = any(param.grad is not None for param in moe.router.parameters())
    expert_has_grad = any(
        param.grad is not None
        for expert in moe.experts
        for param in expert.parameters()
    )

    assert router_has_grad
    assert expert_has_grad
