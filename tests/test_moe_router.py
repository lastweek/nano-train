"""Tests for top-k routing behavior in shared MoE infrastructure."""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.moe import TopKRouter
from src.runtime.contracts import PrecisionConfig
from src.runtime.mixed_precision import build_module_precision_resolver


def test_router_shapes_and_ranges() -> None:
    """Router returns valid top-k shapes and expert id ranges."""
    torch.manual_seed(0)

    router = TopKRouter(
        hidden_size=16,
        num_experts=8,
        top_k=2,
        param_dtype=torch.float32,
        param_device=None,
        precision_resolver=build_module_precision_resolver(PrecisionConfig(mode="fp32")),
        module_prefix="router0",
        scoring_func="sigmoid",
        n_group=1,
        topk_group=1,
    )
    tokens = torch.randn(11, 16)

    output = router(tokens)

    assert output.topk_indices.shape == (11, 2)
    assert output.topk_weights.shape == (11, 2)
    assert torch.isfinite(output.topk_weights).all()
    assert output.topk_indices.min().item() >= 0
    assert output.topk_indices.max().item() < 8


def test_router_group_routing_masks_other_groups() -> None:
    """With topk_group=1, selected experts per token should come from one group."""
    torch.manual_seed(1)

    num_experts = 8
    n_group = 4
    experts_per_group = num_experts // n_group

    router = TopKRouter(
        hidden_size=12,
        num_experts=num_experts,
        top_k=2,
        param_dtype=torch.float32,
        param_device=None,
        precision_resolver=build_module_precision_resolver(PrecisionConfig(mode="fp32")),
        module_prefix="router1",
        scoring_func="sigmoid",
        n_group=n_group,
        topk_group=1,
    )

    tokens = torch.randn(7, 12)
    output = router(tokens)

    expert_groups = output.topk_indices // experts_per_group
    for token_idx in range(expert_groups.size(0)):
        unique_groups = torch.unique(expert_groups[token_idx])
        assert unique_groups.numel() == 1


def test_router_aux_loss_backward_is_finite() -> None:
    """Aux loss should be finite and support backward."""
    torch.manual_seed(2)

    router = TopKRouter(
        hidden_size=10,
        num_experts=6,
        top_k=2,
        param_dtype=torch.float32,
        param_device=None,
        precision_resolver=build_module_precision_resolver(PrecisionConfig(mode="fp32")),
        module_prefix="router2",
        scoring_func="softmax",
        n_group=1,
        topk_group=1,
    )

    tokens = torch.randn(9, 10)
    output = router(tokens)

    loss = output.topk_weights.mean() + output.aux_loss
    loss.backward()

    has_grad = any(param.grad is not None for param in router.parameters())
    assert has_grad
    assert torch.isfinite(output.aux_loss)
