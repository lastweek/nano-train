"""Persistent FP8 parameter storage tests for low-bit-capable linear modules."""

from __future__ import annotations

import torch

from src.layers import Linear
from src.runtime.contracts import ModulePrecisionAssignment



def _assignment() -> ModulePrecisionAssignment:
    return ModulePrecisionAssignment(
        module_name="linear",
        module_type="Linear",
        compute_lowbit_mode=None,
        persistent_lowbit_mode="fp8",
        persistent_scale_granularity="per_channel",
        fp4_persistent_format="nf4",
    )



def test_persistent_fp8_refresh_and_backward_path() -> None:
    layer = Linear(
        4,
        3,
        param_dtype=torch.float32,
        param_device=None,
    )
    layer.set_precision_assignment(_assignment())
    layer.refresh_persistent_lowbit_params()

    assert layer._persistent_fp8_weight.numel() == layer.weight.numel()  # type: ignore[attr-defined]
    assert layer._persistent_scale.numel() > 0  # type: ignore[attr-defined]

    x = torch.randn(2, 4, requires_grad=True)
    y = layer(x)
    y.sum().backward()

    assert layer.weight.grad is not None
    assert x.grad is not None
