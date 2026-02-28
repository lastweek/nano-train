"""Persistent FP8 parameter storage tests for low-bit-capable linear modules."""

from __future__ import annotations

import torch

from src.layers import Linear
from src.runtime.contracts import ModulePrecisionPolicy
from src.runtime.contracts import PrecisionConfig
from src.runtime.mixed_precision import build_module_precision_resolver



def _resolver():
    return build_module_precision_resolver(
        PrecisionConfig(
            mode="fp32",
            module_precision_policy=ModulePrecisionPolicy(
                persistent_lowbit_mode="fp8",
                persistent_lowbit_include=("linear",),
                persistent_scale_granularity="per_channel",
                fp4_persistent_format="nf4",
            ),
        )
    )



def test_persistent_fp8_refresh_and_backward_path() -> None:
    resolver = _resolver()
    layer = Linear(
        4,
        3,
        param_dtype=torch.float32,
        param_device=None,
        module_path="linear",
        precision_resolver=resolver,
    )
    layer.refresh_persistent_lowbit_params()

    assert layer._persistent_fp8_weight.numel() == layer.weight.numel()  # type: ignore[attr-defined]
    assert layer._persistent_scale.numel() > 0  # type: ignore[attr-defined]

    x = torch.randn(2, 4, requires_grad=True)
    y = layer(x)
    y.sum().backward()

    assert layer.weight.grad is not None
    assert x.grad is not None
