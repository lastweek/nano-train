"""Persistent FP4/NF4 parameter storage tests for low-bit-capable linear modules."""

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
                persistent_lowbit_mode="fp4",
                persistent_lowbit_include=("linear",),
                persistent_scale_granularity="per_tensor",
                fp4_persistent_format="nf4",
            ),
        )
    )



def test_persistent_fp4_nf4_refresh_and_forward() -> None:
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

    assert layer._persistent_fp4_codes.numel() > 0  # type: ignore[attr-defined]
    assert int(layer._persistent_numel.item()) == layer.weight.numel()  # type: ignore[attr-defined]

    x = torch.randn(2, 4)
    y = layer(x)
    assert y.shape == (2, 3)
