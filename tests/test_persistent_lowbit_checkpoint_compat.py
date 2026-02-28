"""Checkpoint compatibility tests for persistent low-bit parameter buffers."""

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



def test_load_legacy_state_dict_without_persistent_buffers() -> None:
    src_resolver = _resolver()
    src = Linear(
        4,
        3,
        param_dtype=torch.float32,
        param_device=None,
        module_path="linear",
        precision_resolver=src_resolver,
    )
    legacy_state = {
        key: value
        for key, value in src.state_dict().items()
        if "_persistent_" not in key
    }

    dst_resolver = _resolver()
    dst = Linear(
        4,
        3,
        param_dtype=torch.float32,
        param_device=None,
        module_path="linear",
        precision_resolver=dst_resolver,
    )
    dst.load_state_dict(legacy_state, strict=True)

    assert torch.allclose(dst.weight, src.weight)
