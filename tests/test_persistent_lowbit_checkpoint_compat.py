"""Checkpoint compatibility tests for persistent low-bit parameter buffers."""

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



def test_load_legacy_state_dict_without_persistent_buffers() -> None:
    src = Linear(
        4,
        3,
        param_dtype=torch.float32,
        param_device=None,
    )
    legacy_state = {
        key: value
        for key, value in src.state_dict().items()
        if "_persistent_" not in key
    }

    dst = Linear(
        4,
        3,
        param_dtype=torch.float32,
        param_device=None,
    )
    dst.set_precision_assignment(_assignment())
    dst.load_state_dict(legacy_state, strict=True)

    assert torch.allclose(dst.weight, src.weight)
