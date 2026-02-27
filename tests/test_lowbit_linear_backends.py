"""Low-bit backend dispatch tests for linear building blocks."""

from __future__ import annotations

from typing import Literal

import pytest
import torch

from src.layers import ColumnParallelLinear
from src.layers import Linear
from src.layers import RowParallelLinear
from src.runtime.contracts import ModulePrecisionAssignment
from src.runtime.contracts import PrecisionConfig
from src.runtime.te_backend import ActiveLowBitContext
from src.runtime.te_backend import clear_active_lowbit_context
from src.runtime.te_backend import set_active_lowbit_context


class _CountingBackend:
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def linear(self, input_tensor, weight, bias, *, mode, config):
        del mode, config
        self.calls += 1
        return torch.nn.functional.linear(input_tensor, weight, bias)


def _set_compute_mode(module: torch.nn.Module, mode: Literal["fp8", "fp4"]) -> None:
    setter = getattr(module, "set_precision_assignment", None)
    if not callable(setter):
        raise RuntimeError("Test module does not support precision assignments")
    setter(
        ModulePrecisionAssignment(
            module_name="test_module",
            module_type=module.__class__.__name__,
            compute_lowbit_mode=mode,
            persistent_lowbit_mode="off",
            persistent_scale_granularity="per_channel",
            fp4_persistent_format="nf4",
        )
    )


def test_linear_uses_active_lowbit_backend_dispatch() -> None:
    layer = Linear(
        4,
        3,
        param_dtype=torch.float32,
        param_device=None,
    )
    _set_compute_mode(layer, "fp8")
    x = torch.randn(2, 4, requires_grad=True)
    backend = _CountingBackend()
    cfg = PrecisionConfig(mode="fp8", activation_dtype="bf16", fp8_backend="emulated")

    set_active_lowbit_context(
        ActiveLowBitContext(
            backend_by_mode={"fp8": backend},
            default_mode="fp8",
            config=cfg,
        )
    )
    try:
        y = layer(x)
        y.sum().backward()
    finally:
        clear_active_lowbit_context()

    assert backend.calls == 1
    assert x.grad is not None


def test_tp_linear_layers_dispatch_to_lowbit_backend() -> None:
    backend = _CountingBackend()
    cfg = PrecisionConfig(mode="fp4", activation_dtype="fp32", fp4_backend="emulated")
    set_active_lowbit_context(
        ActiveLowBitContext(
            backend_by_mode={"fp4": backend},
            default_mode="fp4",
            config=cfg,
        )
    )
    try:
        col = ColumnParallelLinear(
            4,
            6,
            tp_rank=0,
            tp_size=1,
            param_dtype=torch.float32,
            param_device=None,
        )
        _set_compute_mode(col, "fp4")
        row = RowParallelLinear(
            6,
            5,
            tp_rank=0,
            tp_size=1,
            param_dtype=torch.float32,
            param_device=None,
        )
        _set_compute_mode(row, "fp4")

        x = torch.randn(2, 4, requires_grad=True)
        y = col(x)
        z = row(y)
        z.sum().backward()
    finally:
        clear_active_lowbit_context()

    # One call for column-parallel linear, one call for row-parallel partial matmul.
    assert backend.calls == 2
    assert x.grad is not None


def test_lowbit_context_requires_explicit_module_assignment() -> None:
    layer = Linear(
        4,
        3,
        param_dtype=torch.float32,
        param_device=None,
    )
    x = torch.randn(2, 4, requires_grad=True)
    backend = _CountingBackend()
    cfg = PrecisionConfig(mode="fp8", activation_dtype="bf16", fp8_backend="emulated")

    set_active_lowbit_context(
        ActiveLowBitContext(
            backend_by_mode={"fp8": backend},
            default_mode="fp8",
            config=cfg,
        )
    )
    try:
        with pytest.raises(RuntimeError, match="has no precision assignment"):
            _ = layer(x)
    finally:
        clear_active_lowbit_context()
