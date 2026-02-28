"""Low-bit backend dispatch tests for linear building blocks."""

from __future__ import annotations

import torch

from src.layers import ColumnParallelLinear
from src.layers import Linear
from src.layers import RowParallelLinear
from src.runtime.contracts import ModulePrecisionAssignment
from src.runtime.contracts import ModulePrecisionInitState
from src.runtime.contracts import ModulePrecisionResolver
from src.runtime.contracts import ModulePrecisionSummary


class _CountingBackend:
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def linear(self, input_tensor, weight, bias):
        self.calls += 1
        return torch.nn.functional.linear(input_tensor, weight, bias)


class _StaticResolver(ModulePrecisionResolver):
    def __init__(
        self,
        *,
        compute_lowbit_mode: str | None,
        lowbit_backend: _CountingBackend | None,
        compute_dtype_override: str | None = None,
    ) -> None:
        self._compute_lowbit_mode = compute_lowbit_mode
        self._lowbit_backend = lowbit_backend
        self._compute_dtype_override = compute_dtype_override

    def resolve_module_init_state(
        self,
        *,
        module_path: str,
        module_type: str,
        lowbit_capable_type,
        kernel_spec=None,
    ) -> ModulePrecisionInitState:
        del lowbit_capable_type, kernel_spec
        assignment = ModulePrecisionAssignment(
            module_name=module_path,
            module_type=module_type,
            compute_lowbit_mode=self._compute_lowbit_mode,  # type: ignore[arg-type]
            persistent_lowbit_mode="off",
            persistent_scale_granularity="per_channel",
            fp4_persistent_format="nf4",
            compute_dtype_override=self._compute_dtype_override,  # type: ignore[arg-type]
        )
        return ModulePrecisionInitState(
            assignment=assignment,
            lowbit_backend=self._lowbit_backend,
            lowbit_capable_type="linear",
        )

    def finalize(self) -> ModulePrecisionSummary:
        return ModulePrecisionSummary()

    def deepseek_v3_recipe(self):
        return None


def test_linear_uses_constructor_bound_backend_dispatch() -> None:
    backend = _CountingBackend()
    resolver = _StaticResolver(compute_lowbit_mode="fp8", lowbit_backend=backend)
    layer = Linear(
        4,
        3,
        param_dtype=torch.float32,
        param_device=None,
        module_path="linear",
        precision_resolver=resolver,
    )
    x = torch.randn(2, 4, requires_grad=True)
    y = layer(x)
    y.sum().backward()

    assert backend.calls == 1
    assert x.grad is not None


def test_tp_linear_layers_dispatch_to_lowbit_backend() -> None:
    backend = _CountingBackend()
    resolver = _StaticResolver(compute_lowbit_mode="fp4", lowbit_backend=backend)
    col = ColumnParallelLinear(
        4,
        6,
        tp_rank=0,
        tp_size=1,
        param_dtype=torch.float32,
        param_device=None,
        module_path="col",
        precision_resolver=resolver,
    )
    row = RowParallelLinear(
        6,
        5,
        tp_rank=0,
        tp_size=1,
        param_dtype=torch.float32,
        param_device=None,
        module_path="row",
        precision_resolver=resolver,
    )

    x = torch.randn(2, 4, requires_grad=True)
    y = col(x)
    z = row(y)
    z.sum().backward()

    # One call for column-parallel linear, one call for row-parallel partial matmul.
    assert backend.calls == 2
    assert x.grad is not None


def test_unassigned_module_uses_standard_linear_path() -> None:
    backend = _CountingBackend()
    resolver = _StaticResolver(compute_lowbit_mode=None, lowbit_backend=backend)
    layer = Linear(
        4,
        3,
        param_dtype=torch.float32,
        param_device=None,
        module_path="linear",
        precision_resolver=resolver,
    )
    x = torch.randn(2, 4, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert backend.calls == 0
    assert x.grad is not None


def test_compute_dtype_override_forces_linear_compute_dtype() -> None:
    resolver = _StaticResolver(
        compute_lowbit_mode=None,
        lowbit_backend=None,
        compute_dtype_override="fp16",
    )
    layer = Linear(
        4,
        3,
        param_dtype=torch.float32,
        param_device=None,
        module_path="linear",
        precision_resolver=resolver,
    )
    x = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
    y = layer(x)
    assert y.dtype == torch.float16
    y.float().sum().backward()
    assert x.grad is not None
