"""Transformer Engine backend behavior for training-time parameter ownership."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.runtime.contracts import LowBitKernelSpec
from src.runtime.contracts import PrecisionConfig
from src.runtime.te_backend import build_lowbit_backend_for_mode


class _FakeAutocast:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class _FakeTELinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def _install_fake_te(monkeypatch: pytest.MonkeyPatch) -> None:
    root_mod = types.ModuleType("transformer_engine")
    pytorch_mod = types.ModuleType("transformer_engine.pytorch")

    def _autocast(**kwargs):
        del kwargs
        return _FakeAutocast()

    pytorch_mod.Linear = _FakeTELinear
    pytorch_mod.autocast = _autocast
    root_mod.pytorch = pytorch_mod

    monkeypatch.setitem(sys.modules, "transformer_engine", root_mod)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", pytorch_mod)


def _build_kernel_spec() -> LowBitKernelSpec:
    return LowBitKernelSpec(
        module_type="linear",
        in_features=4,
        out_features=3,
        has_bias=True,
    )


def _build_config() -> PrecisionConfig:
    return PrecisionConfig(
        mode="fp8",
        fp8_backend="transformer_engine",
    )


def test_te_backend_requires_bind_before_linear(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_te(monkeypatch)
    backend = build_lowbit_backend_for_mode(
        _build_config(),
        mode="fp8",
        kernel_spec=_build_kernel_spec(),
    )
    assert backend is not None

    with pytest.raises(RuntimeError, match="before parameter binding"):
        backend.linear(
            torch.randn(2, 4),
            torch.randn(3, 4),
            torch.randn(3),
        )


def test_te_backend_uses_bound_parameters_and_keeps_grad_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_te(monkeypatch)
    backend = build_lowbit_backend_for_mode(
        _build_config(),
        mode="fp8",
        kernel_spec=_build_kernel_spec(),
    )
    assert backend is not None

    weight = nn.Parameter(torch.randn(3, 4))
    bias = nn.Parameter(torch.randn(3))

    bind_fn = getattr(backend, "bind_parameters", None)
    assert callable(bind_fn)
    bind_fn(weight, bias)

    x = torch.randn(5, 4, requires_grad=True)
    y = backend.linear(x, weight, bias)
    y.sum().backward()

    assert weight.grad is not None
    assert bias.grad is not None
    assert x.grad is not None

    # Verify that TE module points to original trainable Parameter objects.
    te_linear = getattr(backend, "_te_linear")
    assert te_linear.weight is weight
    assert te_linear.bias is bias

    # Non-parameter tensor fallback path should still execute.
    y_fallback = backend.linear(x.detach(), weight + 0.0, bias)
    assert y_fallback.shape == (5, 3)
