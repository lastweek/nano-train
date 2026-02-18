"""
Pytest fixtures and shared test helpers.

Some test modules import helpers via `from conftest import ...`, so this file lives at the
repository root (which pytest adds to `sys.path`) rather than only under `tests/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest
import torch


@dataclass(frozen=True)
class Tolerances:
    """Numerical tolerances for tensor comparisons."""

    RTOL: float = 1e-5
    ATOL: float = 1e-6


@pytest.fixture(params=["cpu"])
def device(request) -> torch.device:
    """Device fixture used by unit tests."""
    return torch.device(request.param)


@pytest.fixture()
def tolerances() -> Tolerances:
    """Default numerical tolerances used by accuracy tests."""
    return Tolerances()


def assert_tensor_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: Optional[str] = None,
) -> None:
    """
    Assert two tensors are close within tolerances.

    Args:
        actual: Tensor under test.
        expected: Reference tensor.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        msg: Optional message prefix on failure.
    """
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} vs {expected.shape}")

    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = (actual - expected).abs()
        max_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
        raise AssertionError(f"{msg or 'Tensors not close'}: max diff = {max_diff}")


def assert_grad_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: Optional[str] = None,
) -> None:
    """
    Assert two gradient tensors are close within tolerances.

    This is a thin wrapper around `assert_tensor_close` for readability in tests.
    """
    assert_tensor_close(actual, expected, rtol=rtol, atol=atol, msg=msg or "Gradients not close")

