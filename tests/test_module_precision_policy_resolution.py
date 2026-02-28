"""Resolution tests for module-level precision policy mapping."""

from __future__ import annotations

import argparse

import torch

from src.layers import LayerNorm
from src.layers import Linear
from src.models.deepseek import RMSNorm
from src.runtime.mixed_precision import build_module_precision_resolver
from src.runtime.mixed_precision import finalize_module_precision_resolver
from src.runtime.mixed_precision import resolve_precision_config


class _TinyModel(torch.nn.Module):
    def __init__(self, *, precision_resolver) -> None:
        super().__init__()
        self.proj = Linear(
            4,
            4,
            param_dtype=torch.float32,
            param_device=None,
            module_path="proj",
            precision_resolver=precision_resolver,
        )
        self.skip = Linear(
            4,
            4,
            param_dtype=torch.float32,
            param_device=None,
            module_path="skip",
            precision_resolver=precision_resolver,
        )
        self.relu = torch.nn.ReLU()



def _args(**overrides) -> argparse.Namespace:
    base = {
        "bf16": False,
        "fp16": False,
        "fp8": True,
        "fp4": False,
        "fp8_backend": "emulated",
        "fp8_format": "e4m3",
        "fp8_amax_history_len": 16,
        "fp8_amax_compute_algo": "most_recent",
        "fp4_backend": "emulated",
        "params_dtype": None,
        "main_params_dtype": None,
        "main_grads_dtype": None,
        "exp_avg_dtype": None,
        "exp_avg_sq_dtype": None,
        "loss_scale_init": 1024.0,
        "loss_scale_growth_factor": 2.0,
        "loss_scale_backoff_factor": 0.5,
        "loss_scale_growth_interval": 100,
        "loss_scale_min": 1.0,
        "loss_scale_max": 65536.0,
        "fp8_param": False,
        "fp4_param": False,
        "fp4_param_format": "nf4",
        "persistent_scale_granularity": "per_channel",
        "module_pattern_type": "regex",
        "compute_lowbit_mode": None,
        "compute_lowbit_include": None,
        "compute_lowbit_exclude": [".*skip"],
        "persistent_lowbit_mode": "off",
        "persistent_lowbit_include": None,
        "persistent_lowbit_exclude": None,
        "module_compute_dtype_rule": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)



def test_constructor_resolver_respects_exclude_pattern() -> None:
    cfg = resolve_precision_config(_args(), torch.device("cpu"))
    resolver = build_module_precision_resolver(cfg)
    model = _TinyModel(precision_resolver=resolver)
    summary = finalize_module_precision_resolver(resolver)

    proj = model.proj._module_precision_state.assignment  # type: ignore[attr-defined]
    skip = model.skip._module_precision_state.assignment  # type: ignore[attr-defined]
    assert proj.compute_lowbit_mode == "fp8"
    assert skip.compute_lowbit_mode is None
    assert summary.compute_lowbit_module_count == 1



def test_include_pattern_must_match_at_least_one_module() -> None:
    cfg = resolve_precision_config(
        _args(compute_lowbit_include=["^does_not_exist$"]),
        torch.device("cpu"),
    )
    resolver = build_module_precision_resolver(cfg)
    _ = _TinyModel(precision_resolver=resolver)

    try:
        finalize_module_precision_resolver(resolver)
    except ValueError as exc:
        assert "matched zero modules" in str(exc)
    else:
        raise AssertionError("Expected ValueError when include pattern matches no modules")


def test_module_compute_dtype_override_assigns_non_lowbit_linear() -> None:
    cfg = resolve_precision_config(
        _args(
            fp8=False,
            module_compute_dtype_rule=["^skip=fp16"],
        ),
        torch.device("cpu"),
    )
    resolver = build_module_precision_resolver(cfg)
    model = _TinyModel(precision_resolver=resolver)
    summary = finalize_module_precision_resolver(resolver)

    proj_assignment = model.proj._module_precision_state.assignment  # type: ignore[attr-defined]
    skip_assignment = model.skip._module_precision_state.assignment  # type: ignore[attr-defined]

    assert proj_assignment.compute_lowbit_mode is None
    assert proj_assignment.compute_dtype_override is None
    assert skip_assignment.compute_lowbit_mode is None
    assert skip_assignment.compute_dtype_override == "fp16"
    assert summary.compute_lowbit_module_count == 0


def test_module_compute_dtype_override_conflicts_with_lowbit_assignment() -> None:
    cfg = resolve_precision_config(
        _args(
            fp8=True,
            module_compute_dtype_rule=["^proj=bf16"],
        ),
        torch.device("cpu"),
    )
    resolver = build_module_precision_resolver(cfg)
    try:
        _ = _TinyModel(precision_resolver=resolver)
    except ValueError as exc:
        assert "both low-bit compute and compute-dtype override" in str(exc)
    else:
        raise AssertionError("Expected ValueError for low-bit and dtype override conflict")


def test_module_compute_dtype_override_applies_to_layernorm() -> None:
    cfg = resolve_precision_config(
        _args(
            fp8=False,
            module_compute_dtype_rule=["^norm=bf16"],
        ),
        torch.device("cpu"),
    )
    resolver = build_module_precision_resolver(cfg)
    norm = LayerNorm(
        8,
        param_dtype=torch.float32,
        param_device=None,
        module_path="norm",
        precision_resolver=resolver,
    )
    _ = finalize_module_precision_resolver(resolver)

    x = torch.randn(2, 8, dtype=torch.float32)
    y = norm(x)
    assert y.dtype == torch.bfloat16


def test_module_compute_dtype_override_applies_to_deepseek_rmsnorm() -> None:
    cfg = resolve_precision_config(
        _args(
            fp8=False,
            module_compute_dtype_rule=["^rms=fp16"],
        ),
        torch.device("cpu"),
    )
    resolver = build_module_precision_resolver(cfg)
    norm = RMSNorm(
        8,
        param_dtype=torch.float32,
        param_device=None,
        module_path="rms",
        precision_resolver=resolver,
    )
    _ = finalize_module_precision_resolver(resolver)

    x = torch.randn(2, 8, dtype=torch.float32)
    y = norm(x)
    assert y.dtype == torch.float16
