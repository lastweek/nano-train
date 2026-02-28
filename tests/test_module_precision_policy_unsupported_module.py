"""Unsupported module selection tests for module-level precision policy."""

from __future__ import annotations

import argparse

import torch

from src.layers import Linear
from src.layers import LayerNorm
from src.runtime.mixed_precision import build_module_precision_resolver
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
        self.norm = LayerNorm(
            4,
            param_dtype=torch.float32,
            param_device=None,
            module_path="norm",
            precision_resolver=precision_resolver,
        )



def _args() -> argparse.Namespace:
    return argparse.Namespace(
        bf16=False,
        fp16=False,
        fp8=False,
        fp4=False,
        fp8_backend="emulated",
        fp8_format="e4m3",
        fp8_amax_history_len=16,
        fp8_amax_compute_algo="most_recent",
        fp4_backend="emulated",
        params_dtype=None,
        main_params_dtype=None,
        main_grads_dtype=None,
        exp_avg_dtype=None,
        exp_avg_sq_dtype=None,
        loss_scale_init=1024.0,
        loss_scale_growth_factor=2.0,
        loss_scale_backoff_factor=0.5,
        loss_scale_growth_interval=100,
        loss_scale_min=1.0,
        loss_scale_max=65536.0,
        fp8_param=False,
        fp4_param=False,
        fp4_param_format="nf4",
        persistent_scale_granularity="per_channel",
        module_pattern_type="regex",
        compute_lowbit_mode="fp8",
        compute_lowbit_include=[".*norm"],
        compute_lowbit_exclude=None,
        persistent_lowbit_mode="off",
        persistent_lowbit_include=None,
        persistent_lowbit_exclude=None,
    )



def test_resolver_rejects_unsupported_module_match() -> None:
    cfg = resolve_precision_config(_args(), torch.device("cpu"))
    resolver = build_module_precision_resolver(cfg)

    try:
        _ = _TinyModel(precision_resolver=resolver)
    except ValueError as exc:
        assert "without low-bit support" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported module low-bit selection")
