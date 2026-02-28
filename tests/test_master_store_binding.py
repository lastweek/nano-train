"""Optimizer-owned master store binding behavior for low-bit modules."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from src.layers import Linear
from src.runtime.master_store import materialize_optimizer_owned_masters
from src.runtime.mixed_precision import build_module_precision_resolver
from src.runtime.mixed_precision import resolve_precision_config


def _precision_args() -> argparse.Namespace:
    return argparse.Namespace(
        bf16=False,
        fp16=False,
        fp8=True,
        fp4=False,
        fp8_backend="emulated",
        fp8_format="e4m3",
        fp8_amax_history_len=8,
        fp8_amax_compute_algo="most_recent",
        fp4_backend="emulated",
        params_dtype="bf16",
        main_params_dtype="fp32",
        main_grads_dtype="fp32",
        exp_avg_dtype="fp32",
        exp_avg_sq_dtype="fp32",
        loss_scale_init=65536.0,
        loss_scale_growth_factor=2.0,
        loss_scale_backoff_factor=0.5,
        loss_scale_growth_interval=2000,
        loss_scale_min=1.0,
        loss_scale_max=16777216.0,
        fp8_param=True,
        fp4_param=False,
        fp4_param_format="nf4",
        persistent_scale_granularity="per_channel",
        module_pattern_type="regex",
        compute_lowbit_mode=None,
        compute_lowbit_include=None,
        compute_lowbit_exclude=None,
        persistent_lowbit_mode=None,
        persistent_lowbit_include=None,
        persistent_lowbit_exclude=None,
        lowbit_master_ownership="optimizer",
    )


class _TinyModel(nn.Module):
    def __init__(self, resolver) -> None:
        super().__init__()
        self.proj = Linear(
            8,
            8,
            param_dtype=torch.bfloat16,
            param_device=None,
            module_path="proj",
            precision_resolver=resolver,
        )


def test_materialize_optimizer_owned_masters_rebinds_module_weight() -> None:
    cfg = resolve_precision_config(_precision_args(), torch.device("cpu"))
    resolver = build_module_precision_resolver(cfg)
    model = _TinyModel(resolver)
    resolver.finalize()

    original_weight = model.proj.weight
    store = materialize_optimizer_owned_masters(model, precision_config=cfg)
    assert store is not None
    assert len(store.metadata) == 1
    assert model.proj.weight is not original_weight
    key = next(iter(store.metadata.keys()))
    assert model.proj.weight is store.masters[key]
