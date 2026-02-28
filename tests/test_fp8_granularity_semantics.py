"""FP8 granularity metadata propagation tests."""

from __future__ import annotations

import argparse

import torch

from src.layers import Linear
from src.runtime.mixed_precision import build_module_precision_resolver
from src.runtime.mixed_precision import resolve_precision_config


def _args(**overrides) -> argparse.Namespace:
    base = {
        "precision_recipe": "default",
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
        "compute_lowbit_exclude": None,
        "persistent_lowbit_mode": "off",
        "persistent_lowbit_include": None,
        "persistent_lowbit_exclude": None,
        "lowbit_master_ownership": "optimizer",
        "fp8_rounding": None,
        "fp8_activation_granularity": None,
        "fp8_weight_granularity": None,
        "fp8_comm_quant": None,
        "fp8_comm_granularity": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _layer_backend(*, args: argparse.Namespace):
    config = resolve_precision_config(args, torch.device("cpu"))
    resolver = build_module_precision_resolver(config)
    layer = Linear(
        257,
        129,
        param_dtype=torch.float32,
        param_device=None,
        module_path="decoder.blocks.0.mlp.fc1",
        precision_resolver=resolver,
    )
    state = layer._module_precision_state  # type: ignore[attr-defined]
    backend = state.lowbit_backend
    assert backend is not None
    return layer, backend


def test_deepseek_recipe_propagates_tile_block_stochastic_to_backend() -> None:
    _, backend = _layer_backend(args=_args(precision_recipe="deepseek_v3", fp8=True))

    assert getattr(backend, "_activation_granularity") == "tile_1x128"
    assert getattr(backend, "_weight_granularity") == "block_128x128"
    assert getattr(backend, "_rounding_mode") == "stochastic"


def test_recipe_overrides_propagate_to_backend_kernel_metadata() -> None:
    _, backend = _layer_backend(
        args=_args(
            precision_recipe="deepseek_v3",
            fp8=True,
            fp8_activation_granularity="tensor",
            fp8_weight_granularity="channel",
            fp8_rounding="nearest",
        )
    )

    assert getattr(backend, "_activation_granularity") == "tensor"
    assert getattr(backend, "_weight_granularity") == "channel"
    assert getattr(backend, "_rounding_mode") == "nearest"


def test_emulated_granularity_dispatch_runs_with_padding_shapes() -> None:
    layer, _ = _layer_backend(args=_args(precision_recipe="deepseek_v3", fp8=True))

    x = torch.randn(3, 257, requires_grad=True)
    y = layer(x)
    assert y.shape == (3, 129)
    assert torch.isfinite(y).all()

    y.sum().backward()
    assert x.grad is not None
