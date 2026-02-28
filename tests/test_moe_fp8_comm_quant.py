"""MoE communication quantization behavior tests."""

from __future__ import annotations

import argparse

import torch

from src.models.moe import _comm_quant_dequant
from src.models.moe import ExpertParallelMoE
from src.runtime.mixed_precision import build_module_precision_resolver
from src.runtime.mixed_precision import resolve_precision_config


def _args(**overrides) -> argparse.Namespace:
    base = {
        "precision_recipe": "deepseek_v3",
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


def _build_moe(*, args: argparse.Namespace) -> ExpertParallelMoE:
    config = resolve_precision_config(args, torch.device("cpu"))
    resolver = build_module_precision_resolver(config)
    return ExpertParallelMoE(
        hidden_size=16,
        expert_intermediate_size=32,
        num_experts=4,
        top_k=2,
        ep_rank=0,
        ep_size=1,
        ep_group=None,
        param_dtype=torch.float32,
        param_device=None,
        precision_resolver=resolver,
        module_prefix="moe",
        dropout=0.0,
    )


def test_moe_comm_quant_defaults_enabled_for_deepseek_recipe() -> None:
    moe = _build_moe(args=_args())
    assert moe._comm_quant_enabled is True  # type: ignore[attr-defined]
    assert moe._comm_quant_granularity == "block_128x128"  # type: ignore[attr-defined]
    assert moe._comm_quant_rounding_mode == "stochastic"  # type: ignore[attr-defined]


def test_moe_comm_quant_can_be_disabled_by_override() -> None:
    moe = _build_moe(args=_args(fp8_comm_quant=False))
    assert moe._comm_quant_enabled is False  # type: ignore[attr-defined]


def test_comm_quant_dequant_preserves_shape_dtype_and_grad() -> None:
    payload = torch.randn(7, 257, requires_grad=True)

    out = _comm_quant_dequant(
        payload,
        granularity="block_128x128",
        rounding_mode="stochastic",
        bits=8,
    )
    assert out.shape == payload.shape
    assert out.dtype == payload.dtype
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert payload.grad is not None
