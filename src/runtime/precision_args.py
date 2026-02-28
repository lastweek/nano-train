"""Shared mixed-precision CLI argument helpers for runtime scripts."""

from __future__ import annotations

import argparse

import torch

from src.runtime.contracts import PrecisionConfig
from src.runtime.mixed_precision import resolve_precision_config


def add_mixed_precision_args(parser: argparse.ArgumentParser) -> None:
    """Register shared mixed-precision and low-bit policy CLI flags."""
    parser.add_argument(
        "--precision-recipe",
        type=str,
        default="default",
        choices=["default", "deepseek_v3"],
        help="Precision policy preset. deepseek_v3 enables DeepSeek-V3 FP8 defaults.",
    )
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision")
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 mixed precision")
    parser.add_argument("--fp4", action="store_true", help="Enable FP4 mixed precision (emulated)")
    parser.add_argument(
        "--fp8-backend",
        type=str,
        default="transformer_engine",
        choices=["transformer_engine", "emulated"],
        help="FP8 backend implementation",
    )
    parser.add_argument(
        "--fp8-format",
        type=str,
        default="e4m3",
        choices=["e4m3", "hybrid"],
        help="FP8 format recipe",
    )
    parser.add_argument(
        "--fp8-amax-history-len",
        type=int,
        default=16,
        help="FP8 amax history length",
    )
    parser.add_argument(
        "--fp8-amax-compute-algo",
        type=str,
        default="most_recent",
        choices=["most_recent", "max"],
        help="FP8 amax update algorithm",
    )
    parser.add_argument(
        "--fp4-backend",
        type=str,
        default="emulated",
        choices=["emulated"],
        help="FP4 backend implementation",
    )
    parser.add_argument(
        "--params-dtype",
        type=str,
        default=None,
        choices=["fp32", "bf16", "fp16"],
        help="Model parameter dtype",
    )
    parser.add_argument(
        "--main-params-dtype",
        type=str,
        default=None,
        choices=["fp32", "bf16", "fp16"],
        help="Optimizer main parameter dtype",
    )
    parser.add_argument(
        "--main-grads-dtype",
        type=str,
        default=None,
        choices=["fp32", "bf16", "fp16"],
        help="Optimizer main grad dtype",
    )
    parser.add_argument(
        "--exp-avg-dtype",
        type=str,
        default=None,
        choices=["fp32", "bf16", "fp16"],
        help="Optimizer first-moment dtype",
    )
    parser.add_argument(
        "--exp-avg-sq-dtype",
        type=str,
        default=None,
        choices=["fp32", "bf16", "fp16"],
        help="Optimizer second-moment dtype",
    )
    parser.add_argument("--loss-scale-init", type=float, default=65536.0)
    parser.add_argument("--loss-scale-growth-factor", type=float, default=2.0)
    parser.add_argument("--loss-scale-backoff-factor", type=float, default=0.5)
    parser.add_argument("--loss-scale-growth-interval", type=int, default=2000)
    parser.add_argument("--loss-scale-min", type=float, default=1.0)
    parser.add_argument("--loss-scale-max", type=float, default=16777216.0)

    parser.add_argument(
        "--fp8-param",
        action="store_true",
        help="Enable persistent FP8 parameter storage for selected modules",
    )
    parser.add_argument(
        "--fp4-param",
        action="store_true",
        help="Enable persistent FP4 (NF4) parameter storage for selected modules",
    )
    parser.add_argument(
        "--fp4-param-format",
        type=str,
        default="nf4",
        choices=["nf4"],
        help="Persistent FP4 storage format",
    )
    parser.add_argument(
        "--persistent-scale-granularity",
        type=str,
        default="per_channel",
        choices=["per_tensor", "per_channel"],
        help="Scale granularity for persistent low-bit parameter quantization",
    )
    parser.add_argument(
        "--module-pattern-type",
        type=str,
        default="regex",
        choices=["regex", "glob"],
        help="Pattern matcher used by per-module low-bit include/exclude flags",
    )
    parser.add_argument(
        "--compute-lowbit-mode",
        type=str,
        default=None,
        choices=["fp8", "fp4"],
        help="Per-module low-bit compute mode override",
    )
    parser.add_argument(
        "--compute-lowbit-include",
        action="append",
        default=None,
        help="Repeatable include patterns for low-bit compute module selection",
    )
    parser.add_argument(
        "--compute-lowbit-exclude",
        action="append",
        default=None,
        help="Repeatable exclude patterns for low-bit compute module selection",
    )
    parser.add_argument(
        "--persistent-lowbit-mode",
        type=str,
        default="off",
        choices=["off", "fp8", "fp4"],
        help="Per-module persistent low-bit parameter mode override",
    )
    parser.add_argument(
        "--persistent-lowbit-include",
        action="append",
        default=None,
        help="Repeatable include patterns for persistent low-bit module selection",
    )
    parser.add_argument(
        "--persistent-lowbit-exclude",
        action="append",
        default=None,
        help="Repeatable exclude patterns for persistent low-bit module selection",
    )
    parser.add_argument(
        "--module-compute-dtype-rule",
        action="append",
        default=None,
        help=(
            "Repeatable per-module compute dtype override rule in "
            "'<pattern>=<fp32|bf16|fp16>' format"
        ),
    )
    parser.add_argument(
        "--lowbit-master-ownership",
        type=str,
        default="optimizer",
        choices=["module", "optimizer"],
        help="Ownership mode for trainable master weights in low-bit persistent paths",
    )
    parser.add_argument(
        "--fp8-rounding",
        type=str,
        default=None,
        choices=["nearest", "stochastic"],
        help="Rounding mode for FP8 quantization recipe",
    )
    parser.add_argument(
        "--fp8-activation-granularity",
        type=str,
        default=None,
        choices=["tile_1x128", "tensor", "channel"],
        help="Activation quantization granularity for FP8 recipe",
    )
    parser.add_argument(
        "--fp8-weight-granularity",
        type=str,
        default=None,
        choices=["block_128x128", "tensor", "channel"],
        help="Weight quantization granularity for FP8 recipe",
    )
    parser.add_argument(
        "--fp8-comm-quant",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable FP8 communication quantization in MoE dispatch/combine",
    )
    parser.add_argument(
        "--fp8-comm-granularity",
        type=str,
        default=None,
        choices=["block_128x128", "tensor", "channel"],
        help="Communication quantization granularity for FP8 recipe",
    )


def normalize_and_resolve_precision(
    args: argparse.Namespace,
    device: torch.device,
) -> PrecisionConfig:
    """Resolve the shared precision config from CLI args and runtime device."""
    return resolve_precision_config(args, device)
