"""Shared mixed-precision parser coverage for all runtime scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.runtime.precision_args import add_mixed_precision_args


def _parse_precision_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_mixed_precision_args(parser)
    return parser.parse_args(argv)


def test_shared_precision_args_default_master_ownership() -> None:
    args = _parse_precision_args([])
    assert args.lowbit_master_ownership == "optimizer"
    assert args.fp8_backend == "transformer_engine"
    assert args.fp4_backend == "emulated"


def test_shared_precision_args_accept_module_policy_flags() -> None:
    args = _parse_precision_args(
        [
            "--fp8",
            "--fp8-backend",
            "emulated",
            "--compute-lowbit-mode",
            "fp8",
            "--compute-lowbit-include",
            ".*mlp.*",
            "--persistent-lowbit-mode",
            "fp4",
            "--persistent-lowbit-include",
            ".*experts.*",
            "--module-compute-dtype-rule",
            ".*norm.*=fp32",
            "--lowbit-master-ownership",
            "module",
        ]
    )
    assert args.fp8 is True
    assert args.compute_lowbit_mode == "fp8"
    assert args.compute_lowbit_include == [".*mlp.*"]
    assert args.persistent_lowbit_mode == "fp4"
    assert args.persistent_lowbit_include == [".*experts.*"]
    assert args.module_compute_dtype_rule == [".*norm.*=fp32"]
    assert args.lowbit_master_ownership == "module"


def test_all_runtime_scripts_use_shared_precision_arg_builder() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    scripts = (
        repo_root / "examples" / "train_4d.py",
        repo_root / "examples" / "train_tp.py",
        repo_root / "examples" / "train_ddp.py",
        repo_root / "examples" / "train_mvp.py",
    )
    for path in scripts:
        source = path.read_text(encoding="utf-8")
        assert "add_mixed_precision_args(parser)" in source
