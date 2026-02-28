"""Example-script CLI wiring tests for per-module low-bit precision flags."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module



def test_example_scripts_parse_new_precision_flags(monkeypatch) -> None:
    repo_root = Path(__file__).parent.parent

    train_4d = _load_module(repo_root / "examples" / "train_4d.py", "train_4d_precision_wiring")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_4d.py",
            "--tensor-model-parallel-size",
            "1",
            "--pipeline-model-parallel-size",
            "1",
            "--expert-model-parallel-size",
            "1",
            "--fp8-param",
            "--compute-lowbit-include",
            "lm_head",
        ],
    )
    args = train_4d.parse_args()
    assert args.fp8_param is True
    assert args.compute_lowbit_include == ["lm_head"]

    tp = _load_module(repo_root / "examples" / "train_tp.py", "tp_precision_wiring")
    monkeypatch.setattr(sys, "argv", ["train_tp.py", "--fp4-param", "--module-pattern-type", "glob"])
    args = tp.parse_args()
    assert args.fp4_param is True
    assert args.module_pattern_type == "glob"

    ddp = _load_module(repo_root / "examples" / "train_ddp.py", "ddp_precision_wiring")
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_ddp.py", "--persistent-lowbit-mode", "fp8", "--compute-lowbit-mode", "fp8"],
    )
    args = ddp.parse_args()
    assert args.persistent_lowbit_mode == "fp8"
    assert args.compute_lowbit_mode == "fp8"

    mvp = _load_module(repo_root / "examples" / "train_mvp.py", "mvp_precision_wiring")
    monkeypatch.setattr(sys, "argv", ["train_mvp.py", "--fp4", "--fp4-param-format", "nf4"])
    args = mvp.parse_args()
    assert args.fp4 is True
    assert args.fp4_param_format == "nf4"


def test_example_scripts_wire_strict_lowbit_assignment_guard() -> None:
    repo_root = Path(__file__).parent.parent
    scripts = (
        "train_4d.py",
        "train_tp.py",
        "train_ddp.py",
        "train_mvp.py",
    )
    for script_name in scripts:
        source = (repo_root / "examples" / script_name).read_text()
        assert "build_module_precision_resolver" in source
        assert "finalize_module_precision_resolver" in source
        assert "build_model_precision_plan" not in source
        assert "apply_model_precision_plan" not in source
