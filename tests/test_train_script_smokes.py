"""Lightweight smoke checks for runtime entry scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from src.runtime.contracts import RuntimeComponents


def _load_script(script_name: str, module_name: str) -> ModuleType:
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / "examples" / script_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_train_4d_components_bootstrap_smoke(monkeypatch) -> None:
    module = _load_script("train_4d.py", "train_4d_smoke_module")
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
            "--max_steps",
            "1",
            "--epochs",
            "1",
        ],
    )
    args = module.parse_args()
    components = module.build_train_4d_components()
    assert isinstance(components, RuntimeComponents)
    ctx = components.bootstrap.build_context(args)
    assert ctx.run_config.precision_config is not None


def test_train_tp_components_bootstrap_smoke(monkeypatch) -> None:
    module = _load_script("train_tp.py", "train_tp_smoke_module")
    monkeypatch.setattr(sys, "argv", ["train_tp.py", "--max_steps", "1", "--epochs", "1"])
    args = module.parse_args()
    components = module.build_tp_components()
    assert isinstance(components, RuntimeComponents)
    ctx = components.bootstrap.build_context(args)
    assert ctx.run_config.precision_config is not None


def test_train_ddp_components_bootstrap_smoke(monkeypatch) -> None:
    module = _load_script("train_ddp.py", "train_ddp_smoke_module")
    monkeypatch.setattr(sys, "argv", ["train_ddp.py", "--max_steps", "1", "--epochs", "1"])
    args = module.parse_args()
    components = module.build_ddp_components()
    assert isinstance(components, RuntimeComponents)
    ctx = components.bootstrap.build_context(args)
    assert ctx.run_config.precision_config is not None


def test_train_mvp_components_bootstrap_smoke(monkeypatch) -> None:
    module = _load_script("train_mvp.py", "train_mvp_smoke_module")
    monkeypatch.setattr(sys, "argv", ["train_mvp.py", "--max_steps", "1"])
    args = module.parse_args()
    components = module.build_mvp_components()
    assert isinstance(components, RuntimeComponents)
    ctx = components.bootstrap.build_context(args)
    assert ctx.run_config.precision_config is not None
