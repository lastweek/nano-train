"""Wiring tests for example-local runtime components."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from src.runtime.contracts import RuntimeComponents


def _load_example_module(script_name: str, module_name: str) -> ModuleType:
    repo_root = Path(__file__).parent.parent
    module_path = repo_root / "examples" / script_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_train_4p_components_are_local_to_script() -> None:
    repo_root = Path(__file__).parent.parent
    text = (repo_root / "examples" / "train_4d.py").read_text(encoding="utf-8")
    assert "src.runtime.plugins" not in text

    module = _load_example_module("train_4d.py", "train_4p_runtime_wiring")
    components = module.build_train_4p_components()
    assert isinstance(components, RuntimeComponents)
    assert components.bootstrap is not None


def test_tp_components_builder_exists_and_returns_runtime_components() -> None:
    module = _load_example_module("train_tp.py", "tp_runtime_wiring")
    components = module.build_tp_components()
    assert isinstance(components, RuntimeComponents)
    assert components.bootstrap is not None


def test_ddp_components_builder_exists_and_returns_runtime_components() -> None:
    module = _load_example_module("train_ddp.py", "ddp_runtime_wiring")
    components = module.build_ddp_components()
    assert isinstance(components, RuntimeComponents)
    assert components.bootstrap is not None


def test_mvp_components_builder_exists_and_returns_runtime_components() -> None:
    module = _load_example_module("train_mvp.py", "mvp_runtime_wiring")
    components = module.build_mvp_components()
    assert isinstance(components, RuntimeComponents)
    assert components.bootstrap is not None
