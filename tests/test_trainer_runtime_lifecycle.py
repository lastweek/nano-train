"""Tests for MVP runtime lifecycle hooks backed by Trainer-compatible objects."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from src.runtime.context import RuntimeContext
from src.runtime.context import TrainState
from src.runtime.contracts import OptimizerState
from src.runtime.sync import ParamShardInfo


def _load_mvp_module():
    repo_root = Path(__file__).parent.parent
    module_path = repo_root / "examples" / "train_mvp.py"
    spec = importlib.util.spec_from_file_location("mvp_runtime_lifecycle_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load examples/train_mvp.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mvp_runtime_lifecycle_test"] = module
    spec.loader.exec_module(module)
    return module


class _FakeTrainer:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            monitoring=SimpleNamespace(probe_steps=10, tensorboard_flush_on_log=True)
        )
        self.events: list[str] = []

    def _run_fixed_probe(self, *, step: int) -> None:
        self.events.append(f"probe:{step}")

    def _flush_writer(self) -> None:
        self.events.append("flush")

    def save_checkpoint(self, step: int, final: bool = False) -> None:
        self.events.append(f"save:{step}:{final}")

    def _finalize_training(self, *, step: int, totals, elapsed_seconds: float, final_loss: float) -> None:
        del totals, elapsed_seconds, final_loss
        self.events.append(f"finalize:{step}")


def test_mvp_checkpoint_manager_load_and_run_end_call_expected_trainer_hooks() -> None:
    mvp = _load_mvp_module()
    manager = mvp.MVPCheckpointManager()

    fake_trainer = _FakeTrainer()
    optimizer_state = OptimizerState(
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1),
        shard_info=ParamShardInfo(set(), set()),
        extra_state={
            "trainer": fake_trainer,
            "totals": mvp._TrainTotals(),
            "start_time": 0.0,
            "last_loss": 1.0,
        },
    )

    ctx = RuntimeContext(parallel=SimpleNamespace(), mode="single", run_config=SimpleNamespace())
    manager.load(model=torch.nn.Linear(1, 1), optimizer_state=optimizer_state, ctx=ctx)
    manager.on_run_end(
        model=torch.nn.Linear(1, 1),
        optimizer_state=optimizer_state,
        state=TrainState(global_step=7),
        ctx=ctx,
    )

    assert "probe:0" in fake_trainer.events
    assert "flush" in fake_trainer.events
    assert "save:7:True" in fake_trainer.events
    assert "finalize:7" in fake_trainer.events
