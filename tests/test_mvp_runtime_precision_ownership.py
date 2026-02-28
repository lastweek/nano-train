"""MVP runtime should own mixed-precision step policy, not Trainer.training_step."""

from __future__ import annotations

from pathlib import Path


def test_mvp_runtime_schedule_owns_precision_step_policy() -> None:
    source = (Path(__file__).resolve().parent.parent / "examples" / "train_mvp.py").read_text(
        encoding="utf-8"
    )
    assert "trainer.training_step(" not in source
    assert "trainer.runtime_forward_loss(" in source
    assert "trainer.runtime_post_backward_metrics(" in source
    assert "precision_controller.backward(" in source
    assert "precision_controller.prepare_optimizer_step(" in source
    assert "trainer.runtime_apply_optimizer_step(" in source
