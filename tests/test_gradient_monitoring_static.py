"""
Static regression tests for gradient monitoring integration.

These tests use AST parsing to avoid importing torch. They ensure that
gradient-dependent logging runs before gradients are cleared.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


def _contains_name(node: ast.AST, name: str) -> bool:
    """Return True if the AST subtree contains a Name node matching `name`."""
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == name:
            return True
    return False


def _dotted_name(node: ast.AST) -> str | None:
    """Return a dotted-name string for an Attribute/Name chain (best-effort)."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        return None
    return ".".join(reversed(parts))


def _find_trainer_method(tree: ast.AST, method_name: str) -> ast.FunctionDef:
    """Find `Trainer.<method_name>` in the parsed module AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Trainer":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"Could not find Trainer.{method_name}() in src/trainer.py")


@dataclass(frozen=True)
class _CallSite:
    lineno: int
    guarded_by_log_this_step: bool
    guarded_by_hist_this_step: bool


class _MethodCallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self._log_guard_depth = 0
        self._hist_guard_depth = 0
        self.calls: dict[str, list[_CallSite]] = {
            "self._log_detailed_gradient_stats": [],
            "self._log_weight_update_ratios": [],
            "self._log_gradient_histograms": [],
            "self.optimizer.zero_grad": [],
            "self.runtime_post_backward_metrics": [],
            "self.runtime_apply_optimizer_step": [],
        }

    def visit_If(self, node: ast.If) -> None:
        is_log_guard = _contains_name(node.test, "log_this_step")
        is_hist_guard = _contains_name(node.test, "hist_this_step")
        if is_log_guard:
            self._log_guard_depth += 1
        if is_hist_guard:
            self._hist_guard_depth += 1
        for child in node.body:
            self.visit(child)
        for child in node.orelse:
            self.visit(child)
        if is_log_guard:
            self._log_guard_depth -= 1
        if is_hist_guard:
            self._hist_guard_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted_name(node.func)
        if dotted in self.calls:
            self.calls[dotted].append(
                _CallSite(
                    lineno=int(getattr(node, "lineno", -1)),
                    guarded_by_log_this_step=self._log_guard_depth > 0,
                    guarded_by_hist_this_step=self._hist_guard_depth > 0,
                )
            )
        self.generic_visit(node)


def test_gradient_monitoring_logged_before_zero_grad() -> None:
    trainer_path = Path(__file__).resolve().parent.parent / "src" / "trainer.py"
    code = trainer_path.read_text(encoding="utf-8")
    tree = ast.parse(code)

    training_step = _find_trainer_method(tree, "training_step")
    runtime_post_backward_metrics = _find_trainer_method(tree, "runtime_post_backward_metrics")
    runtime_apply_optimizer_step = _find_trainer_method(tree, "runtime_apply_optimizer_step")

    metrics_collector = _MethodCallCollector()
    metrics_collector.visit(runtime_post_backward_metrics)

    def _min_line(name: str) -> int:
        if not metrics_collector.calls[name]:
            raise AssertionError(f"Expected call to {name} in Trainer.runtime_post_backward_metrics()")
        return min(call.lineno for call in metrics_collector.calls[name])

    apply_collector = _MethodCallCollector()
    apply_collector.visit(runtime_apply_optimizer_step)
    if not apply_collector.calls["self.optimizer.zero_grad"]:
        raise AssertionError("Expected call to self.optimizer.zero_grad in runtime_apply_optimizer_step")
    zero_grad_line = min(call.lineno for call in apply_collector.calls["self.optimizer.zero_grad"])

    training_step_collector = _MethodCallCollector()
    training_step_collector.visit(training_step)
    post_backward_line = min(
        call.lineno for call in training_step_collector.calls["self.runtime_post_backward_metrics"]
    )
    apply_step_line = min(
        call.lineno for call in training_step_collector.calls["self.runtime_apply_optimizer_step"]
    )
    assert post_backward_line < apply_step_line, (
        "Expected training_step() to invoke runtime_post_backward_metrics() before "
        f"runtime_apply_optimizer_step() (got {post_backward_line} >= {apply_step_line})"
    )

    detailed_stats_line = _min_line("self._log_detailed_gradient_stats")
    update_ratios_line = _min_line("self._log_weight_update_ratios")
    histograms_line = _min_line("self._log_gradient_histograms")

    assert detailed_stats_line < zero_grad_line, (
        "Expected _log_detailed_gradient_stats() to run before optimizer.zero_grad() "
        f"(got {detailed_stats_line} >= {zero_grad_line})"
    )
    assert update_ratios_line < zero_grad_line, (
        "Expected _log_weight_update_ratios() to run before optimizer.zero_grad() "
        f"(got {update_ratios_line} >= {zero_grad_line})"
    )
    assert histograms_line < zero_grad_line, (
        "Expected _log_gradient_histograms() to run before optimizer.zero_grad() "
        f"(got {histograms_line} >= {zero_grad_line})"
    )

    for name in [
        "self._log_detailed_gradient_stats",
        "self._log_weight_update_ratios",
    ]:
        unguarded = [
            call.lineno for call in metrics_collector.calls[name] if not call.guarded_by_log_this_step
        ]
        assert not unguarded, (
            f"Expected {name} to be guarded by log_this_step (unguarded at {unguarded})"
        )

    unguarded_hist = [
        call.lineno
        for call in metrics_collector.calls["self._log_gradient_histograms"]
        if not call.guarded_by_hist_this_step
    ]
    assert not unguarded_hist, (
        "Expected self._log_gradient_histograms to be guarded by hist_this_step "
        f"(unguarded at {unguarded_hist})"
    )
