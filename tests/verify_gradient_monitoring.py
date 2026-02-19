#!/usr/bin/env python3
"""
Verification script for gradient monitoring implementation.

This script checks that the gradient monitoring functionality
is correctly integrated into the training loop without requiring
PyTorch to be installed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _contains_name(node: ast.AST, name: str) -> bool:
    """Return True if the AST subtree contains a Name node matching `name`."""
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == name:
            return True
    return False


def _dotted_name(node: ast.AST) -> str | None:
    """Return a dotted-name string for an Attribute/Name chain (best-effort)."""
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        return None
    return ".".join(reversed(parts))


def check_gradient_monitoring() -> bool:
    """Verify gradient monitoring is properly implemented."""
    trainer_path = Path(__file__).parent.parent / "src" / "trainer.py"
    if not trainer_path.exists():
        print(f"✗ Trainer file not found: {trainer_path}")
        return False

    code = trainer_path.read_text(encoding="utf-8")
    tree = ast.parse(code)

    # Check 1: _log_detailed_gradient_stats method exists
    method_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_log_detailed_gradient_stats":
            method_found = True
            print(f"✓ Method _log_detailed_gradient_stats defined (line {node.lineno})")

            args = [arg.arg for arg in node.args.args]
            if args[:2] == ["self", "step"]:
                print("  ✓ Has correct parameters: self, step")
            else:
                print(f"  ✗ Unexpected parameters: {args}")
                return False
            break

    if not method_found:
        print("✗ Method _log_detailed_gradient_stats not found")
        return False

    # Check 2: Methods are called in training_step() before optimizer.zero_grad()
    trainer_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Trainer":
            trainer_class = node
            break

    if trainer_class is None:
        print("✗ Trainer class not found")
        return False

    training_step_method = None
    for item in trainer_class.body:
        if isinstance(item, ast.FunctionDef) and item.name == "training_step":
            training_step_method = item
            break

    if training_step_method is None:
        print("✗ training_step method not found in Trainer class")
        return False

    class CallCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.log_guard_depth = 0
            self.hist_guard_depth = 0
            self.calls = {
                "self._log_detailed_gradient_stats": [],
                "self._log_weight_update_ratios": [],
                "self._log_gradient_histograms": [],
                "self.optimizer.zero_grad": [],
            }

        def visit_If(self, node: ast.If) -> None:
            is_log_guard = _contains_name(node.test, "log_this_step")
            is_hist_guard = _contains_name(node.test, "hist_this_step")
            if is_log_guard:
                self.log_guard_depth += 1
            if is_hist_guard:
                self.hist_guard_depth += 1
            for child in node.body:
                self.visit(child)
            for child in node.orelse:
                self.visit(child)
            if is_log_guard:
                self.log_guard_depth -= 1
            if is_hist_guard:
                self.hist_guard_depth -= 1

        def visit_Call(self, node: ast.Call) -> None:
            dotted = _dotted_name(node.func)
            if dotted in self.calls:
                self.calls[dotted].append(
                    (
                        int(getattr(node, "lineno", -1)),
                        self.log_guard_depth > 0,
                        self.hist_guard_depth > 0,
                    )
                )
            self.generic_visit(node)

    collector = CallCollector()
    collector.visit(training_step_method)

    def min_line(name: str) -> int:
        if not collector.calls[name]:
            return -1
        return min(lineno for lineno, _guarded, _hist_guarded in collector.calls[name])

    log_line = min_line("self._log_detailed_gradient_stats")
    ratios_line = min_line("self._log_weight_update_ratios")
    hist_line = min_line("self._log_gradient_histograms")
    zero_grad_line = min_line("self.optimizer.zero_grad")

    if log_line < 0:
        print("✗ _log_detailed_gradient_stats not called in training_step()")
        return False
    print(f"✓ _log_detailed_gradient_stats called in training_step() at line {log_line}")

    if ratios_line < 0:
        print("✗ _log_weight_update_ratios not called in training_step()")
        return False
    print(f"✓ _log_weight_update_ratios called in training_step() at line {ratios_line}")

    if hist_line < 0:
        print("✗ _log_gradient_histograms not called in training_step()")
        return False
    print(f"✓ _log_gradient_histograms called in training_step() at line {hist_line}")

    if zero_grad_line < 0:
        print("✗ optimizer.zero_grad not called in training_step()")
        return False
    print(f"✓ optimizer.zero_grad called in training_step() at line {zero_grad_line}")

    for name, call_line in [
        ("_log_detailed_gradient_stats", log_line),
        ("_log_weight_update_ratios", ratios_line),
        ("_log_gradient_histograms", hist_line),
    ]:
        if call_line >= zero_grad_line:
            print(
                f"✗ {name} called after optimizer.zero_grad "
                f"(line {call_line} >= {zero_grad_line})"
            )
            return False
        print(f"  ✓ {name} runs before optimizer.zero_grad")

    for dotted in [
        "self._log_detailed_gradient_stats",
        "self._log_weight_update_ratios",
    ]:
        unguarded = [
            lineno
            for lineno, guarded, _hist_guarded in collector.calls[dotted]
            if not guarded
        ]
        if unguarded:
            print(f"✗ {dotted} is not guarded by log_this_step at lines {unguarded}")
            return False
    print("✓ Scalar monitoring calls are guarded by log_this_step")

    unguarded_hist = [
        lineno
        for lineno, _guarded, hist_guarded in collector.calls["self._log_gradient_histograms"]
        if not hist_guarded
    ]
    if unguarded_hist:
        print(f"✗ self._log_gradient_histograms is not guarded by hist_this_step at lines {unguarded_hist}")
        return False
    print("✓ Histogram monitoring call is guarded by hist_this_step")

    print("\n✓ All gradient monitoring checks passed!")
    print("\nWhat this enables:")
    print("  • Bounded monitoring by default (Standard mode)")
    print("  • Deep-dive per-parameter monitoring (Debug mode)")
    print("  • Reliable grad-dependent metrics (logged before zero_grad)")

    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Gradient Monitoring Implementation Verification")
    print("=" * 70)
    print()

    success = check_gradient_monitoring()

    print()
    print("=" * 70)
    if success:
        print("VERIFICATION PASSED ✓")
        print("=" * 70)
        print("\nGradient monitoring is ready to use!")
        print("Run 'python examples/train_mvp.py' to start training with monitoring.")
        sys.exit(0)
    else:
        print("VERIFICATION FAILED ✗")
        print("=" * 70)
        sys.exit(1)

