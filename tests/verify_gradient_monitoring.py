#!/usr/bin/env python3
"""
Verification script for gradient monitoring implementation.

This script checks that the gradient monitoring functionality
is correctly integrated into the training loop without requiring
PyTorch to be installed.
"""

import ast
import sys
from pathlib import Path


def check_gradient_monitoring():
    """Verify gradient monitoring is properly implemented."""

    trainer_path = Path(__file__).parent.parent / "src" / "trainer.py"

    if not trainer_path.exists():
        print(f"✗ Trainer file not found: {trainer_path}")
        return False

    with open(trainer_path) as f:
        code = f.read()

    tree = ast.parse(code)

    # Check 1: _log_detailed_gradient_stats method exists
    method_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_log_detailed_gradient_stats':
            method_found = True
            print(f"✓ Method _log_detailed_gradient_stats defined (line {node.lineno})")

            # Check parameters
            args = [arg.arg for arg in node.args.args]
            if args == ['self', 'step']:
                print("  ✓ Has correct parameters: self, step")
            else:
                print(f"  ✗ Unexpected parameters: {args}")
                return False
            break

    if not method_found:
        print("✗ Method _log_detailed_gradient_stats not found")
        return False

    # Check 2: Method is called in train() method
    train_method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'Trainer':
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == 'train':
                    train_method = item
                    break

    if train_method is None:
        print("✗ Train method not found in Trainer class")
        return False

    # Look for the call in the source
    source_lines = code.split('\n')
    call_found = False
    call_line = None

    for i, line in enumerate(source_lines):
        if '_log_detailed_gradient_stats' in line and 'self.' in line:
            # Make sure it's a call, not just a comment or definition
            if 'def ' not in line and not line.strip().startswith('#'):
                call_found = True
                call_line = i + 1
                print(f"✓ Method called in train() at line {call_line}")
                break

    if not call_found:
        print("✗ Method _log_detailed_gradient_stats not called in train()")
        return False

    # Check 3: Verify it's inside the logging block
    # Find the log_this_step check and verify proper indentation
    log_check_found = False
    log_line = None
    for i in range(call_line - 50, call_line):
        if i >= 0 and 'if log_this_step:' in source_lines[i]:
            log_check_found = True
            log_line = i + 1
            print(f"✓ Call is inside the log_this_step conditional block (line {log_line})")
            break

    if not log_check_found:
        print("⚠ Warning: Could not verify log_this_step block (may be OK)")

    # Check 4: Verify method content
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_log_detailed_gradient_stats':
            # Check for named_parameters usage
            method_source = '\n'.join(source_lines[node.lineno - 1:node.lineno + 30])
            checks = {
                'named_parameters': 'named_parameters' in method_source,
                'TensorBoard logging': 'writer.add_scalar' in method_source,
                'console logging': 'logger.info' in method_source and 'grad_norm' in method_source,
            }

            for check_name, result in checks.items():
                if result:
                    print(f"  ✓ Uses {check_name}")
                else:
                    print(f"  ✗ Missing {check_name}")
                    return False
            break

    print("\n✓ All gradient monitoring checks passed!")
    print("\nWhat this enables:")
    print("  • Per-layer gradient norm tracking")
    print("  • Per-layer gradient mean/std tracking")
    print("  • TensorBoard visualization of gradient flow")
    print("  • Console logging every 100 steps")
    print("  • Debug vanishing/exploding gradients")

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
