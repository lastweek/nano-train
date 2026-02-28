"""
Simple unit tests for native layer implementations (no conftest dependency).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import math
from itertools import count

from src.runtime.contracts import PrecisionConfig
from src.runtime.mixed_precision import build_module_precision_resolver


_MODULE_COUNTER = count()


def _next_module_path(prefix: str) -> str:
    return f"tests.simple.{prefix}.{next(_MODULE_COUNTER)}"


def _resolver():
    return build_module_precision_resolver(PrecisionConfig(mode="fp32"))


def assert_tensor_close(actual, expected,
                      rtol=1e-5, atol=1e-6,
                      msg=None):
    """Assert two tensors are close within tolerances."""
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} vs {expected.shape}")

    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = torch.abs(actual - expected)
        max_diff = diff.max().item()
        raise AssertionError(f"{msg or 'Tensors not close'}: max diff = {max_diff}")


def test_linear():
    """Test Linear layer."""
    print("Testing Linear...")
    device = torch.device('cpu')

    from src.layers import Linear

    native = Linear(
        10,
        20,
        param_dtype=torch.float32,
        param_device=None,
        module_path=_next_module_path("linear"),
        precision_resolver=_resolver(),
    ).to(device)
    ref = nn.Linear(10, 20).to(device)

    with torch.no_grad():
        native.weight.copy_(ref.weight)
        native.bias.copy_(ref.bias)

    x = torch.randn(5, 10, device=device)
    native_out = native(x)
    ref_out = ref(x)

    assert_tensor_close(native_out, ref_out, msg="Linear mismatch")
    print("  ✓ Linear passed")


def test_layer_norm():
    """Test LayerNorm layer."""
    print("Testing LayerNorm...")
    device = torch.device('cpu')

    from src.layers import LayerNorm

    native = LayerNorm(
        20,
        param_dtype=torch.float32,
        param_device=None,
        module_path=_next_module_path("layernorm"),
        precision_resolver=_resolver(),
    ).to(device)
    ref = nn.LayerNorm(20).to(device)

    with torch.no_grad():
        native.weight.copy_(ref.weight)
        native.bias.copy_(ref.bias)

    x = torch.randn(5, 20, device=device)
    native_out = native(x)
    ref_out = ref(x)

    assert_tensor_close(native_out, ref_out, msg="LayerNorm mismatch")
    print("  ✓ LayerNorm passed")


def test_embedding():
    """Test Embedding layer."""
    print("Testing Embedding...")
    device = torch.device('cpu')

    from src.layers import Embedding

    native = Embedding(
        100,
        20,
        param_dtype=torch.float32,
        param_device=None,
        module_path=_next_module_path("embedding"),
        precision_resolver=_resolver(),
    ).to(device)
    ref = nn.Embedding(100, 20).to(device)

    with torch.no_grad():
        native.weight.copy_(ref.weight)

    indices = torch.randint(0, 100, (5, 10), device=device)
    native_out = native(indices)
    ref_out = ref(indices)

    assert_tensor_close(native_out, ref_out, msg="Embedding mismatch")
    print("  ✓ Embedding passed")


def test_dropout():
    """Test Dropout layer."""
    print("Testing Dropout...")
    device = torch.device('cpu')

    from src.layers import Dropout

    p = 0.5
    native = Dropout(p).to(device)
    ref = nn.Dropout(p).to(device)

    native.eval()
    ref.eval()

    x = torch.randn(5, 10, device=device)
    native_out = native(x)
    ref_out = ref(x)

    assert_tensor_close(native_out, ref_out, msg="Dropout mismatch (eval mode)")
    assert_tensor_close(native_out, x, msg="Dropout eval not identity")
    print("  ✓ Dropout passed")


def test_gelu():
    """Test GELU activation."""
    print("Testing GELU...")
    device = torch.device('cpu')

    from src.layers import GELU

    native = GELU().to(device)
    ref = nn.GELU().to(device)

    x = torch.randn(100, 20, device=device)
    native_out = native(x)
    ref_out = ref(x)

    # GELU approximation may differ slightly from PyTorch
    if not torch.allclose(native_out, ref_out, rtol=1e-4, atol=1e-6):
        diff = torch.abs(native_out - ref_out)
        max_diff = diff.max().item()
        rel_diff = max_diff / torch.abs(ref_out).max().item()
        # Allow up to 0.1% relative difference
        if rel_diff > 0.001:
            raise AssertionError(f"GELU mismatch: max diff = {max_diff}, rel = {rel_diff}")
    print("  ✓ GELU passed")
    print("  ✓ GELU passed")


def test_integration():
    """Test layers working together."""
    print("Testing Integration (MLP block)...")
    device = torch.device('cpu')

    from src.layers import Linear, GELU, Dropout

    # Native MLP
    native = torch.nn.Sequential(
        Linear(
            10,
            20,
            param_dtype=torch.float32,
            param_device=None,
            module_path=_next_module_path("linear"),
            precision_resolver=_resolver(),
        ),
        GELU(),
        Dropout(0.0),
        Linear(
            20,
            15,
            param_dtype=torch.float32,
            param_device=None,
            module_path=_next_module_path("linear"),
            precision_resolver=_resolver(),
        )
    ).to(device)

    # Reference MLP
    ref = torch.nn.Sequential(
        nn.Linear(10, 20),
        nn.GELU(),
        nn.Dropout(0.0),
        nn.Linear(20, 15)
    ).to(device)

    # Copy weights
    with torch.no_grad():
        native[0].weight.copy_(ref[0].weight)
        native[0].bias.copy_(ref[0].bias)
        native[3].weight.copy_(ref[3].weight)
        native[3].bias.copy_(ref[3].bias)

    x = torch.randn(5, 10, device=device)
    native_out = native(x)
    ref_out = ref(x)

    # Integration may accumulate small differences
    if not torch.allclose(native_out, ref_out, rtol=1e-5, atol=1e-8):
        diff = torch.abs(native_out - ref_out)
        max_diff = diff.max().item()
        rel_diff = max_diff / torch.abs(ref_out).max().item()
        if rel_diff > 0.001:
            raise AssertionError(f"Integration mismatch: max diff = {max_diff}, rel = {rel_diff}")
    print("  ✓ Integration passed")
    print("  ✓ Integration passed")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running Native Layer Tests")
    print("=" * 60)

    test_linear()
    test_layer_norm()
    test_embedding()
    test_dropout()
    test_gelu()
    test_integration()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60 + "\n")
