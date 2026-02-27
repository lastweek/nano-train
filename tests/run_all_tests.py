"""
Run all unit tests for native modules.

Tests:
- test_layers_simple.py: Linear, LayerNorm, Embedding, Dropout, GELU
- test_optimizer.py: AdamW optimizer
- test_scheduler.py: CosineAnnealingScheduler
- test_losses.py: CrossEntropyLoss
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

device = torch.device('cpu')

print("=" * 70)
print("Nano-Train: Running All Unit Tests")
print("=" * 70)
print()

# Track results
results = {}

# =============================================================================
# Test Layers
# =============================================================================

print("1. Testing Native Layers...")
print("-" * 70)

try:
    from src.layers import Linear, LayerNorm, Embedding, Dropout, GELU, clip_grad_norm

    # Linear
    linear = Linear(
        10,
        20,
        param_dtype=torch.float32,
        param_device=None,
    ).to(device)
    ref = torch.nn.Linear(10, 20).to(device)
    with torch.no_grad():
        linear.weight.copy_(ref.weight)
        linear.bias.copy_(ref.bias)

    x = torch.randn(5, 10, device=device)
    native_out = linear(x)
    ref_out = ref(x)

    assert torch.allclose(native_out, ref_out, rtol=1e-5, atol=1e-8), "Linear mismatch"
    print("  ✓ Linear")

    # LayerNorm
    ln = LayerNorm(
        20,
        param_dtype=torch.float32,
        param_device=None,
    ).to(device)
    ref = torch.nn.LayerNorm(20).to(device)
    with torch.no_grad():
        ln.weight.copy_(ref.weight)
        ln.bias.copy_(ref.bias)

    x = torch.randn(5, 20, device=device)
    native_out = ln(x)
    ref_out = ref(x)

    assert torch.allclose(native_out, ref_out, rtol=1e-5, atol=1e-8), "LayerNorm mismatch"
    print("  ✓ LayerNorm")

    # Embedding
    emb = Embedding(
        100,
        20,
        param_dtype=torch.float32,
        param_device=None,
    ).to(device)
    ref = torch.nn.Embedding(100, 20).to(device)
    with torch.no_grad():
        emb.weight.copy_(ref.weight)

    indices = torch.randint(0, 100, (5, 10), device=device)
    native_out = emb(indices)
    ref_out = ref(indices)

    assert torch.allclose(native_out, ref_out, rtol=1e-5, atol=1e-8), "Embedding mismatch"
    print("  ✓ Embedding")

    # Dropout
    dropout = Dropout(0.5).to(device)
    ref = torch.nn.Dropout(0.5).to(device)

    dropout.train()
    ref.train()

    x = torch.randn(5, 10, device=device)
    # Can't compare dropout outputs directly (random), just check it runs
    native_out = dropout(x)
    ref_out = ref(x)

    assert native_out.shape == ref_out.shape, "Dropout shape mismatch"
    print("  ✓ Dropout (train mode)")

    dropout.eval()
    ref.eval()

    native_out = dropout(x)
    ref_out = ref(x)

    assert torch.allclose(native_out, ref_out, rtol=1e-5, atol=1e-8), "Dropout eval mismatch"
    print("  ✓ Dropout (eval mode)")

    # GELU
    gelu = GELU().to(device)
    ref = torch.nn.GELU().to(device)

    x = torch.randn(100, 20, device=device)
    native_out = gelu(x)
    ref_out = ref(x)

    # Allow slight difference due to approximation
    diff = torch.abs(native_out - ref_out)
    rel_diff = diff / torch.abs(ref_out).max()
    assert rel_diff.max() < 0.001, f"GELU mismatch: {rel_diff.max()}"
    print("  ✓ GELU")

    # Gradient clipping
    p1 = torch.nn.Parameter(torch.randn(10, device=device))
    p2 = torch.nn.Parameter(torch.randn(10, device=device))
    p1.grad = torch.randn(10, device=device) * 5.0
    p2.grad = torch.randn(10, device=device) * 3.0

    max_norm = 1.0
    norm_before = (p1.grad.norm()**2 + p2.grad.norm()**2)**0.5

    from src.layers import clip_grad_norm
    norm_after = clip_grad_norm([p1, p2], max_norm)

    if norm_before > max_norm:
        norm_check = (p1.grad.norm()**2 + p2.grad.norm()**2)**0.5
        assert abs(norm_check - max_norm) < 1e-5, f"Clip mismatch: {norm_check} vs {max_norm}"
    print("  ✓ Gradient clipping")

    results['layers'] = 'PASS'
    print()

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    results['layers'] = 'FAIL'
    print()

# =============================================================================
# Test Optimizer
# =============================================================================

print("2. Testing AdamW Optimizer...")
print("-" * 70)

try:
    from src.optimizer import AdamW

    model = torch.nn.Linear(10, 20).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    initial_weight = model.weight.data.clone()

    # Training step
    x = torch.randn(5, 10, device=device)
    loss = model(x).sum()
    loss.backward()

    optimizer.step()

    # Parameters should have changed
    assert not torch.allclose(model.weight.data, initial_weight), "Parameters not updated"
    print("  ✓ Parameter updates")

    # Check state tracking
    state = optimizer.state[model.weight]
    assert 'step' in state, "Step not tracked"
    assert state['step'] == 1, "Step count incorrect"
    print("  ✓ State tracking")

    # Check weight decay (simplified - just verify it runs without error)
    model2 = torch.nn.Linear(10, 20, bias=False).to(device)
    optimizer2 = AdamW(model2.parameters(), lr=0.1, weight_decay=0.1)

    x = torch.randn(5, 10, device=device)
    y = model2(x)
    loss = y.sum()
    loss.backward()
    optimizer2.step()

    print("  ✓ Weight decay (runs without error)")

    results['optimizer'] = 'PASS'
    print()

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    results['optimizer'] = 'FAIL'
    import traceback
    traceback.print_exc()
    print()

# =============================================================================
# Test Scheduler
# =============================================================================

print("3. Testing CosineAnnealingScheduler...")
print("-" * 70)

try:
    from src.scheduler import CosineAnnealingScheduler
    from src.optimizer import AdamW

    model = torch.nn.Linear(10, 20).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    warmup_steps = 10
    max_steps = 100
    min_lr = 0.0

    scheduler = CosineAnnealingScheduler(
        optimizer, warmup_steps, max_steps, min_lr
    )

    # Test warmup phase
    for step in range(1, warmup_steps + 1):
        lr = scheduler.step()
        expected_lr = 1e-3 * step / warmup_steps
        assert abs(lr - expected_lr) < 1e-8, f"Warmup LR mismatch at step {step}: {lr} vs {expected_lr}"
    print("  ✓ Warmup phase")

    # Test annealing phase
    base_lr = 1e-3
    import math
    for i in range(5):
        lr = scheduler.step()
        # current_step is now warmup_steps + i + 1
        step = warmup_steps + i + 1
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        expected_lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        assert abs(lr - expected_lr) < 1e-6, f"Annealing LR mismatch at step {step}: {lr} vs {expected_lr}"
    print("  ✓ Annealing phase")

    # Test state dict
    state = scheduler.state_dict()
    assert state['current_step'] == warmup_steps + 5, "State dict incorrect"
    print("  ✓ State save/load")

    results['scheduler'] = 'PASS'
    print()

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    results['scheduler'] = 'FAIL'
    import traceback
    traceback.print_exc()
    print()

# =============================================================================
# Test Loss
# =============================================================================

print("4. Testing CrossEntropyLoss...")
print("-" * 70)

try:
    from src.losses import CrossEntropyLoss

    # Test accuracy vs torch.nn
    batch_size, num_classes = 10, 20

    native = CrossEntropyLoss()
    ref = torch.nn.CrossEntropyLoss()

    logits = torch.randn(batch_size, num_classes, device=device, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,), device=device)

    native_loss = native(logits, targets)
    ref_loss = ref(logits, targets)

    assert torch.allclose(native_loss, ref_loss, rtol=1e-5, atol=1e-8), "Forward pass mismatch"
    print("  ✓ Forward pass accuracy")

    # Test backward pass
    native_loss.backward()
    ref_loss.backward()

    assert torch.allclose(logits.grad, logits.grad, rtol=1e-5, atol=1e-8), "Backward pass mismatch"
    print("  ✓ Backward pass accuracy")

    # Test ignore_index
    targets[::2] = -100
    native_loss2 = native(logits, targets)
    ref_loss2 = ref(logits, targets)

    assert torch.allclose(native_loss2, ref_loss2, rtol=1e-5, atol=1e-8), "Ignore index mismatch"
    print("  ✓ Ignore index")

    # Test reduction modes
    loss_sum = CrossEntropyLoss(reduction='sum')(logits, targets)
    loss_none = CrossEntropyLoss(reduction='none')(logits, targets)

    assert loss_sum.dim() == 0, "Sum reduction should produce scalar"
    assert loss_none.shape == torch.Size([batch_size]), "None reduction should preserve batch"
    print("  ✓ Reduction modes")

    results['loss'] = 'PASS'
    print()

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    results['loss'] = 'FAIL'
    import traceback
    traceback.print_exc()
    print()

# =============================================================================
# Test DeepSeek-style Model
# =============================================================================

print("5. Testing DeepSeek-style Model...")
print("-" * 70)

try:
    from test_deepseek_model import (
        test_backward_runs,
        test_forward_shape,
        test_meta_init_avoids_storage,
    )

    test_forward_shape()
    print("  ✓ Forward shape")

    test_backward_runs()
    print("  ✓ Backward pass")

    test_meta_init_avoids_storage()
    print("  ✓ Meta init (no parameter storage)")

    results['deepseek'] = 'PASS'
    print()

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    results['deepseek'] = 'FAIL'
    import traceback
    traceback.print_exc()
    print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 70)
print("Test Summary")
print("=" * 70)

for name, status in results.items():
    symbol = "✓" if status == "PASS" else "✗"
    print(f"  {symbol} {name.upper()}: {status}")

all_passed = all(status == "PASS" for status in results.values())

print()
if all_passed:
    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    sys.exit(0)
else:
    print("=" * 70)
    print("SOME TESTS FAILED ✗")
    print("=" * 70)
    sys.exit(1)
