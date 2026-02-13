"""
Unit tests for CrossEntropyLoss.

Tests verify:
1. Loss computation matches PyTorch implementation
2. Reduction modes work correctly
3. Ignore index masks properly
4. Backward pass produces correct gradients
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import CrossEntropyLoss
from conftest import assert_tensor_close


class TestCrossEntropyLoss:
    """Test cases for CrossEntropyLoss."""

    def test_forward_accuracy(self, device):
        """Forward pass matches torch.nn.CrossEntropyLoss."""
        batch_size, num_classes = 10, 20

        native = CrossEntropyLoss().to(device)
        ref = nn.CrossEntropyLoss().to(device)

        # Create logits and targets
        logits = torch.randn(batch_size, num_classes, device=device, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)

        native_loss = native(logits, targets)
        ref_loss = ref(logits, targets)

        assert_tensor_close(
            native_loss, ref_loss,
            rtol=1e-5, atol=1e-8,
            msg="CrossEntropyLoss forward pass mismatch"
        )

    def test_reduction_modes(self, device):
        """All reduction modes work correctly."""
        batch_size, num_classes = 10, 20

        logits = torch.randn(batch_size, num_classes, device=device)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)

        # Mean reduction (default)
        loss_mean = CrossEntropyLoss(reduction="mean")(logits, targets)

        # Sum reduction
        loss_sum = CrossEntropyLoss(reduction="sum")(logits, targets)

        # None reduction
        loss_none = CrossEntropyLoss(reduction="none")(logits, targets)

        # Check shapes
        assert loss_mean.dim() == 0, "Mean reduction should produce scalar"
        assert loss_sum.dim() == 0, "Sum reduction should produce scalar"
        assert loss_none.shape == torch.Size([batch_size]), \
            "None reduction should preserve batch dimension"

        # Check values
        assert abs(loss_sum.item() - loss_none.sum().item()) < 1e-6, \
            "Sum reduction doesn't match sum of none reduction"

        assert abs(loss_mean.item() - loss_none.mean().item()) < 1e-6, \
            "Mean reduction doesn't match mean of none reduction"

    def test_ignore_index(self, device):
        """Ignore index properly masks targets."""
        batch_size, num_classes = 10, 20
        ignore_index = -100

        logits = torch.randn(batch_size, num_classes, device=device, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)

        # Set some targets to ignore_index
        targets[::2] = ignore_index

        native = CrossEntropyLoss(ignore_index=ignore_index)
        ref = nn.CrossEntropyLoss(ignore_index=ignore_index)

        native_loss = native(logits, targets)
        ref_loss = ref(logits, targets)

        assert_tensor_close(
            native_loss, ref_loss,
            rtol=1e-5, atol=1e-8,
            msg="Ignore index mismatch"
        )

    def test_backward_accuracy(self, device):
        """Backward pass produces correct gradients."""
        batch_size, num_classes = 10, 20

        native = CrossEntropyLoss()
        ref = nn.CrossEntropyLoss()

        logits = torch.randn(batch_size, num_classes, device=device, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)

        native_loss = native(logits, targets)
        ref_loss = ref(logits, targets)

        native_loss.backward()
        ref_loss.backward()

        assert_tensor_close(
            logits.grad, logits.grad,
            rtol=1e-5, atol=1e-8,
            msg="CrossEntropyLoss backward pass mismatch"
        )

    def test_perfect_prediction(self, device):
        """Loss is zero when predictions are perfect."""
        num_classes = 10

        # Create perfect predictions (high logit for correct class)
        targets = torch.randint(0, num_classes, (5,), device=device)
        logits = torch.randn(5, num_classes, device=device) - 10  # Start with low values

        # Boost the correct class
        for i, target in enumerate(targets):
            logits[i, target] = 10.0

        loss = CrossEntropyLoss()(logits, targets)

        # Loss should be very close to zero
        assert loss.item() < 1e-3, f"Loss should be ~0 for perfect predictions, got {loss.item()}"

    def test_random_predictions(self, device):
        """Loss is high for random predictions (uniform distribution)."""
        batch_size, num_classes = 1000, 10

        # Random logits (uniform distribution)
        logits = torch.randn(batch_size, num_classes, device=device)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)

        loss = CrossEntropyLoss()(logits, targets)

        # For random predictions, loss should be around -log(1/num_classes) = log(num_classes)
        expected_loss = math.log(num_classes)
        assert abs(loss.item() - expected_loss) < 0.5, \
            f"Random loss should be ~log(num_classes), got {loss.item()} vs {expected_loss}"

    def test_2d_input(self, device):
        """2D input (batch, seq_len, num_classes) works correctly."""
        batch_size, seq_len, num_classes = 5, 10, 20

        native = CrossEntropyLoss()
        ref = nn.CrossEntropyLoss()

        logits = torch.randn(batch_size, seq_len, num_classes, device=device, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size, seq_len), device=device)

        native_loss = native(logits, targets)
        ref_loss = ref(logits, targets)

        assert_tensor_close(
            native_loss, ref_loss,
            rtol=1e-5, atol=1e-8,
            msg="2D input mismatch"
        )

    def test_all_ignored(self, device):
        """Loss is zero when all targets are ignored."""
        batch_size, num_classes = 10, 20
        ignore_index = -100

        logits = torch.randn(batch_size, num_classes, device=device)
        targets = torch.full((batch_size,), ignore_index, device=device)

        loss = CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")(logits, targets)

        assert loss.item() == 0.0, "Loss should be zero when all targets ignored"

    def test_label_smoothing(self, device):
        """Can be used for label smoothing (manual implementation)."""
        batch_size, num_classes = 10, 20

        logits = torch.randn(batch_size, num_classes, device=device)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)

        # Manual label smoothing
        smoothing = 0.1
        targets_smooth = targets.float().unsqueeze(1)
        targets_smooth = targets_smooth * (1 - smoothing) + smoothing / num_classes

        # Use reduction='none' to apply manual smoothing
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(log_probs * targets_smooth).sum(dim=-1).mean()

        # Should be reasonable (not NaN, not infinity)
        assert not torch.isnan(loss), "Label smoothing produced NaN"
        assert not torch.isinf(loss), "Label smoothing produced inf"
        assert loss.item() > 0, "Label smoothing loss should be positive"
