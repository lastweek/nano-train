"""
Unit tests for AdamW optimizer.

Tests verify:
1. Parameter updates follow Adam algorithm
2. Weight decay is applied correctly
3. State is tracked properly across steps
4. Matches PyTorch AdamW behavior
"""

import pytest
import torch
import torch.nn as nn

from src.optimizer import AdamW
from conftest import assert_tensor_close


class TestAdamW:
    """Test cases for AdamW optimizer."""

    def test_parameter_update(self, device):
        """Optimizer updates parameters correctly."""
        model = nn.Linear(10, 20).to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        # Get initial parameters
        initial_weight = model.weight.data.clone()

        # Forward/backward
        x = torch.randn(5, 10, device=device)
        loss = model(x).sum()
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Parameters should have changed
        assert not torch.allclose(model.weight.data, initial_weight), \
            "Parameters not updated"

    def test_state_tracking(self, device):
        """Optimizer tracks momentum and variance correctly."""
        model = nn.Linear(10, 20).to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

        # First step
        x = torch.randn(5, 10, device=device)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Check state exists
        state = optimizer.state[model.weight]
        assert 'step' in state, "Step not tracked"
        assert 'exp_avg' in state, "Momentum not tracked"
        assert 'exp_avg_sq' in state, "Variance not tracked"
        assert state['step'] == 1, "Step count incorrect"

        # Second step
        optimizer.zero_grad()
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        assert optimizer.state[model.weight]['step'] == 2, "Step not incremented"

    def test_weight_decay(self, device):
        """Weight decay is applied correctly (AdamW style)."""
        in_features, out_features = 10, 20
        model = nn.Linear(in_features, out_features, bias=False).to(device)
        lr = 0.1
        weight_decay = 0.1

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Store initial weight
        initial_weight = model.weight.data.clone()

        # Zero gradient: AdamW should still apply decoupled weight decay.
        optimizer.zero_grad(set_to_none=True)
        model.weight.grad = torch.zeros_like(model.weight)

        optimizer.step()

        # Weight should be smaller due to decay
        # w_new = w_old - lr * weight_decay * w_old
        expected = initial_weight * (1 - lr * weight_decay)

        assert_tensor_close(
            model.weight.data, expected,
            rtol=1e-4, atol=1e-6,
            msg="Weight decay not applied correctly"
        )

    def test_beta1_beta2(self, device):
        """Beta parameters control momentum vs variance tracking."""
        model = nn.Linear(10, 20).to(device)

        # High beta1, low beta2: more momentum, less variance adaptivity
        opt1 = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.9))
        model1 = model
        state1 = opt1.state

        x = torch.randn(5, 10, device=device)

        for _ in range(5):
            opt1.zero_grad()
            loss = model1(x).sum()
            loss.backward()
            opt1.step()

        # Check that state is being tracked
        assert state1[model1.weight]['step'] == 5

    def test_matches_reference(self, device):
        """Native AdamW produces same updates as torch.optim.AdamW."""
        torch.manual_seed(42)

        # Create two identical models
        model1 = nn.Linear(10, 20).to(device)
        model2 = nn.Linear(10, 20).to(device)

        # Copy weights
        model2.weight.data.copy_(model1.weight.data)
        model2.bias.data.copy_(model1.bias.data)

        # Create optimizers with same config
        lr = 1e-3
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.1

        opt_native = AdamW(model1.parameters(), lr=lr, betas=betas,
                          eps=eps, weight_decay=weight_decay)
        opt_ref = torch.optim.AdamW(model2.parameters(), lr=lr, betas=betas,
                               eps=eps, weight_decay=weight_decay)

        # Training loop
        x = torch.randn(5, 10, device=device)

        for step in range(5):
            opt_native.zero_grad()
            opt_ref.zero_grad()

            loss1 = model1(x).sum()
            loss2 = model2(x).sum()

            loss1.backward()
            loss2.backward()

            opt_native.step()
            opt_ref.step()

            # Parameters should match after each step
            assert_tensor_close(
                model1.weight.data, model2.weight.data,
                rtol=1e-5, atol=1e-7,
                msg=f"Weight mismatch at step {step}"
            )
            assert_tensor_close(
                model1.bias.data, model2.bias.data,
                rtol=1e-5, atol=1e-7,
                msg=f"Bias mismatch at step {step}"
            )

    def test_zero_grad(self, device):
        """zero_grad() clears gradients correctly."""
        model = nn.Linear(10, 20).to(device)
        optimizer = AdamW(model.parameters())

        # Create gradients
        x = torch.randn(5, 10, device=device)
        loss = model(x).sum()
        loss.backward()

        assert model.weight.grad is not None, "Gradients not created"

        # Zero gradients
        optimizer.zero_grad()

        assert model.weight.grad is None or torch.allclose(model.weight.grad,
                                                                 torch.zeros_like(model.weight.grad)), \
            "Gradients not cleared"

    def test_state_dict(self, device):
        """State dict save/load preserves optimizer state."""
        model = nn.Linear(10, 20).to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        # Train for a few steps
        x = torch.randn(5, 10, device=device)
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(x).sum()
            loss.backward()
            optimizer.step()

        # Save state
        state_dict = optimizer.state_dict()

        # Create new optimizer and load state
        new_optimizer = AdamW(model.parameters(), lr=1e-3)
        new_optimizer.load_state_dict(state_dict)

        # Check state is preserved
        old_state = optimizer.state[model.weight]
        new_state = new_optimizer.state[model.weight]

        assert old_state['step'] == new_state['step'], "Step not preserved"

        assert_tensor_close(
            old_state['exp_avg'], new_state['exp_avg'],
            msg="Momentum not preserved"
        )
        assert_tensor_close(
            old_state['exp_avg_sq'], new_state['exp_avg_sq'],
            msg="Variance not preserved"
        )
