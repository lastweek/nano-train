"""
Unit tests for CosineAnnealingScheduler.

Tests verify:
1. Warmup phase increases LR linearly
2. Annealing phase follows cosine curve
3. LR is correctly applied to optimizer
4. State save/load works
"""

import pytest
import torch
import math

from src.scheduler import CosineAnnealingScheduler
from src.optimizer import AdamW


class TestCosineAnnealing:
    """Test cases for CosineAnnealingScheduler."""

    def test_warmup_phase(self, device):
        """Learning rate increases linearly during warmup."""
        model = torch.nn.Linear(10, 20).to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        warmup_steps = 10
        max_steps = 100
        min_lr = 0.0

        scheduler = CosineAnnealingScheduler(
            optimizer, warmup_steps, max_steps, min_lr
        )

        # Check warmup phase
        for step in range(1, warmup_steps + 1):
            lr = scheduler.step()
            expected_lr = 1e-3 * step / warmup_steps

            assert abs(lr - expected_lr) < 1e-8, \
                f"Warmup LR mismatch at step {step}: {lr} vs {expected_lr}"

    def test_annealing_phase(self, device):
        """Learning rate follows cosine curve after warmup."""
        model = torch.nn.Linear(10, 20).to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        warmup_steps = 10
        max_steps = 100
        min_lr = 0.0

        scheduler = CosineAnnealingScheduler(
            optimizer, warmup_steps, max_steps, min_lr
        )

        # Skip to annealing phase
        for _ in range(warmup_steps):
            scheduler.step()

        # Check annealing phase
        base_lr = 1e-3
        for step in range(warmup_steps + 1, max_steps + 1):
            lr = scheduler.step()

            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            expected_lr = min_lr + (base_lr - min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

            assert abs(lr - expected_lr) < 1e-6, \
                f"Annealing LR mismatch at step {step}: {lr} vs {expected_lr}"

    def test_optimizer_lr_updated(self, device):
        """Optimizer parameter groups have correct LR."""
        model = torch.nn.Linear(10, 20).to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        scheduler = CosineAnnealingScheduler(
            optimizer, warmup_steps=5, max_steps=100, min_lr=0.0
        )

        # Take a few steps
        for _ in range(10):
            scheduler.step()

        # Check optimizer LR matches scheduler
        scheduler_lr = scheduler.get_lr()
        optimizer_lr = optimizer.param_groups[0]['lr']

        assert abs(scheduler_lr - optimizer_lr) < 1e-8, \
            f"Optimizer LR not updated: {optimizer_lr} vs {scheduler_lr}"

    def test_min_lr_reached(self, device):
        """LR reaches min_lr at max_steps."""
        model = torch.nn.Linear(10, 20).to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        warmup_steps = 10
        max_steps = 100
        min_lr = 0.0

        scheduler = CosineAnnealingScheduler(
            optimizer, warmup_steps, max_steps, min_lr
        )

        # Train to max_steps
        for _ in range(max_steps):
            scheduler.step()

        lr = scheduler.get_lr()
        assert abs(lr - min_lr) < 1e-6, \
            f"LR should be min_lr at end: {lr} vs {min_lr}"

    def test_state_dict(self, device):
        """State dict save/load preserves step count."""
        model = torch.nn.Linear(10, 20).to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        scheduler = CosineAnnealingScheduler(
            optimizer, warmup_steps=5, max_steps=100, min_lr=0.0
        )

        # Take some steps
        for _ in range(10):
            scheduler.step()

        # Save state
        state = scheduler.state_dict()
        assert state['current_step'] == 10, "Step count incorrect in state_dict"

        # Create new scheduler and load state
        new_scheduler = CosineAnnealingScheduler(
            optimizer, warmup_steps=5, max_steps=100, min_lr=0.0
        )
        new_scheduler.load_state_dict(state)

        assert new_scheduler.current_step == 10, "Step not preserved"

        # LR should match
        lr_old = scheduler.get_lr()
        lr_new = new_scheduler.get_lr()

        assert abs(lr_old - lr_new) < 1e-8, "LR not preserved after load"

    def test_get_lr_no_step(self, device):
        """get_lr() returns current LR without advancing."""
        model = torch.nn.Linear(10, 20).to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        scheduler = CosineAnnealingScheduler(
            optimizer, warmup_steps=10, max_steps=100, min_lr=0.0
        )

        # Get LR multiple times without step()
        lr1 = scheduler.get_lr()
        lr2 = scheduler.get_lr()
        lr3 = scheduler.get_lr()

        # All should be same (step hasn't advanced)
        assert lr1 == lr2 == lr3, "get_lr() changed state"

    def test_single_step_warmup(self, device):
        """Single step warmup (warmup_steps=1)."""
        model = torch.nn.Linear(10, 20).to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        warmup_steps = 1
        scheduler = CosineAnnealingScheduler(
            optimizer, warmup_steps=warmup_steps, max_steps=100, min_lr=0.0
        )

        # First step should complete warmup
        lr = scheduler.step()
        assert lr == 1e-3, f"Warmup should reach base_lr: {lr}"

        # Second step should start annealing
        lr2 = scheduler.step()
        assert lr2 < 1e-3, "Should start annealing"
