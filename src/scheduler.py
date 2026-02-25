"""
Learning rate scheduler for MVP.
"""

import math
from typing import Any

from torch.optim import Optimizer


class CosineAnnealingScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
    ) -> None:
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            max_steps: Total training steps
            min_lr: Minimum learning rate
        """
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if warmup_steps >= max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")
        if min_lr < 0:
            raise ValueError("min_lr must be >= 0")

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        # Capture the starting LR as the schedule's maximum (base) LR.
        # We keep this stable even if the optimizer LR is later updated.
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.current_step = 0

    def step(self) -> float:
        """Update learning rate."""
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        base_lr = self.base_lrs[0]
        if self.warmup_steps > 0 and self.current_step <= self.warmup_steps:
            # Linear warmup
            return base_lr * self.current_step / self.warmup_steps

        # Cosine annealing
        progress = (self.current_step - self.warmup_steps) / (
            self.max_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        return self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state."""
        return {
            "current_step": self.current_step,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load scheduler state."""
        self.current_step = int(state_dict["current_step"])
        if "base_lrs" in state_dict:
            self.base_lrs = [float(x) for x in state_dict["base_lrs"]]
        # Keep optimizer state consistent with restored schedule position.
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
