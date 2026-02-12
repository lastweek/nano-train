"""
Learning rate scheduler for MVP.
"""

import math
import torch


class CosineAnnealingScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=0.0):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            max_steps: Total training steps
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self):
        """Get current learning rate."""
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def state_dict(self):
        """Return scheduler state."""
        return {
            'current_step': self.current_step,
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.current_step = state_dict['current_step']
