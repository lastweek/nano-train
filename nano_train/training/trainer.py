"""
Training loop for MVP.
"""

import os
import time
from datetime import timedelta

import torch
import torch.nn as nn
from tqdm import tqdm

from nano_train.core.config import Config
from nano_train.training.optimizer import create_optimizer
from nano_train.training.scheduler import CosineAnnealingScheduler


class Trainer:
    """Simple trainer for MVP."""

    def __init__(self, model, config: Config, train_loader, device):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.device = device

        # Optimizer
        self.optimizer = create_optimizer(
            model,
            config.training.learning_rate,
            config.training.weight_decay
        )

        # Scheduler
        self.scheduler = CosineAnnealingScheduler(
            self.optimizer,
            config.training.warmup_steps,
            config.training.max_steps,
            min_lr=0.0
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.training.max_steps} steps...")
        print(f"Model parameters: {self.model.num_parameters:,}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.training.batch_size}")
        print(f"Learning rate: {self.config.training.learning_rate}")

        self.model.train()
        global_step = 0
        losses = []

        # Training loop
        start_time = time.time()

        while global_step < self.config.training.max_steps:
            epoch_progress = tqdm(self.train_loader, desc=f"Step {global_step}")

            for batch in epoch_progress:
                if global_step >= self.config.training.max_steps:
                    break

                # Training step
                loss = self.training_step(batch)

                # Track loss
                losses.append(loss.detach().cpu().item())

                # Logging
                if global_step % 10 == 0:
                    avg_loss = sum(losses[-10:]) / min(10, len(losses))
                    lr = self.scheduler.get_lr()
                    epoch_progress.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}'
                    })

                # Save checkpoint
                if global_step % self.config.training.save_steps == 0 and global_step > 0:
                    self.save_checkpoint(global_step)

                global_step += 1

        # Final checkpoint
        self.save_checkpoint(global_step, final=True)

        # Print training summary
        elapsed_time = time.time() - start_time
        steps_per_second = global_step / elapsed_time
        print(f"\nTraining completed!")
        print(f"Total time: {timedelta(seconds=int(elapsed_time))}")
        print(f"Steps per second: {steps_per_second:.2f}")
        print(f"Final loss: {losses[-1]:.4f}")

    def training_step(self, batch):
        """Single training step."""
        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        if self.config.training.bf16 and torch.cuda.is_bf16_supported():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = self.model(input_ids)
                loss = self._compute_loss(logits, labels)
        else:
            logits = self.model(input_ids)
            loss = self._compute_loss(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.config.training.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.clip_grad
            )

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss

    def _compute_loss(self, logits, labels):
        """
        Compute language modeling loss.

        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len)
        """
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        loss = self.criterion(shift_logits, shift_labels)
        return loss

    def save_checkpoint(self, step, final=False):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(
            self.config.output_dir,
            f"checkpoint-step-{step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(checkpoint_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)

        # Save optimizer and scheduler
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_path)

        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        torch.save(self.scheduler.state_dict(), scheduler_path)

        # Save config
        import json
        from dataclasses import asdict
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        print(f"\nCheckpoint saved at step {step} to {checkpoint_dir}")
