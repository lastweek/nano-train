"""
Training loop for MVP.
"""

import math
import os
import time
from datetime import timedelta

import torch
from tqdm import tqdm

# TensorBoard with graceful fallback
try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        try:
            from tensorboard import SummaryWriter
        except ImportError:
            SummaryWriter = None

from src.config import Config
from src.logging import get_logger
from src.losses import CrossEntropyLoss
from src.optimizer import create_optimizer
from src.scheduler import CosineAnnealingScheduler


# Get logger for this module
logger = get_logger(__name__)


class Trainer:
    """Simple trainer for MVP."""

    def __init__(self, model, config: Config, train_loader, device):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.device = device
        self.log_steps = max(1, config.training.log_steps)

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
        # NATIVE: Our implementation in src/losses.py
        # ORIGINAL: torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion = CrossEntropyLoss(ignore_index=-100)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # TensorBoard writer
        self.writer = None
        if SummaryWriter is not None:
            log_path = os.path.join(config.log_dir, config.run_name)
            self.writer = SummaryWriter(log_path)
            logger.info(f"TensorBoard logs: {log_path}")
            logger.info("View with: python3 scripts/view_logs.py or open scripts/view_logs.html")
        else:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.config.training.max_steps} steps...")
        logger.info(f"Model parameters: {self.model.num_parameters:,}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.config.training.batch_size}")
        logger.info(f"Learning rate: {self.config.training.learning_rate}")

        self.model.train()
        global_step = 0
        losses = []
        total_tokens = 0
        total_samples = 0
        window_tokens = 0
        window_samples = 0
        window_steps = 0
        window_start_time = time.time()

        # Training loop
        start_time = time.time()

        while global_step < self.config.training.max_steps:
            epoch_progress = tqdm(self.train_loader, desc=f"Step {global_step}")

            for batch in epoch_progress:
                if global_step >= self.config.training.max_steps:
                    break

                # Log on a fixed interval to keep overhead predictable.
                log_this_step = global_step % self.log_steps == 0
                step_start = time.time()

                # Training step returns loss plus lightweight stats for logging.
                loss, grad_norm, tokens, samples = self.training_step(
                    batch,
                    compute_grad_norm=log_this_step
                )
                step_time = time.time() - step_start

                # Track totals for throughput metrics.
                losses.append(loss.detach().cpu().item())
                total_tokens += tokens
                total_samples += samples
                window_tokens += tokens
                window_samples += samples
                window_steps += 1

                # Logging
                if log_this_step:
                    avg_loss = sum(losses[-self.log_steps:]) / min(self.log_steps, len(losses))
                    lr = self.scheduler.get_lr()
                    epoch_progress.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}'
                    })

                    # TensorBoard logging (optional dependency).
                    if self.writer is not None:
                        self.writer.add_scalar('Loss/train', avg_loss, global_step)
                        self.writer.add_scalar('LR', lr, global_step)
                        self.writer.add_scalar('Time/step_seconds', step_time, global_step)

                    window_time = time.time() - window_start_time
                    if window_time > 0:
                        steps_per_second = window_steps / window_time
                        tokens_per_second = window_tokens / window_time
                        samples_per_second = window_samples / window_time
                        if self.writer is not None:
                            self.writer.add_scalar(
                                'Throughput/steps_per_second',
                                steps_per_second,
                                global_step
                            )
                            self.writer.add_scalar(
                                'Throughput/tokens_per_second',
                                tokens_per_second,
                                global_step
                            )
                            self.writer.add_scalar(
                                'Throughput/samples_per_second',
                                samples_per_second,
                                global_step
                            )
                    if grad_norm is not None:
                        if self.writer is not None:
                            self.writer.add_scalar('Gradients/norm', grad_norm, global_step)
                    if self.writer is not None:
                        self.writer.add_scalar(
                            'Parameters/norm',
                            self._get_param_norm(),
                            global_step
                        )
                    if self.device.type == 'cuda':
                        allocated_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                        reserved_mb = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
                        max_allocated_mb = (
                            torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                        )
                        max_reserved_mb = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
                        if self.writer is not None:
                            self.writer.add_scalar('Memory/allocated_mb', allocated_mb, global_step)
                            self.writer.add_scalar('Memory/reserved_mb', reserved_mb, global_step)
                            self.writer.add_scalar(
                                'Memory/max_allocated_mb',
                                max_allocated_mb,
                                global_step
                            )
                            self.writer.add_scalar(
                                'Memory/max_reserved_mb',
                                max_reserved_mb,
                                global_step
                            )

                    window_tokens = 0
                    window_samples = 0
                    window_steps = 0
                    window_start_time = time.time()

                    # Save checkpoint on schedule to avoid losing progress.
                    if global_step % self.config.training.save_steps == 0 and global_step > 0:
                        self.save_checkpoint(global_step)

                global_step += 1

        # Final checkpoint to capture end-of-run state.
        self.save_checkpoint(global_step, final=True)

        # Print training summary
        elapsed_time = time.time() - start_time
        steps_per_second = global_step / elapsed_time
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0.0
        samples_per_second = total_samples / elapsed_time if elapsed_time > 0 else 0.0
        logger.info(f"Training completed!")
        logger.info(f"Total time: {timedelta(seconds=int(elapsed_time))}")
        logger.info(f"Steps per second: {steps_per_second:.2f}")
        logger.info(f"Tokens per second: {tokens_per_second:.2f}")
        logger.info(f"Samples per second: {samples_per_second:.2f}")
        logger.info(f"Final loss: {losses[-1]:.4f}")

        # Log final metrics to TensorBoard for run comparison.
        if self.writer is not None:
            self.writer.add_scalar('Throughput/steps_per_second', steps_per_second, global_step)
            self.writer.add_scalar('Throughput/tokens_per_second', tokens_per_second, global_step)
            self.writer.add_scalar('Throughput/samples_per_second', samples_per_second, global_step)
            self.writer.add_hparams(
                {'learning_rate': str(self.config.training.learning_rate),
                     'batch_size': str(self.config.training.batch_size),
                     'max_steps': str(self.config.training.max_steps)},
                {'final_loss': losses[-1],
                 'steps_per_second': steps_per_second}
            )
            # Close writer
            self.writer.close()

    def training_step(self, batch, compute_grad_norm=False):
        """Single training step."""
        # Move to device; input_ids/labels are (B, S-1).
        # SHAPE: (batch_size=2, seq_len=256)
        input_ids = batch['input_ids'].to(self.device)  # (2, 256)
        labels = batch['labels'].to(self.device)        # (2, 255) - shifted by 1
        tokens = input_ids.numel()  # 2*256 = 512 total tokens
        samples = input_ids.size(0)  # batch_size = 2

        # Forward pass with optional bf16 autocast.
        # OUTPUT SHAPE: (batch, seq_len-1, vocab_size)
        if self.config.training.bf16 and torch.cuda.is_bf16_supported():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = self.model(input_ids)  # (2, 255, vocab)
                loss = self._compute_loss(logits, labels)  # scalar
        else:
            logits = self.model(input_ids)
            loss = self._compute_loss(logits, labels)

        # Backward pass computes gradients for all parameters.
        loss.backward()

        # Gradient clipping stabilizes training for large models.
        # NATIVE: Our implementation in src/layers.py
        # ORIGINAL: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        grad_norm = None
        if self.config.training.clip_grad > 0:
            from src.layers import clip_grad_norm
            grad_norm = clip_grad_norm(
                self.model.parameters(),
                self.config.training.clip_grad
            )
        elif compute_grad_norm:
            grad_norm = self._get_grad_norm()

        # Optimizer + scheduler step updates weights and learning rate.
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss, grad_norm, tokens, samples

    def _compute_loss(self, logits, labels):
        """
        Compute language modeling loss.

        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len)
        """
        # Shift logits and labels for next-token prediction.
        # INPUT:    logits(B, S, V),     labels(B, S)
        # OUTPUT:   shift_logits(B, S-1, V), shift_labels(B, S-1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten to (B * (S-1), vocab) for the loss.
        # OUTPUT:   (B*(S-1), V)
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        loss = self.criterion(shift_logits, shift_labels)  # scalar
        return loss

    def save_checkpoint(self, step, final=False):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(
            self.config.output_dir,
            f"checkpoint-step-{step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        # model.state_dict() returns OrderedDict with all parameters
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

        logger.info(f"Checkpoint saved at step {step} to {checkpoint_dir}")

    def _get_grad_norm(self):
        """Compute global grad norm for logging."""
        # Each parameter's grad is same shape as the parameter itself
        total_norm_sq = 0.0
        for param in self.model.parameters():
            if param.grad is None:
                continue
            # SHAPE: param.grad e.g., weight (hidden, out_features)
            param_norm = param.grad.data.norm(2)  # scalar L2 norm
            total_norm_sq += param_norm.item() ** 2
        # OUTPUT: sqrt(sum of all squared norms)
        return math.sqrt(total_norm_sq)

    def _get_param_norm(self):
        """Compute global parameter norm for logging."""
        # Each parameter is same shape as its gradient
        total_norm_sq = 0.0
        for param in self.model.parameters():
            # SHAPE: param.data e.g., weight (hidden, out_features)
            param_norm = param.data.norm(2)  # scalar L2 norm
            total_norm_sq += param_norm.item() ** 2
        # OUTPUT: sqrt(sum of all squared norms) - for logging
        return math.sqrt(total_norm_sq)
