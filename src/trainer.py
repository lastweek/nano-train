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

                    # Log detailed per-layer gradient statistics
                    self._log_detailed_gradient_stats(global_step)

                    # Log weight update ratios (learning rate diagnostic)
                    self._log_weight_update_ratios(global_step)

                    # Log gradient histograms (for visualization)
                    self._log_gradient_histograms(global_step)

                    # Log weight norms and changes (weight evolution tracking)
                    self._log_weight_norms(global_step)
                    self._log_weight_changes(global_step)
                    self._log_weight_histograms(global_step)

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

    def _log_detailed_gradient_stats(self, step):
        """
        Log detailed per-layer gradient statistics.

        Logs per-layer gradient norms, means, and stds to TensorBoard.
        This helps identify which layers are most/least active during training.

        Args:
            step: Current training step
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()

                # Log to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar(f'gradients/{name}/norm', grad_norm, step)
                    self.writer.add_scalar(f'gradients/{name}/mean', grad_mean, step)
                    self.writer.add_scalar(f'gradients/{name}/std', grad_std, step)

                # Log to console every 100 steps
                if step % 100 == 0:
                    logger.info(f"  {name}: grad_norm={grad_norm:.4f}, mean={grad_mean:.6f}, std={grad_std:.6f}")

    def _log_weight_update_ratios(self, step):
        """
        Log weight update ratios to detect learning rate issues.

        Update ratio = ||weight_change|| / ||weight|| where weight_change is the
        optimizer step (lr * gradient). This helps identify if the learning rate
        is too high (ratio > 0.1) or too low (ratio < 1e-7).

        Args:
            step: Current training step
        """
        # Get current learning rate
        lr = self.scheduler.get_lr()

        alerts = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Compute weight norm
                weight_norm = param.data.norm(2).item()

                # Compute update norm (approximate as lr * grad_norm)
                # This is the change the optimizer will apply
                grad_norm = param.grad.norm(2).item()
                update_norm = lr * grad_norm

                # Compute update ratio (relative change)
                if weight_norm > 0:
                    update_ratio = update_norm / weight_norm
                else:
                    update_ratio = 0.0

                # Log to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar(f'updates/{name}/ratio', update_ratio, step)
                    self.writer.add_scalar(f'updates/{name}/update_norm', update_norm, step)
                    self.writer.add_scalar(f'updates/{name}/weight_norm', weight_norm, step)

                # Check for issues
                if update_ratio > 0.1:
                    alerts.append((name, update_ratio, 'TOO_HIGH'))
                elif update_ratio < 1e-7 and weight_norm > 1e-6:
                    alerts.append((name, update_ratio, 'TOO_LOW'))

        # Log alerts to console
        if alerts and step % 100 == 0:
            logger.warning(f"Update ratio alerts (step {step}):")
            for name, ratio, issue in alerts:
                if issue == 'TOO_HIGH':
                    logger.warning(f"  {name}: {ratio:.4f} > 0.1 (learning rate may be too high)")
                else:
                    logger.warning(f"  {name}: {ratio:.8f} < 1e-7 (learning rate may be too low)")

    def _log_gradient_histograms(self, step):
        """
        Log gradient histograms to visualize gradient flow over time.

        Histograms show the full distribution of gradients, not just summary stats.
        This helps identify:
        - Gradient distribution changes over time
        - Multi-modal distributions (multiple peaks)
        - Outliers and heavy tails
        - Dead neurons (gradients concentrated near zero)

        Args:
            step: Current training step
        """
        if self.writer is None:
            return

        # Log histograms less frequently (expensive operation)
        histogram_freq = max(100, self.log_steps * 10)
        if step % histogram_freq != 0:
            return

        # Collect gradients for histogram logging
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Log gradient histogram
                self.writer.add_histogram(
                    f'gradients_hist/{name}',
                    param.grad.data,
                    step
                )

        # Log a summary of gradient distribution statistics
        if step % (histogram_freq * 10) == 0:
            grad_data = []
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_flat = param.grad.data.flatten().cpu()
                    grad_data.append({
                        'name': name,
                        'min': grad_flat.min().item(),
                        'max': grad_flat.max().item(),
                        'p25': grad_flat.quantile(0.25).item(),
                        'p50': grad_flat.quantile(0.50).item(),
                        'p75': grad_flat.quantile(0.75).item(),
                        'p95': grad_flat.quantile(0.95).item(),
                        'p99': grad_flat.quantile(0.99).item(),
                    })

            logger.info(f"Gradient distribution summary (step {step}):")
            for g in grad_data[:5]:  # Show first 5 layers
                logger.info(
                    f"  {g['name']}: "
                    f"min={g['min']:.6f}, p25={g['p25']:.6f}, "
                    f"median={g['p50']:.6f}, p75={g['p75']:.6f}, "
                    f"p95={g['p95']:.6f}, max={g['max']:.6f}"
                )
    def _log_weight_norms(self, step):
        """
        Log per-layer weight norms to track weight evolution during training.

        This helps identify:
        - Weights growing too fast (instability risk)
        - Weights not changing (LR too low or frozen layers)
        - Per-layer weight divergence
        - Weight initialization issues

        Args:
            step: Current training step
        """
        if self.writer is None:
            return

        # Log less frequently than gradients (weights change slower)
        weight_log_freq = max(100, self.log_steps * 10)
        if step % weight_log_freq != 0:
            return

        # Track weight norms for each layer
        layer_norms = {}
        for name, param in self.model.named_parameters():
            if param.data is not None:
                weight_norm = param.data.norm(2).item()
                layer_norms[name] = weight_norm

                # Log to TensorBoard
                self.writer.add_scalar(f'weights/{name}/norm', weight_norm, step)

        # Log summary every 1000 steps
        if step % (weight_log_freq * 10) == 0:
            # Find min/max/median weight norms
            norms_list = list(layer_norms.values())
            if norms_list:
                global_norm = math.sqrt(sum(n**2 for n in norms_list))
                min_norm = min(norms_list)
                max_norm = max(norms_list)
                median_norm = sorted(norms_list)[len(norms_list) // 2]

                # Check for anomalies
                anomalies = []
                if max_norm / min_norm > 1000:
                    anomalies.append(f"Weight norm variance > 1000× (min={min_norm:.4f}, max={max_norm:.4f})")

                # Log summary
                logger.info(f"Weight norm summary (step {step}):")
                logger.info(f"  Global norm: {global_norm:.4f}")
                logger.info(f"  Layer norms: min={min_norm:.4f}, median={median_norm:.4f}, max={max_norm:.4f}")
                logger.info(f"  Ratio (max/min): {max_norm/min_norm:.2f}×")

                if anomalies:
                    for anomaly in anomalies:
                        logger.warning(f"  ⚠ {anomaly}")

    def _log_weight_changes(self, step):
        """
        Track how much weights are changing over time.

        Computes the change in weight norms compared to the previous logging step.
        This helps detect:
        - Sudden weight changes (instability)
        - Weights stagnating (LR too low)
        - Training convergence

        Args:
            step: Current training step
        """
        if self.writer is None:
            return

        # Initialize previous norms if first call
        if not hasattr(self, '_prev_weight_norms'):
            self._prev_weight_norms = {}
            self._weight_log_step = 0
            return

        # Log less frequently
        weight_change_freq = max(500, self.log_steps * 50)
        if step % weight_change_freq != 0:
            return

        # Compute weight changes
        changes = []
        for name, param in self.model.named_parameters():
            if param.data is not None:
                current_norm = param.data.norm(2).item()

                # Compute change if we have previous measurement
                if name in self._prev_weight_norms:
                    prev_norm = self._prev_weight_norms[name]
                    if prev_norm > 0:
                        change_pct = ((current_norm - prev_norm) / prev_norm) * 100
                    else:
                        change_pct = 0.0

                    changes.append({
                        'name': name,
                        'prev_norm': prev_norm,
                        'current_norm': current_norm,
                        'change_pct': change_pct
                    })

                # Store current norm for next time
                self._prev_weight_norms[name] = current_norm

        # Log summary if we have changes
        if changes and step % (weight_change_freq * 10) == 0:
            changes_pct = [c['change_pct'] for c in changes]
            avg_change = sum(changes_pct) / len(changes_pct)

            # Find layers with largest changes
            changes.sort(key=lambda x: abs(x['change_pct']), reverse=True)

            logger.info(f"Weight change summary (step {step}):")
            logger.info(f"  Steps since last check: {step - self._weight_log_step}")
            logger.info(f"  Average change: {avg_change:.2f}%")

            # Show top 5 changed layers
            for c in changes[:5]:
                logger.info(
                    f"  {c['name']}: {c['prev_norm']:.4f} → {c['current_norm']:.4f} "
                    f"({c['change_pct']:+.2f}%)"
                )

            self._weight_log_step = step

    def _log_weight_histograms(self, step):
        """
        Log weight histograms to visualize weight distributions over time.

        This helps identify:
        - Weight initialization quality
        - Weight distribution shifts during training
        - Dead neurons (weights concentrated at zero)
        - Multi-modal distributions
        - Outliers and heavy tails

        Args:
            step: Current training step
        """
        if self.writer is None:
            return

        # Log histograms less frequently (expensive)
        histogram_freq = max(500, self.log_steps * 50)
        if step % histogram_freq != 0:
            return

        # Log weight histograms for each layer
        for name, param in self.model.named_parameters():
            if param.data is not None:
                self.writer.add_histogram(
                    f'weights_hist/{name}',
                    param.data,
                    step
                )

        # Log distribution summary every 5000 steps
        if step % (histogram_freq * 10) == 0:
            weight_data = []
            for name, param in self.model.named_parameters():
                if param.data is not None:
                    weight_flat = param.data.flatten().cpu()
                    weight_data.append({
                        'name': name,
                        'mean': weight_flat.mean().item(),
                        'std': weight_flat.std().item(),
                        'min': weight_flat.min().item(),
                        'max': weight_flat.max().item(),
                        'p01': weight_flat.quantile(0.01).item(),  # 1st percentile
                        'p99': weight_flat.quantile(0.99).item(),  # 99th percentile
                    })

            logger.info(f"Weight distribution summary (step {step}):")
            for w in weight_data[:5]:  # Show first 5 layers
                logger.info(
                    f"  {w['name']}: "
                    f"mean={w['mean']:.6f}, std={w['std']:.6f}, "
                    f"min={w['min']:.6f}, max={w['max']:.6f}, "
                    f"p99={w['p99']:.6f}"
                )
