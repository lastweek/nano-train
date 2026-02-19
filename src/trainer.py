"""
Training loop for MVP.
"""

from __future__ import annotations

import math
import os
import time
from datetime import timedelta
from typing import Optional

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
from src.monitoring import (
    block_main_weight_names as monitoring_block_main_weight_names,
    candidate_sentinel_param_names,
    clip_coef as compute_clip_coef,
    global_update_ratio as compute_global_update_ratio,
    resolve_sentinel_blocks as compute_sentinel_blocks,
    update_ratio as compute_update_ratio,
)
from src.optimizer import create_optimizer
from src.scheduler import CosineAnnealingScheduler


logger = get_logger(__name__)


class Trainer:
    """Simple trainer for MVP."""

    def __init__(self, model, config: Config, train_loader, device, val_loader=None):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.log_steps = max(1, int(config.training.log_steps))
        self.histogram_steps = max(1, int(config.monitoring.histogram_steps))
        self.monitoring_mode = config.monitoring.mode
        if self.monitoring_mode not in {"minimal", "standard", "debug"}:
            logger.warning(
                "Unknown monitoring mode %r; falling back to 'standard'.",
                self.monitoring_mode,
            )
            self.monitoring_mode = "standard"

        self._loss_ema: Optional[float] = None

        # Optimizer
        self.optimizer = create_optimizer(
            model,
            config.training.learning_rate,
            config.training.weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingScheduler(
            self.optimizer,
            config.training.warmup_steps,
            config.training.max_steps,
            min_lr=0.0,
        )

        # Loss function
        # NATIVE: Our implementation in src/losses.py
        # ORIGINAL: torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion = CrossEntropyLoss(ignore_index=-100)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Monitoring: pre-compute bounded parameter sets for Standard mode.
        self._named_params = dict(self.model.named_parameters())
        self._sentinel_blocks = self._resolve_sentinel_blocks()
        self._block_main_params_by_block = self._build_block_main_params_by_block()
        self._sentinel_param_names = self._build_sentinel_param_names()

        # Activation monitoring hooks (LayerNorm output sentinels).
        self._activation_log_step: Optional[int] = None
        self._activation_log_enabled: bool = False
        self._activation_hook_handles = []

        # CUDA memory totals (for reserved fraction).
        self._cuda_total_mem_bytes: Optional[int] = None
        if self.device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(self.device)
                self._cuda_total_mem_bytes = int(props.total_memory)
            except Exception:
                self._cuda_total_mem_bytes = None

        # TensorBoard writer
        self.writer = None
        if SummaryWriter is not None:
            log_path = os.path.join(config.log_dir, config.run_name)
            self.writer = SummaryWriter(log_path)
            logger.info("TensorBoard logs: %s", log_path)
            logger.info("View with: python3 scripts/view_logs.py or open scripts/view_logs.html")
        else:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")

        if self.writer is not None and self._cuda_total_mem_bytes is not None:
            self.writer.add_scalar("Memory/total_mb", self._cuda_total_mem_bytes / (1024**2), 0)

        self._register_activation_hooks()

    def _resolve_sentinel_blocks(self) -> tuple[int, ...]:
        """Resolve sentinel block indices for bounded monitoring."""
        return compute_sentinel_blocks(
            int(self.config.model.num_layers),
            requested=self.config.monitoring.sentinel_blocks,
        )

    @staticmethod
    def _block_main_weight_names(block_idx: int) -> list[str]:
        """Return the "main" weight tensors for a transformer block."""
        return monitoring_block_main_weight_names(block_idx)

    def _build_block_main_params_by_block(self) -> list[list[tuple[str, torch.nn.Parameter]]]:
        """Precompute block-main parameters for bounded monitoring."""
        num_layers = int(self.config.model.num_layers)
        params_by_block: list[list[tuple[str, torch.nn.Parameter]]] = []
        for block_idx in range(num_layers):
            entries: list[tuple[str, torch.nn.Parameter]] = []
            for name in self._block_main_weight_names(block_idx):
                param = self._named_params.get(name)
                if param is None:
                    continue
                entries.append((name, param))
            params_by_block.append(entries)
        return params_by_block

    def _build_sentinel_param_names(self) -> set[str]:
        """Build the per-parameter sentinel set for Standard monitoring."""
        names = candidate_sentinel_param_names(sentinel_blocks=self._sentinel_blocks)
        return {name for name in names if name in self._named_params}

    def _register_activation_hooks(self) -> None:
        """Register forward hooks for activation sentinels (LayerNorm outputs)."""
        if self.monitoring_mode == "minimal":
            return
        if self.config.monitoring.activation_sites == "none":
            return

        sites: list[str] = []
        if self.config.monitoring.activation_sites == "sentinel":
            sites.extend([f"blocks.{i}.attn_norm" for i in self._sentinel_blocks])
            sites.append("ln_f")
        elif self.config.monitoring.activation_sites == "all_lns":
            try:
                from src.layers import LayerNorm
            except Exception:
                LayerNorm = None
            if LayerNorm is None:
                logger.warning("Could not import src.layers.LayerNorm; skipping activation hooks.")
                return
            for name, module in self.model.named_modules():
                if isinstance(module, LayerNorm):
                    sites.append(name)
        else:
            logger.warning(
                "Unknown activation_sites=%r; expected none|sentinel|all_lns.",
                self.config.monitoring.activation_sites,
            )
            return

        for site in sites:
            try:
                module = self.model.get_submodule(site)
            except Exception:
                module = dict(self.model.named_modules()).get(site)
            if module is None:
                logger.warning("Activation site not found: %s", site)
                continue
            handle = module.register_forward_hook(self._make_activation_hook(site))
            self._activation_hook_handles.append(handle)

    def _make_activation_hook(self, site: str):
        """Create a forward hook that logs activation scale metrics for `site`."""
        alerts = self.config.alerts

        def _hook(_module, _inputs, output) -> None:
            if not self._activation_log_enabled:
                return
            if self.writer is None:
                return
            step = self._activation_log_step
            if step is None:
                return
            if not torch.is_tensor(output):
                return

            with torch.no_grad():
                out = output.float()
                rms = torch.sqrt(torch.mean(out * out)).item()
                max_abs = torch.max(torch.abs(out)).item()

            self.writer.add_scalar(f"Activations/{site}/rms", rms, step)
            self.writer.add_scalar(f"Activations/{site}/max_abs", max_abs, step)

            if step % 100 != 0:
                return

            if rms < alerts.activation_rms_warn_low or rms > alerts.activation_rms_warn_high:
                logger.warning("Activation RMS warning (%s, step %d): rms=%.4f", site, step, rms)
            if max_abs > alerts.activation_max_abs_warn:
                logger.warning(
                    "Activation max-abs warning (%s, step %d): max_abs=%.4f",
                    site,
                    step,
                    max_abs,
                )

        return _hook

    @staticmethod
    def _quantile(values: list[float], q: float) -> float:
        """Compute a simple empirical quantile for a non-empty list."""
        if not values:
            return 0.0
        if q <= 0:
            return min(values)
        if q >= 1:
            return max(values)
        xs = sorted(values)
        idx = int(round(q * (len(xs) - 1)))
        idx = max(0, min(len(xs) - 1, idx))
        return xs[idx]

    @staticmethod
    def _l2_norms(tensors: list[torch.Tensor]) -> list[float]:
        """Compute L2 norms for a tensor list with foreach fast-path when available."""
        if not tensors:
            return []
        if hasattr(torch, "_foreach_norm"):
            norms = torch._foreach_norm(tensors, 2)
            return [float(n.item()) for n in norms]
        return [float(t.norm(2).item()) for t in tensors]

    @staticmethod
    def _is_finite_scalar(x: float) -> bool:
        return math.isfinite(float(x))

    def train(self):
        """Main training loop."""
        logger.info("Starting training for %d steps...", self.config.training.max_steps)
        logger.info("Model parameters: %s", f"{self.model.num_parameters:,}")
        logger.info("Device: %s", self.device)
        logger.info("Batch size: %d", self.config.training.batch_size)
        logger.info("Learning rate: %g", self.config.training.learning_rate)

        self.model.train()
        global_step = 0
        losses: list[float] = []
        total_tokens = 0
        total_samples = 0
        window_tokens = 0
        window_samples = 0
        window_steps = 0
        window_start_time = time.time()
        prev_step_end_time = time.time()

        start_time = time.time()

        while global_step < self.config.training.max_steps:
            epoch_progress = tqdm(self.train_loader, desc=f"Step {global_step}")

            for batch in epoch_progress:
                if global_step >= self.config.training.max_steps:
                    break

                log_this_step = global_step % self.log_steps == 0
                hist_this_step = global_step > 0 and (global_step % self.histogram_steps == 0)

                batch_ready_time = time.time()
                data_wait_seconds = max(0.0, batch_ready_time - prev_step_end_time)

                if self.device.type == "cuda" and self.config.monitoring.sync_cuda_timing:
                    torch.cuda.synchronize(self.device)
                step_start = time.time()

                lr_used = float(self.optimizer.param_groups[0]["lr"])

                loss, grad_norm, tokens, samples = self.training_step(
                    batch,
                    step=global_step,
                    log_this_step=log_this_step,
                )

                if self.device.type == "cuda" and self.config.monitoring.sync_cuda_timing:
                    torch.cuda.synchronize(self.device)
                step_seconds = time.time() - step_start

                loss_value = float(loss.detach().cpu().item())
                losses.append(loss_value)
                total_tokens += tokens
                total_samples += samples
                window_tokens += tokens
                window_samples += samples
                window_steps += 1

                prev_loss_ema = self._loss_ema if self._loss_ema is not None else loss_value
                beta = float(self.config.monitoring.loss_ema_beta)
                self._loss_ema = (beta * float(prev_loss_ema)) + ((1.0 - beta) * loss_value)
                loss_spike_ratio = loss_value / max(1e-12, float(prev_loss_ema))

                if log_this_step:
                    epoch_progress.set_postfix(
                        {"loss_ema": f"{self._loss_ema:.4f}", "lr": f"{lr_used:.2e}"}
                    )

                    if self.writer is not None:
                        self.writer.add_scalar("Loss/train", float(self._loss_ema), global_step)
                        self.writer.add_scalar(
                            "Health/loss_spike_ratio",
                            float(loss_spike_ratio),
                            global_step,
                        )
                        self.writer.add_scalar("LR", lr_used, global_step)
                        self.writer.add_scalar("Time/step_seconds", step_seconds, global_step)
                        self.writer.add_scalar(
                            "Time/data_wait_seconds",
                            float(data_wait_seconds),
                            global_step,
                        )
                        data_wait_frac = (
                            float(data_wait_seconds) / float(step_seconds) if step_seconds > 0 else 0.0
                        )
                        self.writer.add_scalar("Time/data_wait_frac", data_wait_frac, global_step)

                    if global_step % 100 == 0:
                        if loss_spike_ratio > self.config.alerts.loss_spike_stop:
                            logger.warning(
                                "Loss spike ratio BAD (step %d): %.3f > %.3f",
                                global_step,
                                float(loss_spike_ratio),
                                self.config.alerts.loss_spike_stop,
                            )
                        elif loss_spike_ratio > self.config.alerts.loss_spike_warn:
                            logger.warning(
                                "Loss spike ratio warning (step %d): %.3f > %.3f",
                                global_step,
                                float(loss_spike_ratio),
                                self.config.alerts.loss_spike_warn,
                            )

                        data_wait_frac = (
                            float(data_wait_seconds) / float(step_seconds) if step_seconds > 0 else 0.0
                        )
                        if data_wait_frac > self.config.alerts.data_wait_frac_bad:
                            logger.warning(
                                "Data wait frac BAD (step %d): %.3f > %.3f",
                                global_step,
                                data_wait_frac,
                                self.config.alerts.data_wait_frac_bad,
                            )
                        elif data_wait_frac > self.config.alerts.data_wait_frac_warn:
                            logger.warning(
                                "Data wait frac warning (step %d): %.3f > %.3f",
                                global_step,
                                data_wait_frac,
                                self.config.alerts.data_wait_frac_warn,
                            )

                    window_time = time.time() - window_start_time
                    if window_time > 0:
                        steps_per_second = window_steps / window_time
                        tokens_per_second = window_tokens / window_time
                        samples_per_second = window_samples / window_time
                        if self.writer is not None:
                            self.writer.add_scalar(
                                "Throughput/steps_per_second",
                                steps_per_second,
                                global_step,
                            )
                            self.writer.add_scalar(
                                "Throughput/tokens_per_second",
                                tokens_per_second,
                                global_step,
                            )
                            self.writer.add_scalar(
                                "Throughput/samples_per_second",
                                samples_per_second,
                                global_step,
                            )

                    # Post-step weight evolution tracking (bounded by monitoring mode).
                    self._log_weight_norms(global_step)

                    if self.device.type == "cuda":
                        allocated_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
                        reserved_mb = torch.cuda.memory_reserved(self.device) / (1024**2)
                        max_allocated_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
                        max_reserved_mb = torch.cuda.max_memory_reserved(self.device) / (1024**2)
                        if self.writer is not None:
                            self.writer.add_scalar("Memory/allocated_mb", allocated_mb, global_step)
                            self.writer.add_scalar("Memory/reserved_mb", reserved_mb, global_step)
                            self.writer.add_scalar("Memory/max_allocated_mb", max_allocated_mb, global_step)
                            self.writer.add_scalar("Memory/max_reserved_mb", max_reserved_mb, global_step)
                            if self._cuda_total_mem_bytes is not None:
                                reserved_bytes = torch.cuda.memory_reserved(self.device)
                                reserved_frac = reserved_bytes / self._cuda_total_mem_bytes
                                self.writer.add_scalar(
                                    "Memory/reserved_frac",
                                    float(reserved_frac),
                                    global_step,
                                )
                        if (
                            global_step % 100 == 0
                            and self._cuda_total_mem_bytes is not None
                            and self._cuda_total_mem_bytes > 0
                        ):
                            reserved_bytes = torch.cuda.memory_reserved(self.device)
                            reserved_frac = reserved_bytes / self._cuda_total_mem_bytes
                            if reserved_frac > self.config.alerts.reserved_frac_bad:
                                logger.warning(
                                    "Reserved frac BAD (step %d): %.3f > %.3f",
                                    global_step,
                                    float(reserved_frac),
                                    self.config.alerts.reserved_frac_bad,
                                )
                            elif reserved_frac > self.config.alerts.reserved_frac_warn:
                                logger.warning(
                                    "Reserved frac warning (step %d): %.3f > %.3f",
                                    global_step,
                                    float(reserved_frac),
                                    self.config.alerts.reserved_frac_warn,
                                )

                    window_tokens = 0
                    window_samples = 0
                    window_steps = 0
                    window_start_time = time.time()

                    if global_step % self.config.training.save_steps == 0 and global_step > 0:
                        ckpt_start = time.time()
                        self.save_checkpoint(global_step)
                        ckpt_seconds = time.time() - ckpt_start
                        if self.writer is not None:
                            self.writer.add_scalar(
                                "Time/checkpoint_seconds",
                                float(ckpt_seconds),
                                global_step,
                            )

                if (
                    self.val_loader is not None
                    and self.config.training.eval_steps > 0
                    and global_step > 0
                    and (global_step % self.config.training.eval_steps) == 0
                ):
                    self.evaluate(self.val_loader, step=global_step)

                if hist_this_step:
                    self._log_weight_histograms(global_step)

                prev_step_end_time = time.time()
                global_step += 1

        self.save_checkpoint(global_step, final=True)

        elapsed_time = time.time() - start_time
        steps_per_second = global_step / elapsed_time if elapsed_time > 0 else 0.0
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0.0
        samples_per_second = total_samples / elapsed_time if elapsed_time > 0 else 0.0
        final_loss = losses[-1] if losses else float("nan")

        logger.info("Training completed!")
        logger.info("Total time: %s", timedelta(seconds=int(elapsed_time)))
        logger.info("Steps per second: %.2f", steps_per_second)
        logger.info("Tokens per second: %.2f", tokens_per_second)
        logger.info("Samples per second: %.2f", samples_per_second)
        logger.info("Final loss: %.4f", final_loss)

        if self.writer is not None:
            self.writer.add_scalar("Throughput/steps_per_second", steps_per_second, global_step)
            self.writer.add_scalar("Throughput/tokens_per_second", tokens_per_second, global_step)
            self.writer.add_scalar("Throughput/samples_per_second", samples_per_second, global_step)
            self.writer.add_hparams(
                {
                    "learning_rate": str(self.config.training.learning_rate),
                    "batch_size": str(self.config.training.batch_size),
                    "max_steps": str(self.config.training.max_steps),
                },
                {"final_loss": final_loss, "steps_per_second": steps_per_second},
            )
            self.writer.close()

    def training_step(self, batch, step: int, log_this_step: bool):
        """
        Run a single training step.

        Args:
            batch: A batch dict from the dataloader.
            step: Current global step (for logging).
            log_this_step: Whether to emit detailed monitoring this step.
        """
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        tokens = input_ids.numel()
        samples = input_ids.size(0)

        hist_this_step = step > 0 and (step % self.histogram_steps == 0)

        # Activation hooks run during forward; enable only for intended monitoring steps.
        self._activation_log_step = step
        self._activation_log_enabled = (
            log_this_step
            and self.monitoring_mode != "minimal"
            and self.config.monitoring.activation_sites != "none"
        )

        if self.config.training.bf16 and torch.cuda.is_bf16_supported():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = self.model(input_ids)
                loss = self._compute_loss(logits, labels)
        else:
            logits = self.model(input_ids)
            loss = self._compute_loss(logits, labels)

        if self.config.monitoring.fail_fast_nonfinite and not torch.isfinite(loss):
            if self.writer is not None:
                self.writer.add_scalar("Health/non_finite_loss", 1.0, step)
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.detach().item()}")

        loss.backward()

        grad_norm: Optional[float] = None
        clip_coef = 1.0
        clipped = False
        if self.config.training.clip_grad > 0:
            from src.layers import clip_grad_norm

            grad_norm = clip_grad_norm(self.model.parameters(), self.config.training.clip_grad)
            clip_coef = compute_clip_coef(
                float(grad_norm),
                float(self.config.training.clip_grad),
                eps=1e-6,
            )
            clipped = float(grad_norm) > float(self.config.training.clip_grad)
        elif log_this_step or hist_this_step:
            grad_norm = self._get_grad_norm()

        if log_this_step:
            labels_used = labels[..., 1:]
            ignore_frac = float((labels_used == -100).sum().item()) / max(1, labels_used.numel())

            if self.writer is not None:
                self.writer.add_scalar("Data/ignore_frac", ignore_frac, step)
                self.writer.add_scalar("Health/non_finite_loss", 0.0, step)

            if step % 100 == 0:
                if ignore_frac > self.config.alerts.ignore_frac_bad:
                    logger.warning(
                        "Ignore fraction BAD (step %d): %.3f > %.3f",
                        step,
                        ignore_frac,
                        self.config.alerts.ignore_frac_bad,
                    )
                elif ignore_frac > self.config.alerts.ignore_frac_warn:
                    logger.warning(
                        "Ignore fraction warning (step %d): %.3f > %.3f",
                        step,
                        ignore_frac,
                        self.config.alerts.ignore_frac_warn,
                    )

            if grad_norm is not None:
                if self.config.monitoring.fail_fast_nonfinite:
                    if not self._is_finite_scalar(grad_norm):
                        raise FloatingPointError(
                            f"Non-finite grad norm at step {step}: {grad_norm}"
                        )

                if self.writer is not None:
                    self.writer.add_scalar("Gradients/norm", float(grad_norm), step)
                    self.writer.add_scalar("Gradients/clip_coef", float(clip_coef), step)
                    self.writer.add_scalar("Gradients/clipped", 1.0 if clipped else 0.0, step)

                if step % 100 == 0:
                    if float(grad_norm) > self.config.alerts.grad_norm_stop_max:
                        logger.warning(
                            "Global grad norm extremely high (step %d): %.4g > %.4g",
                            step,
                            float(grad_norm),
                            self.config.alerts.grad_norm_stop_max,
                        )
                    elif float(grad_norm) > self.config.alerts.grad_norm_warn_max:
                        logger.warning(
                            "Global grad norm high (step %d): %.4g > %.4g",
                            step,
                            float(grad_norm),
                            self.config.alerts.grad_norm_warn_max,
                        )
                    elif float(grad_norm) < self.config.alerts.grad_norm_warn_min:
                        logger.warning(
                            "Global grad norm low (step %d): %.4g < %.4g",
                            step,
                            float(grad_norm),
                            self.config.alerts.grad_norm_warn_min,
                        )

                    if float(clip_coef) < self.config.alerts.clip_coef_bad:
                        logger.warning(
                            "Clip coef BAD (step %d): %.3f < %.3f",
                            step,
                            float(clip_coef),
                            self.config.alerts.clip_coef_bad,
                        )
                    elif float(clip_coef) < self.config.alerts.clip_coef_warn:
                        logger.warning(
                            "Clip coef warning (step %d): %.3f < %.3f",
                            step,
                            float(clip_coef),
                            self.config.alerts.clip_coef_warn,
                        )

            self._log_detailed_gradient_stats(step)
            lr_used = float(self.optimizer.param_groups[0]["lr"])
            self._log_weight_update_ratios(
                step,
                lr_used=lr_used,
                grad_norm=grad_norm,
                clip_coef=clip_coef,
            )

        if hist_this_step:
            self._log_gradient_histograms(step)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss, grad_norm, tokens, samples

    @torch.no_grad()
    def evaluate(self, loader, *, step: int, max_batches: Optional[int] = None) -> dict[str, float]:
        """
        Run an evaluation loop on a held-out loader and log scalar metrics.

        Logs:
        - Loss/val: mean token cross-entropy (ignore_index-aware)
        - PPL/val: exp(Loss/val)
        - Data/ignore_frac_val: fraction of targets ignored in loss
        """
        if loader is None:
            return {}

        self._activation_log_enabled = False
        self.model.eval()

        total_nll = 0.0
        total_tokens = 0
        total_ignored = 0
        total_positions = 0

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.config.training.bf16 and torch.cuda.is_bf16_supported():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = self.model(input_ids)
                    loss = self._compute_loss(logits, labels)
            else:
                logits = self.model(input_ids)
                loss = self._compute_loss(logits, labels)

            labels_used = labels[..., 1:]
            valid = labels_used != -100
            valid_tokens = int(valid.sum().item())
            positions = int(valid.numel())

            total_positions += positions
            total_ignored += positions - valid_tokens
            if valid_tokens > 0:
                total_nll += float(loss.detach().cpu().item()) * valid_tokens
                total_tokens += valid_tokens

        val_loss = float("nan")
        if total_tokens > 0:
            val_loss = total_nll / total_tokens

        try:
            if self._is_finite_scalar(val_loss) and val_loss < 50:
                ppl = float(math.exp(val_loss))
            else:
                ppl = float("inf")
        except OverflowError:
            ppl = float("inf")

        ignore_frac_val = float(total_ignored) / float(total_positions) if total_positions > 0 else 0.0

        if self.writer is not None:
            self.writer.add_scalar("Loss/val", float(val_loss), step)
            self.writer.add_scalar("PPL/val", float(ppl), step)
            self.writer.add_scalar("Data/ignore_frac_val", float(ignore_frac_val), step)

        self.model.train()
        return {"val_loss": float(val_loss), "val_ppl": float(ppl), "ignore_frac_val": ignore_frac_val}

    def _compute_loss(self, logits, labels):
        """
        Compute language modeling loss.

        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len)
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        return self.criterion(shift_logits, shift_labels)

    def save_checkpoint(self, step, final=False):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-step-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_path = os.path.join(checkpoint_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)

        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_path)

        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        torch.save(self.scheduler.state_dict(), scheduler_path)

        import json
        from dataclasses import asdict

        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        if final:
            logger.info("Final checkpoint saved at step %d to %s", step, checkpoint_dir)
        else:
            logger.info("Checkpoint saved at step %d to %s", step, checkpoint_dir)

    def _get_grad_norm(self) -> float:
        """Compute global grad norm for logging."""
        grads = [param.grad.data for param in self.model.parameters() if param.grad is not None]
        norms = self._l2_norms(grads)
        return math.sqrt(sum(n * n for n in norms)) if norms else 0.0

    def _get_param_norm(self) -> float:
        """Compute global parameter norm for logging."""
        params = [param.data for param in self.model.parameters()]
        norms = self._l2_norms(params)
        return math.sqrt(sum(n * n for n in norms)) if norms else 0.0

    def _log_detailed_gradient_stats(self, step: int) -> None:
        """
        Log detailed per-layer gradient statistics.

        In Standard mode, this logs only a deterministic sentinel set.
        In Debug mode, it logs every parameter (expensive).
        """
        if self.monitoring_mode == "minimal":
            return

        if self.monitoring_mode == "debug":
            named_params = self.model.named_parameters()
        else:
            names = sorted(self._sentinel_param_names)
            named_params = [(name, self._named_params[name]) for name in names]

        for name, param in named_params:
            if param.grad is None:
                continue

            grad_norm = param.grad.norm(2).item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()

            if self.writer is not None:
                self.writer.add_scalar(f"gradients/{name}/norm", grad_norm, step)
                self.writer.add_scalar(f"gradients/{name}/mean", grad_mean, step)
                self.writer.add_scalar(f"gradients/{name}/std", grad_std, step)

            if self.monitoring_mode == "standard" and step % 100 == 0:
                logger.info(
                    "  %s: grad_norm=%.4f, mean=%.6f, std=%.6f",
                    name,
                    grad_norm,
                    grad_mean,
                    grad_std,
                )

    def _log_weight_update_ratios(
        self,
        step: int,
        *,
        lr_used: float,
        grad_norm: Optional[float],
        clip_coef: float,
    ) -> None:
        """
        Log weight-update ratios as a stability diagnostic.

        Per-parameter update ratio (proxy):
            ratio_i = (lr_used * ||g_i||_2) / (||w_i||_2 + eps)

        Global update ratio (proxy):
            ratio_global = (lr_used * (clip_coef * ||g||_2)) / (||w||_2 + eps)

        Notes:
        - This is a proxy for AdamW. We approximate the optimizer update as `lr * grad` and ignore
          Adam moments, weight decay, and per-parameter-group specifics.
        """
        if self.monitoring_mode == "minimal":
            return
        if grad_norm is None:
            return

        alerts = self.config.alerts

        param_norm = self._get_param_norm()
        if self.config.monitoring.fail_fast_nonfinite and not self._is_finite_scalar(param_norm):
            raise FloatingPointError(f"Non-finite param norm at step {step}: {param_norm}")

        ratio_global = compute_global_update_ratio(
            float(lr_used),
            grad_norm_pre_clip=float(grad_norm),
            clip_coef_value=float(clip_coef),
            param_norm=float(param_norm),
            eps=1e-12,
        )

        if self.writer is not None:
            self.writer.add_scalar("Parameters/norm", float(param_norm), step)
            self.writer.add_scalar("Updates/ratio_global", float(ratio_global), step)

        if step % 100 == 0:
            if ratio_global > alerts.update_ratio_warn_high:
                logger.warning(
                    "Global update ratio warning (step %d): %.4g > %.4g",
                    step,
                    ratio_global,
                    alerts.update_ratio_warn_high,
                )
            if ratio_global < alerts.update_ratio_warn_low:
                logger.warning(
                    "Global update ratio warning (step %d): %.4g < %.4g",
                    step,
                    ratio_global,
                    alerts.update_ratio_warn_low,
                )

        block_grad_norms: list[float] = []
        block_weight_norms: list[float] = []
        block_update_ratios: list[float] = []
        per_weight_ratios: list[float] = []

        for block_idx, entries in enumerate(self._block_main_params_by_block):
            weight_tensors: list[torch.Tensor] = []
            grad_tensors: list[torch.Tensor] = []
            for _name, param in entries:
                if param.grad is None:
                    continue
                weight_tensors.append(param.data)
                grad_tensors.append(param.grad)

            weight_norms = self._l2_norms(weight_tensors)
            grad_norms = self._l2_norms(grad_tensors)

            for w_norm, g_norm in zip(weight_norms, grad_norms):
                per_weight_ratios.append(
                    compute_update_ratio(
                        float(lr_used),
                        grad_norm=float(g_norm),
                        weight_norm=float(w_norm),
                        eps=1e-12,
                    )
                )

            block_weight_norm = math.sqrt(sum(w * w for w in weight_norms)) if weight_norms else 0.0
            block_grad_norm = math.sqrt(sum(g * g for g in grad_norms)) if grad_norms else 0.0
            block_update_ratio = compute_update_ratio(
                float(lr_used),
                grad_norm=float(block_grad_norm),
                weight_norm=float(block_weight_norm),
                eps=1e-12,
            )

            block_weight_norms.append(block_weight_norm)
            block_grad_norms.append(block_grad_norm)
            block_update_ratios.append(block_update_ratio)

            if self.writer is not None:
                self.writer.add_scalar(f"Gradients/block_{block_idx}/norm", block_grad_norm, step)
                self.writer.add_scalar(f"Updates/block_{block_idx}/ratio", block_update_ratio, step)

        ratio_p50 = self._quantile(per_weight_ratios, 0.50)
        ratio_p95 = self._quantile(per_weight_ratios, 0.95)
        ratio_max = max(per_weight_ratios) if per_weight_ratios else 0.0
        if self.writer is not None:
            self.writer.add_scalar("Updates/ratio_p50", float(ratio_p50), step)
            self.writer.add_scalar("Updates/ratio_p95", float(ratio_p95), step)
            self.writer.add_scalar("Updates/ratio_max", float(ratio_max), step)

        if step % 100 == 0 and per_weight_ratios:
            if ratio_p95 > alerts.update_ratio_warn_high or ratio_max > alerts.update_ratio_stop_high:
                logger.warning(
                    "Update ratio tail warning (step %d): p95=%.4g, max=%.4g",
                    step,
                    ratio_p95,
                    ratio_max,
                )
            if ratio_p50 < alerts.update_ratio_warn_low:
                logger.warning("Update ratio median low (step %d): p50=%.4g", step, ratio_p50)

        if step % 100 == 0 and block_weight_norms:
            min_w = min(block_weight_norms)
            max_w = max(block_weight_norms)
            if min_w > 0 and (max_w / min_w) > 1000:
                logger.warning(
                    "Weight norm variance > 1000× across blocks (step %d): min=%.4g, max=%.4g",
                    step,
                    min_w,
                    max_w,
                )

        if block_grad_norms:
            grad_depth_ratio = block_grad_norms[0] / (block_grad_norms[-1] + 1e-12)
            if self.writer is not None:
                self.writer.add_scalar(
                    "Gradients/depth_ratio_first_last",
                    float(grad_depth_ratio),
                    step,
                )
            if step % 100 == 0:
                if grad_depth_ratio < alerts.depth_ratio_bad:
                    logger.warning(
                        "Grad depth ratio BAD (step %d): %.4g < %.4g",
                        step,
                        grad_depth_ratio,
                        alerts.depth_ratio_bad,
                    )
                elif grad_depth_ratio < alerts.depth_ratio_warn:
                    logger.warning(
                        "Grad depth ratio warning (step %d): %.4g < %.4g",
                        step,
                        grad_depth_ratio,
                        alerts.depth_ratio_warn,
                    )

        if block_update_ratios:
            upd_depth_ratio = block_update_ratios[0] / (block_update_ratios[-1] + 1e-12)
            if self.writer is not None:
                self.writer.add_scalar(
                    "Updates/depth_ratio_first_last",
                    float(upd_depth_ratio),
                    step,
                )
            if step % 100 == 0:
                if upd_depth_ratio < alerts.depth_ratio_bad:
                    logger.warning(
                        "Update depth ratio BAD (step %d): %.4g < %.4g",
                        step,
                        upd_depth_ratio,
                        alerts.depth_ratio_bad,
                    )
                elif upd_depth_ratio < alerts.depth_ratio_warn:
                    logger.warning(
                        "Update depth ratio warning (step %d): %.4g < %.4g",
                        step,
                        upd_depth_ratio,
                        alerts.depth_ratio_warn,
                    )

        if self.monitoring_mode == "debug":
            named_params = self.model.named_parameters()
        else:
            names = sorted(self._sentinel_param_names)
            named_params = [(name, self._named_params[name]) for name in names]

        per_param_alerts: list[tuple[str, float, str]] = []
        for name, param in named_params:
            if param.grad is None:
                continue

            weight_norm = param.data.norm(2).item()
            grad_norm_i = param.grad.norm(2).item()
            update_norm = float(lr_used) * float(grad_norm_i)
            update_ratio = compute_update_ratio(
                float(lr_used),
                grad_norm=float(grad_norm_i),
                weight_norm=float(weight_norm),
                eps=1e-12,
            )

            if self.writer is not None:
                self.writer.add_scalar(f"updates/{name}/ratio", update_ratio, step)
                self.writer.add_scalar(f"updates/{name}/update_norm", update_norm, step)
                self.writer.add_scalar(f"updates/{name}/weight_norm", weight_norm, step)

            if update_ratio > 0.1:
                per_param_alerts.append((name, update_ratio, "TOO_HIGH"))
            elif update_ratio < 1e-7 and weight_norm > 1e-6:
                per_param_alerts.append((name, update_ratio, "TOO_LOW"))

        if per_param_alerts and step % 100 == 0:
            logger.warning("Update ratio alerts (step %d):", step)
            for name, ratio, issue in per_param_alerts[:25]:
                if issue == "TOO_HIGH":
                    logger.warning("  %s: %.4f > 0.1 (learning rate may be too high)", name, ratio)
                else:
                    logger.warning("  %s: %.8f < 1e-7 (learning rate may be too low)", name, ratio)

    def _log_gradient_histograms(self, step: int) -> None:
        """Log gradient histograms (sentinel-only in Standard, full in Debug)."""
        if self.writer is None:
            return
        if self.monitoring_mode == "minimal":
            return
        if step <= 0 or (step % self.histogram_steps) != 0:
            return

        if self.monitoring_mode == "debug":
            named_params = self.model.named_parameters()
        else:
            names = sorted(self._sentinel_param_names)
            named_params = [(name, self._named_params[name]) for name in names]

        for name, param in named_params:
            if param.grad is None:
                continue
            self.writer.add_histogram(f"gradients_hist/{name}", param.grad.data, step)

    def _log_weight_norms(self, step: int) -> None:
        """Log weight norms (bounded in Standard, full-but-throttled in Debug)."""
        if self.writer is None:
            return
        if self.monitoring_mode == "minimal":
            return

        if self.monitoring_mode == "debug":
            weight_log_freq = max(100, self.log_steps * 10)
            if step % weight_log_freq != 0:
                return
            for name, param in self.model.named_parameters():
                self.writer.add_scalar(f"weights/{name}/norm", param.data.norm(2).item(), step)
            return

        for name in sorted(self._sentinel_param_names):
            param = self._named_params[name]
            self.writer.add_scalar(f"weights/{name}/norm", param.data.norm(2).item(), step)

        block_norms: list[float] = []
        for block_idx, entries in enumerate(self._block_main_params_by_block):
            weight_tensors = [param.data for _name, param in entries]
            weight_norms = self._l2_norms(weight_tensors)
            block_norm = math.sqrt(sum(w * w for w in weight_norms)) if weight_norms else 0.0
            block_norms.append(block_norm)
            self.writer.add_scalar(f"Weights/block_{block_idx}/norm", block_norm, step)

        if step % 100 == 0 and block_norms:
            min_w = min(block_norms)
            max_w = max(block_norms)
            if min_w > 0 and (max_w / min_w) > 1000:
                logger.warning(
                    "Weight norm variance > 1000× across blocks (step %d): min=%.4g, max=%.4g",
                    step,
                    min_w,
                    max_w,
                )

    def _log_weight_changes(self, step):
        """
        Track how much weights are changing over time.

        NOTE: not used by Monitoring v2 by default, but kept as a debug utility.
        """
        if self.writer is None:
            return

        if not hasattr(self, "_prev_weight_norms"):
            self._prev_weight_norms = {}
            self._weight_log_step = 0
            return

        weight_change_freq = max(500, self.log_steps * 50)
        if step % weight_change_freq != 0:
            return

        changes = []
        for name, param in self.model.named_parameters():
            current_norm = param.data.norm(2).item()

            if name in self._prev_weight_norms:
                prev_norm = self._prev_weight_norms[name]
                change_pct = ((current_norm - prev_norm) / prev_norm) * 100 if prev_norm > 0 else 0.0
                changes.append(
                    {"name": name, "prev_norm": prev_norm, "current_norm": current_norm, "change_pct": change_pct}
                )

            self._prev_weight_norms[name] = current_norm

        if changes and step % (weight_change_freq * 10) == 0:
            changes_pct = [c["change_pct"] for c in changes]
            avg_change = sum(changes_pct) / len(changes_pct)
            changes.sort(key=lambda x: abs(x["change_pct"]), reverse=True)

            logger.info("Weight change summary (step %d):", step)
            logger.info("  Steps since last check: %d", step - self._weight_log_step)
            logger.info("  Average change: %.2f%%", avg_change)
            for c in changes[:5]:
                logger.info(
                    "  %s: %.4f → %.4f (%+.2f%%)",
                    c["name"],
                    c["prev_norm"],
                    c["current_norm"],
                    c["change_pct"],
                )

            self._weight_log_step = step

    def _log_weight_histograms(self, step: int) -> None:
        """Log weight histograms (sentinel-only in Standard, full in Debug)."""
        if self.writer is None:
            return
        if self.monitoring_mode == "minimal":
            return
        if step <= 0 or (step % self.histogram_steps) != 0:
            return

        if self.monitoring_mode == "debug":
            named_params = self.model.named_parameters()
        else:
            names = sorted(self._sentinel_param_names)
            named_params = [(name, self._named_params[name]) for name in names]

        for name, param in named_params:
            self.writer.add_histogram(f"weights_hist/{name}", param.data, step)

