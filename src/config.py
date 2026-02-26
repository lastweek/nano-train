"""
Configuration system for nano-train.

For MVP: Simple dataclass-based configuration.
For Phase 1+: Will upgrade to OmegaConf + Hydra.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Literal
from typing import Optional
import torch.distributed as dist


@dataclass
class ModelConfig:
    """Model architecture configuration (125M params for MVP)."""
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    vocab_size: int = 50257
    dropout: float = 0.1


@dataclass
class TPConfig:
    """
    Tensor Parallel configuration.

    When enabled, model weights are sharded across multiple GPUs.
    Each GPU stores 1/tp_size of the parameters.
    """

    # Enable tensor parallelism
    enabled: bool = False

    # TP rank of this process (0 to tp_size-1)
    rank: int = 0

    # Total number of TP processes
    size: int = 1

    # Process group for all-reduce operations
    # If None, uses dist.group.WORLD
    group: Optional[object] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.enabled:
            if self.size <= 1:
                raise ValueError("tp_size must be > 1 when TP is enabled")
            if self.rank < 0 or self.rank >= self.size:
                raise ValueError(f"tp_rank must be in [0, {self.size - 1}]")

            if self.group is None:
                self.group = dist.group.WORLD


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    micro_batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 100
    eval_steps: int = 50
    save_steps: int = 50
    log_steps: int = 10
    warmup_steps: int = 10
    clip_grad: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = False  # Will add in Phase 2


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_path: str = "examples/tiny_shakespeare.txt"
    max_seq_length: int = 256  # Reduced from 1024 for MVP with small dataset
    train_split: float = 0.9


@dataclass
class MonitoringConfig:
    """
    Monitoring configuration.

    The goal is to keep the default monitoring "bounded" (scales with model depth, not parameter
    count) while allowing a Debug mode for per-parameter deep dives.
    """

    mode: Literal["minimal", "standard", "debug"] = "standard"
    histogram_steps: int = 1000
    activation_sites: Literal["none", "sentinel", "all_lns"] = "sentinel"
    sync_cuda_timing: bool = False
    fail_fast_nonfinite: bool = True

    # TensorBoard writer buffering. Smaller values make runs feel more "live" at the cost of a bit
    # more overhead.
    tensorboard_max_queue: int = 1
    tensorboard_flush_secs: int = 1
    tensorboard_flush_on_log: bool = True

    # Attention monitoring uses this tau threshold for comparisons (inspired by QK-Clip-like caps).
    attn_tau: float = 100.0

    # Exponential moving average for loss spike detection.
    loss_ema_beta: float = 0.98

    # If None, Trainer derives (0, mid, last) from config.model.num_layers.
    sentinel_blocks: Optional[tuple[int, int, int]] = None

    # Optimizer state safety checks.
    opt_state_check_steps: int = 1000

    # Fixed-probe monitoring (behavioral invariants).
    probe_steps: int = 1000
    probe_topk: int = 10

    # MFU requires peak TFLOPs for the current device/system. If None, we log achieved TFLOPs only.
    peak_tflops: Optional[float] = None
    mfu_flops_multiplier: float = 6.0


@dataclass
class AlertThresholdsConfig:
    """
    Default alert thresholds for monitoring.

    These values are conservative heuristics for AdamW + pre-norm Transformers. They should be
    treated as first-line debugging hints, not universal invariants.
    """

    # Loss health checks
    loss_spike_warn: float = 3.0
    loss_spike_stop: float = 10.0

    # Gradient health checks (pre-clip global norm)
    grad_norm_warn_min: float = 1e-2
    grad_norm_warn_max: float = 1e2
    grad_norm_stop_max: float = 1e3

    # Gradient clipping health checks
    clip_coef_warn: float = 0.5
    clip_coef_bad: float = 0.1

    # Update ratio health checks (proxy)
    update_ratio_warn_low: float = 1e-7
    update_ratio_warn_high: float = 1e-1
    update_ratio_stop_high: float = 3e-1

    # Depth-ratio health checks
    depth_ratio_warn: float = 0.05
    depth_ratio_bad: float = 0.01

    # Data sanity checks
    ignore_frac_warn: float = 0.2
    ignore_frac_bad: float = 0.3

    # Activation sentinels (LayerNorm outputs)
    activation_rms_warn_low: float = 0.5
    activation_rms_warn_high: float = 2.0
    activation_rms_bad_low: float = 0.25
    activation_rms_bad_high: float = 4.0
    activation_max_abs_warn: float = 20.0
    activation_max_abs_bad: float = 50.0

    # Attention vitals
    max_attn_logit_warn: float = 200.0
    max_attn_logit_crit: float = 1000.0
    attn_entropy_norm_warn: float = 0.1

    # Residual stream tails
    residual_outlier_rate_warn: float = 1e-3
    residual_outlier_rate_bad: float = 1e-2

    # Performance (input-bound + OOM risk) checks
    data_wait_frac_warn: float = 0.3
    data_wait_frac_bad: float = 0.5
    reserved_frac_warn: float = 0.90
    reserved_frac_bad: float = 0.95


@dataclass
class DistributedConfig:
    """
    Distributed training configuration.

    Supports both CPU development (gloo backend) and GPU production (nccl backend).
    The same code works on both - only the backend changes.
    """

    # World size (number of devices/processes)
    # If None, auto-detect from available devices
    world_size: Optional[int] = None

    # Distributed backend
    # - "auto": Auto-detect based on CUDA availability
    # - "gloo": CPU-compatible (for development)
    # - "nccl": GPU-optimized (for production)
    backend: Literal["auto", "gloo", "nccl"] = "auto"

    # Master address and port for process group
    master_addr: str = "localhost"
    master_port: int = 29500

    # Device placement
    device: Optional[Literal["cpu", "cuda"]] = None  # Auto-detect

    # DDP settings
    find_unused_parameters: bool = False  # More efficient if False
    broadcast_buffers: bool = False

    # Gradient bucketing for communication efficiency
    bucket_cap_mb: int = 25

    # Megatron-style distributed optimizer / ZeRO sharding.
    use_distributed_optimizer: bool = False
    data_parallel_sharding_strategy: Literal["no_shard", "optim", "optim_grads"] = "no_shard"
    num_distributed_optimizer_instances: int = 1


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    alerts: AlertThresholdsConfig = field(default_factory=AlertThresholdsConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    # Output
    output_dir: str = "outputs"
    run_name: str = "nano_train_mvp"
    log_dir: str = "outputs"
    seed: int = 42
