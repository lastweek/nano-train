"""
Configuration system for nano-train.

For MVP: Simple dataclass-based configuration.
For Phase 1+: Will upgrade to OmegaConf + Hydra.
"""

from dataclasses import dataclass, field
from typing import Optional


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
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    micro_batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 1000
    eval_steps: int = 100
    save_steps: int = 500
    warmup_steps: int = 100
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
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Output
    output_dir: str = "outputs"
    run_name: str = "nano_train_mvp"
    log_dir: str = "logs"
    seed: int = 42
