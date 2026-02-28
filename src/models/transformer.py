"""
Transformer block and full model for MVP.

Supports tensor parallelism via TPConfig parameter.
Phase 2 will add gradient checkpointing.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.config import TPConfig
from src.layers import ColumnParallelLinear
from src.layers import Dropout
from src.layers import Embedding
from src.layers import LayerNorm
from src.layers import RowParallelLinear
from src.models.attention import MultiHeadAttention
from src.models.mlp import MLP
from src.layers import Linear


class TransformerBlock(nn.Module):
    """
    Transformer decoder block with optional tensor parallelism.

    Passes tp_config to attention and MLP sub-layers.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        tp_config: Optional[TPConfig] = None,
        *,
        module_prefix: str,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.attention = MultiHeadAttention(
            config,
            tp_config,
            module_prefix=f"{module_prefix}.attention",
        )
        self.attn_norm = LayerNorm(
            config.hidden_size,
            param_dtype=config.param_dtype,
            param_device=config.param_device,
            module_path=f"{module_prefix}.attn_norm",
            precision_resolver=config.precision_resolver,
        )

        self.mlp = MLP(
            config,
            tp_config,
            module_prefix=f"{module_prefix}.mlp",
        )
        self.mlp_norm = LayerNorm(
            config.hidden_size,
            param_dtype=config.param_dtype,
            param_device=config.param_device,
            module_path=f"{module_prefix}.mlp_norm",
            precision_resolver=config.precision_resolver,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, 1, seq_len, seq_len) or None

        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        if x.dim() != 3:
            raise ValueError("x must have shape [batch_size, seq_len, hidden_size]")

        # Pre-norm attention block
        residual = x
        x = self.attn_norm(x)
        x = self.attention(x, attention_mask)
        x = residual + x

        # Pre-norm MLP block
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class TransformerModel(nn.Module):
    """
    GPT-style transformer model with optional tensor parallelism.

    If tp_config.enabled is True:
        - Attention and MLP use parallel linear layers
        - LM head uses ColumnParallelLinear (split vocab dimension)
        - Communication: 2 all-reduces per layer (attention + MLP)

    If tp_config.enabled is False (default):
        - Uses standard layers
        - No communication
    """

    def __init__(self, config: ModelConfig, tp_config: Optional[TPConfig] = None) -> None:
        super().__init__()
        tp_config = tp_config or TPConfig()

        self.config = config
        self.tp_enabled = tp_config.enabled
        self.tp_rank = tp_config.rank
        self.tp_size = tp_config.size
        self.tp_group = tp_config.group

        # Embeddings are always replicated (not sharded)
        self.token_embeddings = Embedding(
            config.vocab_size,
            config.hidden_size,
            param_dtype=config.param_dtype,
            param_device=config.param_device,
            module_path="token_embeddings",
            precision_resolver=config.precision_resolver,
        )
        self.position_embeddings = Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            param_dtype=config.param_dtype,
            param_device=config.param_device,
            module_path="position_embeddings",
            precision_resolver=config.precision_resolver,
        )
        self.dropout = Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config,
                layer_idx=i,
                tp_config=tp_config,
                module_prefix=f"blocks.{i}",
            )
            for i in range(config.num_layers)
        ])

        # Final layer norm
        self.ln_f = LayerNorm(
            config.hidden_size,
            param_dtype=config.param_dtype,
            param_device=config.param_device,
            module_path="ln_f",
            precision_resolver=config.precision_resolver,
        )

        # LM head
        if self.tp_enabled:
            # Column Parallel: split vocab dimension
            if config.vocab_size % self.tp_size != 0:
                raise ValueError(
                    f"vocab_size ({config.vocab_size}) must be divisible by "
                    f"tp_size ({self.tp_size})"
                )

            # ColumnParallelLinear expects GLOBAL vocab size and returns local shard.
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                tp_group=self.tp_group,
                bias=False,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
                module_path="lm_head",
                precision_resolver=config.precision_resolver,
            )
        else:
            # Standard Linear
            self.lm_head = Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
                module_path="lm_head",
                precision_resolver=config.precision_resolver,
            )
            # Weight tying: share embeddings and lm_head weights
            # Note: Not done in TP mode due to shape mismatch
            self.lm_head.weight = self.token_embeddings.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 style."""
        if isinstance(module, (Linear, ColumnParallelLinear, RowParallelLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, 1, seq_len, seq_len) or None

        Returns:
            logits: (batch_size, seq_len, vocab_size)
                    If TP enabled: sharded on vocab dim (vocab_size/tp_size)
                    If TP disabled: full vocab_size
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch_size, seq_len]")

        batch_size, seq_len = input_ids.shape

        # Position ids for absolute position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        x = self.token_embeddings(input_ids)
        pos = self.position_embeddings(position_ids)
        x = x + pos
        x = self.dropout(x)

        # Causal mask
        if attention_mask is None:
            attention_mask = self._create_causal_mask(
                batch_size, seq_len, device=input_ids.device
            )

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final layer norm and LM head
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def _create_causal_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create causal attention mask for autoregressive generation."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.view(1, 1, seq_len, seq_len)
        mask = mask.expand(batch_size, -1, -1, -1)
        attention_mask = mask.masked_fill(mask == 0, float("-inf"))
        return attention_mask

    @property
    def num_parameters(self) -> int:
        """Return total number of parameters on this GPU."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_parameters_global(self) -> int:
        """
        Return total number of parameters across all TP GPUs.

        For non-TP mode: same as num_parameters
        For TP mode: num_parameters * tp_size
        """
        if self.tp_enabled:
            return self.num_parameters * self.tp_size
        return self.num_parameters
