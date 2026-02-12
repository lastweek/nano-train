"""
Transformer block and full model for MVP.

Phase 2 will add gradient checkpointing.
"""

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.models.attention import MultiHeadAttention
from src.models.mlp import MLP


class TransformerBlock(nn.Module):
    """Transformer decoder block with pre-norm architecture."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Attention
        self.attention = MultiHeadAttention(config)
        self.attn_norm = nn.LayerNorm(config.hidden_size)

        # MLP
        self.mlp = MLP(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, 1, seq_len, seq_len) or None

        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        # Pre-norm architecture
        # Attention block
        residual = x
        x = self.attn_norm(x)
        x = self.attention(x, attention_mask)
        x = residual + x  # Residual connection

        # MLP block
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x  # Residual connection

        return x


class TransformerModel(nn.Module):
    """GPT-style transformer model for language modeling."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Position embeddings (learned absolute positions)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size)

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying: share embeddings and lm_head weights
        self.lm_head.weight = self.token_embeddings.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02
            )
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, 1, seq_len, seq_len) or None

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        x = self.token_embeddings(input_ids)
        pos = self.position_embeddings(position_ids)
        x = x + pos
        x = self.dropout(x)

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = self._create_causal_mask(
                batch_size, seq_len, device=input_ids.device
            )

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final layer norm and head
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def _create_causal_mask(self, batch_size, seq_len, device):
        """Create causal attention mask for autoregressive generation."""
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device)
        )
        mask = mask.view(1, 1, seq_len, seq_len)
        mask = mask.expand(batch_size, -1, -1, -1)

        # Convert to attention mask format (0 for allowed, -inf for masked)
        attention_mask = mask.masked_fill(mask == 0, float('-inf'))

        return attention_mask

    @property
    def num_parameters(self):
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
