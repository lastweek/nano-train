"""
Simple dataset and data loader for MVP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from typing import TypedDict

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from tqdm import tqdm

from src.logging import get_logger


# Get logger for this module
logger = get_logger(__name__)


class Batch(TypedDict):
    input_ids: torch.Tensor
    labels: torch.Tensor


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(self, text_path: str, max_seq_length: int = 256) -> None:
        """
        Args:
            text_path: Path to text file
            max_seq_length: Maximum sequence length (reduced from 1024 to 256 for MVP)
        """
        if max_seq_length < 2:
            raise ValueError("max_seq_length must be at least 2")

        path = Path(text_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {text_path}")

        self.max_seq_length = max_seq_length

        # Load text
        logger.info("Loading data from %s...", text_path)
        with path.open("r", encoding="utf-8") as f:
            text = f.read()

        # Create character-level vocab for MVP (simple but works)
        # Phase 1+: will use proper tokenizer
        # Reserve a dedicated PAD token so padding is distinct from real characters.
        self.pad_token = "<pad>"
        self.pad_id = 0
        self.chars = [self.pad_token] + sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        logger.info("Vocab size: %d", self.vocab_size)
        logger.info("Text length: %d characters", len(text))

        # Encode text into integer ids. Length equals number of characters.
        encoded = [self.char_to_idx[ch] for ch in tqdm(text, desc="Encoding text")]

        # Create overlapping sequences to improve data utilization on tiny text.
        self.sequences = []
        stride = max(1, max_seq_length // 2)  # Use stride to create more sequences

        for i in range(0, len(encoded) - max_seq_length + 1, stride):
            self.sequences.append(encoded[i:i + max_seq_length])

        # If still no sequences, at least create one from beginning
        if len(self.sequences) == 0 and len(encoded) > 0:
            # Take whatever we can get
            seq_len = min(max_seq_length, len(encoded))
            self.sequences.append(encoded[:seq_len] + [self.pad_id] * (max_seq_length - seq_len))

        logger.info("Total sequences: %d", len(self.sequences))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Batch:
        """
        Return a sequence with input_ids and labels.

        Labels are shifted by 1 for next token prediction.
        """
        sequence = self.sequences[idx]

        # Pad sequence if needed
        if len(sequence) < self.max_seq_length:
            padded_seq = sequence + [self.pad_id] * (self.max_seq_length - len(sequence))
        else:
            padded_seq = sequence

        # Input: all tokens except last -> (seq_len - 1,)
        input_ids = torch.tensor(padded_seq[:-1], dtype=torch.long)

        # Labels: all tokens except first -> (seq_len - 1,)
        # This shift makes each position predict the next token.
        labels = torch.tensor(padded_seq[1:], dtype=torch.long)
        # Ignore padding positions in the loss.
        labels[labels == self.pad_id] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    sampler: Optional[Sampler] = None,
) -> DataLoader:
    """Create simple dataloader with optional sampler override."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    use_shuffle = bool(shuffle) and sampler is None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=use_shuffle,
        sampler=sampler,
        num_workers=0,  # MVP: no multiprocessing
        drop_last=False,  # Don't drop last batch
    )
