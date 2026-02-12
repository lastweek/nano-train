"""
Simple dataset and data loader for MVP.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(self, text_path, max_seq_length=256):
        """
        Args:
            text_path: Path to text file
            max_seq_length: Maximum sequence length (reduced from 1024 to 256 for MVP)
        """
        self.max_seq_length = max_seq_length

        # Load text
        print(f"Loading data from {text_path}...")
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Create character-level vocab for MVP (simple but works)
        # Phase 1+: will use proper tokenizer
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        print(f"Vocab size: {self.vocab_size}")
        print(f"Text length: {len(text)} characters")

        # Encode text
        encoded = [self.char_to_idx[ch] for ch in tqdm(text, desc="Encoding text")]

        # Create sequences with stride for better utilization
        self.sequences = []
        stride = max_seq_length // 2  # Use stride to create more sequences

        for i in range(0, len(encoded) - max_seq_length + 1, stride):
            self.sequences.append(encoded[i:i + max_seq_length])

        # If still no sequences, at least create one from beginning
        if len(self.sequences) == 0 and len(encoded) > 0:
            # Take whatever we can get
            seq_len = min(max_seq_length, len(encoded))
            self.sequences.append(encoded[:seq_len] + [0] * (max_seq_length - seq_len))

        print(f"Total sequences: {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Return a sequence with input_ids and labels.

        Labels are shifted by 1 for next token prediction.
        """
        sequence = self.sequences[idx]

        # Pad sequence if needed
        if len(sequence) < self.max_seq_length:
            padded_seq = sequence + [0] * (self.max_seq_length - len(sequence))
        else:
            padded_seq = sequence

        # Input: all tokens except last
        input_ids = torch.tensor(padded_seq[:-1], dtype=torch.long)

        # Labels: all tokens except first, shifted by 1
        labels = torch.tensor(padded_seq[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels
        }


def create_dataloader(dataset, batch_size, shuffle=True):
    """Create simple dataloader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # MVP: no multiprocessing
        drop_last=False,  # Don't drop last batch
    )
