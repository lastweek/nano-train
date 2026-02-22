"""
Unit tests for fixed-probe artifact hashing.

This is intentionally torch-free and deterministic.
"""

from __future__ import annotations

from src.monitoring import eval_artifact_hash


def test_eval_artifact_hash_deterministic_and_sensitive() -> None:
    h1 = eval_artifact_hash(
        input_ids_bytes=b"\x01\x02\x03",
        labels_bytes=b"\x04\x05",
        input_dtype="torch.int64",
        labels_dtype="torch.int64",
        input_shape=(1, 3),
        labels_shape=(1, 2),
        vocab_size=100,
        max_seq_length=16,
        seed=42,
    )
    h2 = eval_artifact_hash(
        input_ids_bytes=b"\x01\x02\x03",
        labels_bytes=b"\x04\x05",
        input_dtype="torch.int64",
        labels_dtype="torch.int64",
        input_shape=(1, 3),
        labels_shape=(1, 2),
        vocab_size=100,
        max_seq_length=16,
        seed=42,
    )
    assert h1 == h2

    h3 = eval_artifact_hash(
        input_ids_bytes=b"\x01\x02\x04",
        labels_bytes=b"\x04\x05",
        input_dtype="torch.int64",
        labels_dtype="torch.int64",
        input_shape=(1, 3),
        labels_shape=(1, 2),
        vocab_size=100,
        max_seq_length=16,
        seed=42,
    )
    assert h1 != h3
