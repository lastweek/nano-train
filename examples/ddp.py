#!/usr/bin/env python3
"""
Simple distributed training script using DDP.

This script demonstrates:
1. Device-agnostic distributed training (works on CPU and GPU)
2. Using gloo backend for CPU development
3. Using nccl backend for GPU production

Usage:
    # CPU development (4 processes)
    python examples/launch.py --world_size 4 --script examples/ddp.py

    # Or directly with torchrun
    torchrun --nproc_per_node=4 --backend=gloo examples/ddp.py

    # GPU production (4 GPUs)
    torchrun --nproc_per_node=4 examples/ddp.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.config import Config
from src.distributed.device import get_device_info, get_device, get_backend
from src.logging import setup_logging, get_logger

logger = get_logger(__name__)


def setup_distributed():
    """
    Initialize distributed training environment.

    Reads environment variables set by torchrun or launch_ddp.py:
    - RANK: Global rank of this process
    - WORLD_SIZE: Total number of processes
    - LOCAL_RANK: Local rank within this node
    - MASTER_ADDR: Address of the master node
    - MASTER_PORT: Port for communication

    Returns:
        tuple: (rank, world_size, local_rank, device)
    """
    # Get distributed info from environment
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Auto-detect backend
    backend = get_backend()

    logger.info(f"Initializing process group...")
    logger.info(f"  Rank: {rank}/{world_size}")
    logger.info(f"  Local Rank: {local_rank}")
    logger.info(f"  Backend: {backend}")

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )

    # Get device
    device_info = get_device_info()
    device = get_device(device_info.device_type, local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    logger.info(f"  Device: {device}")

    return rank, world_size, local_rank, device


class SimpleModel(nn.Module):
    """Simple model for testing distributed training."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


def create_dummy_dataset(batch_size: int = 32, num_samples: int = 1000):
    """Create a simple dummy dataset for testing."""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples):
            # Deterministic construction so all ranks shard the same underlying dataset.
            generator = torch.Generator().manual_seed(1234)
            self.data = torch.randn(num_samples, 10, generator=generator)
            self.targets = torch.randn(num_samples, 1, generator=generator)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    dataset = DummyDataset(num_samples)
    return dataset


def train_step(model, optimizer, data, target, device):
    """Single training step."""
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()
    output = model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    """Main training loop."""
    # Setup logging
    setup_logging(log_level="INFO")

    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()

    # Only log from rank 0 to avoid duplicate output
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Distributed DDP Training Test")
        logger.info("=" * 60)
        logger.info(f"World Size: {world_size}")
        logger.info(f"Device: {device}")
        logger.info(f"Backend: {dist.get_backend()}")
        logger.info("=" * 60)
        logger.info("")

    # Create model
    model = SimpleModel(hidden_size=128).to(device)

    # Wrap model with DDP
    # Note: device_ids must be specified for DDP
    if device.type == "cuda":
        model = DDP(model, device_ids=[local_rank])
    else:
        # For CPU, device_ids should be None
        model = DDP(model)

    if rank == 0:
        logger.info(f"Model wrapped with DDP")
        logger.info(f"  Device IDs: {model.device_ids if hasattr(model, 'device_ids') else 'N/A (CPU)'}")
        logger.info("")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create dataset and dataloader with DistributedSampler
    dataset = create_dummy_dataset(batch_size=32, num_samples=1000)

    # DistributedSampler ensures each process gets different data
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=0,
    )

    if rank == 0:
        logger.info("Starting training...")
        logger.info("")

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        # Set epoch for sampler to ensure different shuffling each epoch
        sampler.set_epoch(epoch)

        if rank == 0:
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            loss = train_step(model, optimizer, data, target, device)
            epoch_loss += loss
            num_batches += 1

        # Only rank 0 logs metrics
        if rank == 0:
            # Placeholder; true global average is logged below.
            avg_loss = epoch_loss / num_batches

        # Aggregate loss stats across all ranks for a global view.
        loss_stats = torch.tensor(
            [epoch_loss, float(num_batches)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
        global_avg_loss = float(loss_stats[0].item() / max(1.0, loss_stats[1].item()))

        if rank == 0:
            logger.info(f"  Average Loss: {global_avg_loss:.4f}")

    # Clean up
    if rank == 0:
        logger.info("")
        logger.info("Training completed successfully!")
        logger.info("")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
