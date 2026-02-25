"""
Device abstraction layer for CPU/GPU agnostic distributed training.

This module provides a unified interface for device management that works with:
- CPU processes during development (gloo backend)
- GPUs during production (nccl backend)

The key insight: same code, different backend.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import torch


@dataclass
class DeviceInfo:
    """Information about available compute devices."""
    device_type: Literal["cpu", "cuda"]
    device_count: int
    backend: Literal["gloo", "nccl"]
    world_size: int


def get_device_info(
    world_size: Optional[int] = None,
    backend: Optional[Literal["gloo", "nccl", "auto"]] = None,
) -> DeviceInfo:
    """
    Auto-detect device information for distributed training.

    Args:
        world_size: Number of devices to use. If None, auto-detect.
        backend: Distributed backend. If "auto", detect based on available devices.

    Returns:
        DeviceInfo with detected configuration.

    Examples:
        >>> # Auto-detect on CPU machine
        >>> info = get_device_info()
        >>> print(info)
        DeviceInfo(device_type='cpu', device_count=8, backend='gloo', world_size=8)

        >>> # Force specific world size
        >>> info = get_device_info(world_size=4)
        >>> print(info.world_size)
        4

        >>> # Force specific backend
        >>> info = get_device_info(backend="gloo")
        >>> print(info.backend)
        'gloo'
    """
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()

    # Determine device type
    if cuda_available:
        device_type: Literal["cpu", "cuda"] = "cuda"
        device_count = torch.cuda.device_count()
    else:
        device_type = "cpu"
        # For CPU, we can use multiprocessing with any number of processes
        # Default to CPU count for world_size
        device_count = os.cpu_count() or 1

    # Determine backend
    if backend == "auto" or backend is None:
        if cuda_available:
            backend = "nccl"  # GPU backend
        else:
            backend = "gloo"  # CPU backend
    elif backend == "nccl" and not cuda_available:
        raise RuntimeError(
            "Requested 'nccl' backend but CUDA is not available. "
            "Use backend='gloo' for CPU distributed training."
        )
    elif backend == "gloo" and cuda_available:
        # Allow gloo on GPU (for testing), but warn
        warnings.warn(
            "Using 'gloo' backend with CUDA available. "
            "Consider using 'nccl' for better performance on GPUs.",
            UserWarning,
        )

    # Determine world size
    if world_size is None:
        world_size = device_count

    return DeviceInfo(
        device_type=device_type,
        device_count=device_count,
        backend=backend,
        world_size=world_size,
    )


def get_device(device_type: Literal["cpu", "cuda"], rank: int = 0) -> torch.device:
    """
    Get the appropriate torch.device for the given rank.

    Args:
        device_type: Type of device ('cpu' or 'cuda')
        rank: Rank of this process (for GPU device selection)

    Returns:
        torch.device object

    Examples:
        >>> # CPU device
        >>> dev = get_device("cpu")
        >>> print(dev)
        device(type='cpu')

        >>> # GPU device for rank 0
        >>> dev = get_device("cuda", rank=0)
        >>> print(dev)
        device(type='cuda', index=0)
    """
    if device_type == "cpu":
        return torch.device("cpu")
    if device_type == "cuda":
        return torch.device(f"cuda:{rank}")
    raise ValueError(f"Unknown device type: {device_type}")


def get_backend(
    backend: Optional[Literal["gloo", "nccl", "auto"]] = None,
) -> Literal["gloo", "nccl"]:
    """
    Get the appropriate distributed backend.

    Args:
        backend: Backend to use. If 'auto', detect based on CUDA availability.

    Returns:
        Backend string ('gloo' or 'nccl')

    Examples:
        >>> # Auto-detect
        >>> backend = get_backend()
        >>> if torch.cuda.is_available():
        ...     assert backend == "nccl"
        ... else:
        ...     assert backend == "gloo"
    """
    if backend == "auto" or backend is None:
        return "nccl" if torch.cuda.is_available() else "gloo"

    if backend not in ("gloo", "nccl"):
        raise ValueError(f"Unknown backend: {backend}")

    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested 'nccl' backend but CUDA is not available. "
            "Use backend='gloo' for CPU distributed training."
        )

    return backend


def validate_world_size(world_size: int, available_devices: int) -> None:
    """
    Validate that the requested world size doesn't exceed available devices.

    Args:
        world_size: Requested world size
        available_devices: Number of available devices

    Raises:
        ValueError: If world_size exceeds available devices
    """
    if world_size > available_devices:
        raise ValueError(
            f"Requested world_size ({world_size}) exceeds available devices ({available_devices}). "
            f"Reduce world_size or run on a machine with more devices."
        )


def print_device_info(info: DeviceInfo) -> None:
    """
    Print device information in a human-readable format.

    Args:
        info: DeviceInfo object to print
    """
    print("=" * 60)
    print("Device Configuration")
    print("=" * 60)
    print(f"Device Type:   {info.device_type.upper()}")
    print(f"Device Count:  {info.device_count}")
    print(f"Backend:       {info.backend}")
    print(f"World Size:    {info.world_size}")
    print("=" * 60)

    if info.device_type == "cpu":
        print(f"\nRunning on {info.world_size} CPU processes (development mode)")
        print("Each CPU process acts as a 'device' for distributed training")
        print("This simulates multi-GPU training for development and testing")
    else:
        print(f"\nRunning on {info.world_size} GPUs (production mode)")
        print("Using NCCL backend for high-performance GPU communication")
    print("=" * 60)
