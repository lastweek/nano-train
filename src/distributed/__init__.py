"""
Distributed training utilities for nano-train.

This module provides utilities for distributed training that work on both:
- CPU (development mode with gloo backend)
- GPU (production mode with nccl backend)

Key components:
- device: Device abstraction layer
- launch: Process launcher
- zero: ZeRO optimizer sharding (to be implemented)

Usage:
    from src.distributed import get_device_info, launch_multiprocessing

    # Auto-detect devices
    info = get_device_info()
    print(f"Running on {info.world_size} {info.device_type} devices")

    # Launch distributed training
    launch_multiprocessing(
        script_path="examples/train_ddp.py",
        world_size=4,
        backend="gloo"  # or "nccl" for GPU
    )
"""

from src.distributed.device import (
    DeviceInfo,
    get_backend,
    get_device,
    get_device_info,
    print_device_info,
    validate_world_size,
)
from src.distributed.launch import (
    LaunchConfig,
    auto_detect_and_print,
    get_launch_command,
    launch_multiprocessing,
    print_launch_instructions,
)

__all__ = [
    # Device info
    "DeviceInfo",
    "get_device_info",
    "get_device",
    "get_backend",
    "print_device_info",
    "validate_world_size",
    # Launch
    "LaunchConfig",
    "launch_multiprocessing",
    "get_launch_command",
    "print_launch_instructions",
    "auto_detect_and_print",
]
