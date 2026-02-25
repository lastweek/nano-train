"""
Process launcher for distributed training on CPU or GPU.

This module provides utilities to launch multiple processes for distributed training:
- On CPU: Spawns multiple Python processes using multiprocessing
- On GPU: Prepares for torchrun/torch.distributed.launch

The key insight: same training script works on CPU and GPU.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch.multiprocessing as mp

from src.distributed.device import DeviceInfo, get_device_info, print_device_info


@dataclass
class LaunchConfig:
    """Configuration for launching distributed training."""

    world_size: int
    backend: str
    script_path: str
    script_args: list[str]


def launch_multiprocessing(
    script_path: str,
    world_size: int,
    backend: str = "gloo",
    script_args: Optional[list[str]] = None,
) -> None:
    """
    Launch training using multiprocessing (for CPU development).

    This spawns multiple Python processes, each with its own rank and world_size.

    Args:
        script_path: Path to the training script to run
        world_size: Number of processes to launch
        backend: Distributed backend ('gloo' for CPU, 'nccl' for GPU)
        script_args: Additional arguments to pass to the training script

    Examples:
        >>> # Launch training on 4 CPU processes
        >>> launch_multiprocessing(
        ...     script_path="examples/train_ddp.py",
        ...     world_size=4,
        ...     backend="gloo"
        ... )
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not Path(script_path).exists():
        raise FileNotFoundError(f"script_path does not exist: {script_path}")

    if script_args is None:
        script_args = []

    print(f"Launching {world_size} processes with backend '{backend}'")
    print(f"Script: {script_path}")
    print(f"Args: {script_args}")
    print()

    # Set environment variables for each process
    # These are read by torch.distributed.init_process_group()
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    # Launch processes
    mp.start_processes(
        _train_worker,
        args=(world_size, backend, script_path, script_args),
        nprocs=world_size,
        start_method="spawn",  # Use spawn for CUDA compatibility
    )


def _train_worker(
    rank: int,
    world_size: int,
    _backend: str,
    script_path: str,
    script_args: list[str],
) -> None:
    """
    Worker function that runs in each process.

    Args:
        rank: Rank of this process (0 to world_size-1)
        world_size: Total number of processes
        backend: Distributed backend
        script_path: Path to training script
        script_args: Arguments to pass to script
    """
    # Set rank-specific environment variables
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    # Import and run the training script
    # We need to add the script's directory to sys.path
    script_dir = os.path.dirname(os.path.abspath(script_path))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Get the module name from the script path
    script_module = os.path.basename(script_path).replace(".py", "")

    # Import the module and run its main function.
    # We inject script_args into sys.argv so scripts that parse CLI args via
    # argparse behave the same way they do under torchrun/python invocations.
    import importlib

    module = importlib.import_module(script_module)

    if not hasattr(module, "main"):
        raise AttributeError(f"Script {script_path} must have a main() function")

    original_argv = sys.argv
    try:
        sys.argv = [script_path, *script_args]
        module.main()
    finally:
        sys.argv = original_argv


def get_launch_command(
    world_size: int,
    script_path: str,
    script_args: Optional[list[str]] = None,
    use_torchrun: bool = True,
) -> str:
    """
    Generate the launch command for distributed training.

    This returns a command string that can be copied and run in a terminal.

    Args:
        world_size: Number of processes/devices
        script_path: Path to training script
        script_args: Arguments to pass to script
        use_torchrun: If True, generate torchrun command; otherwise generate python command

    Returns:
        Command string to launch distributed training

    Examples:
        >>> # Generate torchrun command (for GPU production)
        >>> cmd = get_launch_command(4, "examples/train_ddp.py")
        >>> print(cmd)
        torchrun --nproc_per_node=4 examples/train_ddp.py

        >>> # Generate python command (for CPU development)
        >>> cmd = get_launch_command(2, "examples/train_ddp.py", use_torchrun=False)
        >>> print(cmd)
        python examples/launch_ddp.py --world_size 2
    """
    if script_args is None:
        script_args = []

    args_str = " ".join(script_args)

    if use_torchrun:
        # Production: Use torchrun (for GPU)
        cmd = f"torchrun --nproc_per_node={world_size} {script_path} {args_str}"
    else:
        # Development: Use our launcher (for CPU)
        cmd = f"python {script_path} --world_size {world_size} {args_str}"

    return cmd.strip()


def print_launch_instructions(info: DeviceInfo, script_path: str) -> None:
    """
    Print instructions for launching distributed training.

    Args:
        info: DeviceInfo with detected configuration
        script_path: Path to the training script
    """
    print_device_info(info)
    print()

    if info.device_type == "cpu":
        print("Launch Instructions (CPU Development):")
        print("-" * 60)
        cmd = get_launch_command(info.world_size, script_path, use_torchrun=False)
        print(f"  {cmd}")
        print()
        print("Or programmatically:")
        print("  python examples/launch_ddp.py --world_size 4")
    else:
        print("Launch Instructions (GPU Production):")
        print("-" * 60)
        cmd = get_launch_command(info.world_size, script_path, use_torchrun=True)
        print(f"  {cmd}")
        print()
        print("This will use the NCCL backend for GPU communication.")

    print()
    print("Environment Variables:")
    print("-" * 60)
    print(f"  MASTER_ADDR={os.environ.get('MASTER_ADDR', 'localhost')}")
    print(f"  MASTER_PORT={os.environ.get('MASTER_PORT', '29500')}")
    print(f"  RANK=<set by launcher>")
    print(f"  WORLD_SIZE={info.world_size}")
    print(f"  LOCAL_RANK=<set by launcher>")
    print()


def auto_detect_and_print(script_path: str = "examples/train_ddp.py") -> DeviceInfo:
    """
    Auto-detect device configuration and print launch instructions.

    This is a convenience function for interactive use.

    Args:
        script_path: Path to the training script

    Returns:
        Detected DeviceInfo

    Examples:
        >>> info = auto_detect_and_print("examples/train_ddp.py")
        Device Configuration
        ==================
        Device Type:   CPU
        Device Count:  8
        Backend:       gloo
        World Size:    8
        ...
    """
    info = get_device_info()
    print_launch_instructions(info, script_path)
    return info


if __name__ == "__main__":
    # When run directly, show device info and launch instructions
    import argparse

    parser = argparse.ArgumentParser(description="Show device info for distributed training")
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Override auto-detected world size",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["gloo", "nccl", "auto"],
        default="auto",
        help="Distributed backend to use",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="examples/train_ddp.py",
        help="Training script path (for launch instructions)",
    )
    args = parser.parse_args()

    # Get device info
    info = get_device_info(world_size=args.world_size, backend=args.backend)

    # Print launch instructions
    print_launch_instructions(info, args.script)
