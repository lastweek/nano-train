#!/usr/bin/env python3
"""
Launch helper for distributed training scripts on CPU or GPU.

Execution patterns:
- Local development: spawn multiple ranks with torch.multiprocessing.
- torchrun jobs: load the target script in the current rank process.

Usage:
    # TP tutorial on 4 ranks (TP-only)
    python3 examples/launch.py --world-size 4 --backend gloo \
        --script examples/tp.py --script-args --tensor-model-parallel-size 4

    # TP tutorial on 4 ranks (TP=2, DP=2)
    python3 examples/launch.py --world-size 4 --backend gloo \
        --script examples/tp.py --script-args --tensor-model-parallel-size 2

    # 4P tutorial on 8 ranks (Tensor MP=2, Expert MP=2, Data Parallel=2)
    python3 examples/launch.py --world-size 8 --backend gloo \
        --script examples/train_4p.py --script-args \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 1 \
        --expert-model-parallel-size 2

    # Multi-GPU NCCL (recommended launch path)
    torchrun --nproc_per_node=4 examples/tp.py --tensor-model-parallel-size 2

Pass target-script flags after `--script-args`.
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.multiprocessing as mp

from src.config import Config
from src.distributed.device import DeviceInfo, get_device_info
from src.logging import setup_logging, get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch distributed training"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Number of processes/devices to use. If None, auto-detect.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "gloo", "nccl"],
        default="auto",
        help="Distributed backend to use (default: auto-detect).",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="examples/ddp.py",
        help="Training script to run (default: examples/ddp.py).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print launch commands without actually launching.",
    )
    parser.add_argument(
        "--script-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments passed through to the target script after --script-args.",
    )
    return parser.parse_args()


def setup_env_vars(world_size: int, backend: str):
    """Set up environment variables for distributed training."""
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")


def spawn_worker(rank: int, world_size: int, backend: str, script_path: str, script_args=None):
    """
    Worker function that runs in each spawned process.

    This function sets up environment variables and then imports and runs
    the training script. The training script is responsible for initializing
    the process group.

    Args:
        rank: Rank of this process (0 to world_size-1)
        world_size: Total number of processes
        backend: Distributed backend ('gloo' or 'nccl')
        script_path: Path to training script
        script_args: Additional arguments to pass to the target script.
    """
    import importlib

    # Set rank-specific environment variables for the training script
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    # Note: We do NOT initialize the process group here.
    # The training script is responsible for that.

    # Set sys.argv for the training script
    script_dir = os.path.dirname(os.path.abspath(script_path))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    script_name = os.path.basename(script_path)
    sys.argv = [script_name] + list(script_args or [])

    # Import and run the training script
    script_module = script_name.replace(".py", "")
    module = importlib.import_module(script_module)

    if hasattr(module, "main"):
        module.main()
    else:
        raise AttributeError(f"Script {script_path} must have a main() function")


def launch_multiprocessing(world_size: int, backend: str, script_path: str, script_args=None):
    """
    Launch multiple processes using torch.multiprocessing.

    Args:
        world_size: Number of processes to spawn
        backend: Distributed backend
        script_path: Path to training script
        script_args: Additional arguments to pass to training script
    """
    logger.info(f"Launching {world_size} processes with backend '{backend}'")
    if script_args:
        logger.info(f"Script args: {' '.join(script_args)}")

    mp.spawn(
        spawn_worker,
        args=(world_size, backend, script_path, script_args),
        nprocs=world_size,
        join=True,
    )


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level="INFO")

    # Get device info
    info = get_device_info(world_size=args.world_size, backend=args.backend)

    logger.info("=" * 60)
    logger.info("Distributed Training Launch")
    logger.info("=" * 60)
    logger.info(f"Device Type:   {info.device_type}")
    logger.info(f"Device Count:  {info.device_count}")
    logger.info(f"Backend:       {info.backend}")
    logger.info(f"World Size:    {info.world_size}")
    logger.info(f"Script:        {args.script}")
    logger.info("=" * 60)
    logger.info("")

    # Set up environment variables
    setup_env_vars(info.world_size, info.backend)

    if args.dry_run:
        logger.info("Dry run mode - not launching processes")
        logger.info("")
        logger.info("To launch for real, remove --dry-run flag")
        return

    # Launch training.
    if info.device_type == "cpu" or args.backend == "gloo":
        # Local launcher path: spawn all ranks from this process.
        logger.info("Using multiprocessing for CPU development")
        launch_multiprocessing(info.world_size, info.backend, args.script, args.script_args)
    else:
        # torchrun path: one OS process already exists per rank.
        logger.info("Running as part of torchrun distributed job")
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        spawn_worker(rank, world_size, info.backend, args.script, args.script_args)


if __name__ == "__main__":
    main()
