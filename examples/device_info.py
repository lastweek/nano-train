#!/usr/bin/env python3
"""
Show device information for distributed training.

This script detects available devices (CPU/GPU) and prints configuration
for distributed training.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.distributed.device import get_device_info, print_device_info


def main():
    """Show device information and launch instructions."""
    print("Device Detection for Distributed Training")
    print("=" * 60)
    print()

    # Auto-detect device configuration
    info = get_device_info()
    print_device_info(info)

    print()
    print("Launch Commands:")
    print("-" * 60)

    if info.device_type == "cpu":
        print("CPU Development Mode:")
        print(f"  Launch {info.world_size} CPU processes:")
        print(f"  python examples/launch.py --world_size {info.world_size}")
        print()
        print("  Or directly with torchrun (using gloo backend):")
        print(f"  torchrun --nproc_per_node={info.world_size} \\")
        print(f"    --backend=gloo \\")
        print(f"    examples/train_ddp.py")
    else:
        print("GPU Production Mode:")
        print(f"  Launch on {info.world_size} GPUs:")
        print(f"  torchrun --nproc_per_node={info.world_size} examples/train_ddp.py")

    print()
    print("Environment Variables:")
    print("-" * 60)
    print(f"  MASTER_ADDR={os.environ.get('MASTER_ADDR', 'localhost')}")
    print(f"  MASTER_PORT={os.environ.get('MASTER_PORT', '29500')}")
    print()


if __name__ == "__main__":
    main()
