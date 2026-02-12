"""
Simple GPU detection and availability checker.
Run this to verify your PyTorch + CUDA setup.
"""

import sys


def check_gpu():
    """Check GPU/CUDA availability and print system info."""
    print("=" * 60)
    print("GPU Detection for Nano-Train")
    print("=" * 60)

    # Check Python version
    print(f"\nPython Version: {sys.version}")

    # Try to import PyTorch
    try:
        import torch
        print(f"\n✓ PyTorch {torch.__version__} installed")

        # Check CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA Available: {torch.version.cuda}")
            print(f"✓ CUDA Driver: {torch.version.cuda or 'N/A'}")
            gpu_count = torch.cuda.device_count()
            print(f"✓ GPU Count: {gpu_count}")

            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                print(f"\n  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
        else:
            print("✗ CUDA NOT available")
            print("  CPU-only mode will be used")

        # Check for common issues
        if torch.cuda.is_available():
            # Try a simple CUDA operation
            try:
                import torch.nn as nn
                x = torch.randn(10, 10).cuda()
                y = x * 2
                y = y.cpu()
                print("\n✓ GPU compute test: PASSED")
            except RuntimeError as e:
                print(f"\n✗ GPU compute test: FAILED - {e}")

    except ImportError:
        print("✗ PyTorch NOT installed")
        print("\n" + "=" * 60)
        print("\nACTION REQUIRED:")
        print("  Install PyTorch: pip install torch torchvision")
        print("\nFor Google Colab:")
        print("  No action needed - Colab has PyTorch pre-installed")


if __name__ == "__main__":
    check_gpu()
