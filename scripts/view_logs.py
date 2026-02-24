#!/usr/bin/env python3
"""
Start TensorBoard for nano-train logs.

This is a convenience wrapper around `scripts/start_tensorboard.sh`, which can bootstrap a local
TensorBoard install without requiring TensorFlow.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Start TensorBoard for nano-train logs.")
    parser.add_argument("--logdir", default="outputs", help="Log directory passed to TensorBoard.")
    parser.add_argument("--port", default="6006", help="Port to bind TensorBoard to.")
    parser.add_argument("--host", default="localhost", help="Host to bind TensorBoard to.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    start_script = repo_root / "scripts" / "start_tensorboard.sh"
    if not start_script.exists():
        raise FileNotFoundError(f"Missing start script: {start_script}")

    cmd = [
        "bash",
        str(start_script),
        "--logdir",
        str(args.logdir),
        "--port",
        str(args.port),
        "--host",
        str(args.host),
    ]
    return subprocess.call(cmd, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())

