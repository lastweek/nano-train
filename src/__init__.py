"""
nano-train package entrypoint.

We redirect Python bytecode caches to a single repo-local directory to avoid scattering
`__pycache__/` folders across the source tree. Users can override by setting:
- `PYTHONDONTWRITEBYTECODE=1` to disable bytecode caches entirely, or
- `PYTHONPYCACHEPREFIX=/some/path` to choose a different cache prefix.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _configure_pycache_prefix() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") == "1":
        return

    if os.environ.get("PYTHONPYCACHEPREFIX"):
        return

    repo_root = Path(__file__).resolve().parent.parent
    prefix = repo_root / ".pycache"
    try:
        prefix.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    os.environ["PYTHONPYCACHEPREFIX"] = str(prefix)
    try:
        sys.pycache_prefix = str(prefix)  # type: ignore[attr-defined]
    except Exception:
        pass


_configure_pycache_prefix()
del _configure_pycache_prefix

from src.config import Config
from src.config import DataConfig
from src.config import ModelConfig
from src.config import TrainingConfig

__all__ = ["Config", "ModelConfig", "TrainingConfig", "DataConfig"]
