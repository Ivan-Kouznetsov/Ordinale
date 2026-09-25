#!/usr/bin/env python3
"""Runner for offline unit tests in pre-commit hooks and local environments.

Automatically locates the local virtual environment (.venv) if present,
enforces offline mode (HF_HUB_OFFLINE=1), and executes tests marked as not requires_model.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parent.parent

    # Detect project-local virtual environment python if available
    venv_python_win = root / ".venv" / "Scripts" / "python.exe"
    venv_python_unix = root / ".venv" / "bin" / "python"

    if venv_python_win.is_file():
        python_executable = str(venv_python_win)
    elif venv_python_unix.is_file():
        python_executable = str(venv_python_unix)
    else:
        python_executable = sys.executable

    # Ensure offline mode is active to prevent unmocked Hugging Face network downloads
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

    cmd = [
        python_executable,
        "-m",
        "pytest",
        "-v",
        "-m",
        "not requires_model",
    ] + sys.argv[1:]

    result = subprocess.run(cmd, env=env)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
