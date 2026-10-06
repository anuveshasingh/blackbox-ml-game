"""
viewer.py — Set up VS Code to open the 3D plot files.

The game installs the PLY viewer extension on first use, so the player only
runs ``play.py show <puzzle> --plot``. If the ``code`` command is not on PATH,
the functions report that and the game prints the file paths instead.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

VIEWER_EXTENSION_ID = "kleinicke.ply-visualizer"


def ensure_viewer_extension() -> str:
    """
    Install the PLY viewer into VS Code if it is missing.

    Returns one of: "ready" (already installed), "installed" (installed now),
    "no-vscode" (the ``code`` command is not on PATH), or "failed".
    """
    code = shutil.which("code")
    if code is None:
        return "no-vscode"
    try:
        listed = subprocess.run(
            [code, "--list-extensions"], capture_output=True, text=True, timeout=120,
        )
        installed = {line.strip().lower() for line in listed.stdout.splitlines()}
        if VIEWER_EXTENSION_ID in installed:
            return "ready"
        result = subprocess.run(
            [code, "--install-extension", VIEWER_EXTENSION_ID, "--force"],
            capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return "failed"
    return "installed" if result.returncode == 0 else "failed"


def open_in_vscode(paths: list[Path]) -> bool:
    """Open the given files in VS Code, reusing the current window."""
    code = shutil.which("code")
    if code is None or not paths:
        return False
    try:
        result = subprocess.run(
            [code, "--reuse-window", *map(str, paths)],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0
