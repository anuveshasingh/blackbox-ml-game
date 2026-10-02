from __future__ import annotations

import os
import platform
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

REPOSITORY = "anuveshasingh/blackbox-ml-game"


def asset_name() -> str:
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return "blackbox-ml-game-macos-arm64"
    if system == "Darwin" and machine in {"x86_64", "amd64"}:
        return "blackbox-ml-game-macos-x86_64"
    if system == "Windows" and machine in {"amd64", "x86_64"}:
        return "blackbox-ml-game-windows-x86_64.exe"
    if system == "Linux" and machine in {"x86_64", "amd64"}:
        return "blackbox-ml-game-linux-x86_64"

    raise RuntimeError(
        f"Unsupported platform: {system} {platform.machine()}. "
        "Supported targets are macOS arm64/x86_64, Windows x86_64, and Linux x86_64."
    )


def binary_path(name: str) -> Path:
    cache_root = Path.home() / ".cache" / "blackbox-ml-game"
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / name
    if not path.exists():
        url = f"https://github.com/{REPOSITORY}/releases/latest/download/{name}"
        temporary_path = path.with_name(f"{path.name}.part")

        def report_progress(block_count: int, block_size: int, total_size: int) -> None:
            downloaded = block_count * block_size
            if total_size > 0:
                downloaded = min(downloaded, total_size)
                percent = downloaded / total_size * 100
                status = f"{percent:5.1f}% ({downloaded / 1_048_576:.1f}/{total_size / 1_048_576:.1f} MB)"
            else:
                status = f"{downloaded / 1_048_576:.1f} MB"
            print(f"\rDownloading the game: {status}", end="", flush=True)

        print(f"Downloading the game for {platform.system()} {platform.machine()}...", flush=True)
        try:
            urllib.request.urlretrieve(url, temporary_path, reporthook=report_progress)
            temporary_path.replace(path)
            print("", flush=True)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        if os.name != "nt":
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def main() -> int:
    try:
        executable = binary_path(asset_name())
    except Exception as exc:
        print(f"Could not prepare the game: {exc}", file=sys.stderr)
        return 1

    if os.name == "nt":
        return subprocess.call([str(executable), *sys.argv[1:]])
    os.execv(str(executable), [str(executable), *sys.argv[1:]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
