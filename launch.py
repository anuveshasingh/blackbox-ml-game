"""
launch.py — zero-dependency bootstrapper for the Blackbox ML Game.

Players run `uv run launch.py <command>`. This script never touches the
game logic itself: it just figures out which prebuilt native binary
matches the current machine, downloads + caches it on first use, and then
replaces itself with that binary (passing through all CLI arguments).

The actual game (puzzles, plotting) is written in Rust on the `rust`
branch, and gets compiled into the binaries this script downloads. This branch (`binaries`) only
ever contains this launcher plus the GitHub Actions workflow that builds
those binaries — see CLAUDE.md before changing either half.
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path
from zipfile import ZipFile

REPOSITORY = "anuveshasingh/blackbox-ml-game"
RELEASE_TAG = "v0.4.1"

# Must match the `asset:` values in .github/workflows/build-binaries.yml
# exactly (no extensions baked in here — see CLAUDE.md "Windows asset
# naming" for why that bit us once).
_ASSET_NAMES = {
    ("Darwin", "arm64"): "blackbox-ml-game-macos-arm64",
    ("Darwin", "aarch64"): "blackbox-ml-game-macos-arm64",
    ("Darwin", "x86_64"): "blackbox-ml-game-macos-x86_64",
    ("Darwin", "amd64"): "blackbox-ml-game-macos-x86_64",
    ("Windows", "amd64"): "blackbox-ml-game-windows-x86_64",
    ("Windows", "x86_64"): "blackbox-ml-game-windows-x86_64",
    ("Linux", "x86_64"): "blackbox-ml-game-linux-x86_64",
    ("Linux", "amd64"): "blackbox-ml-game-linux-x86_64",
}


def asset_name() -> str:
    """Map this machine to the release asset built for it."""
    system = platform.system()
    machine = platform.machine().lower()
    name = _ASSET_NAMES.get((system, machine))
    if name is None:
        raise RuntimeError(
            f"Unsupported platform: {system} {platform.machine()}. "
            "Supported targets are macOS arm64/x86_64, Windows x86_64, and Linux x86_64."
        )
    return name


def _report_progress(block_count: int, block_size: int, total_size: int) -> None:
    downloaded = block_count * block_size
    if total_size > 0:
        downloaded = min(downloaded, total_size)
        percent = downloaded / total_size * 100
        status = f"{percent:5.1f}% ({downloaded / 1_048_576:.1f}/{total_size / 1_048_576:.1f} MB)"
    else:
        status = f"{downloaded / 1_048_576:.1f} MB"
    print(f"\rDownloading the game: {status}", end="", flush=True)


def _download_and_extract(name: str, cache_root: Path, bundle_dir: Path) -> None:
    """Fetch the release zip and extract it into `bundle_dir`, atomically."""
    archive_name = f"{name}.zip"
    url = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{archive_name}"
    temporary_archive = cache_root / f"{archive_name}.part"
    temporary_bundle = cache_root / f".{name}.tmp"

    print(f"Downloading the game for {platform.system()} {platform.machine()}...", flush=True)
    try:
        urllib.request.urlretrieve(url, temporary_archive, reporthook=_report_progress)
        print(flush=True)

        shutil.rmtree(temporary_bundle, ignore_errors=True)
        temporary_bundle.mkdir()
        with ZipFile(temporary_archive) as archive:
            archive.extractall(temporary_bundle)

        shutil.rmtree(bundle_dir, ignore_errors=True)
        temporary_bundle.replace(bundle_dir)
    finally:
        temporary_archive.unlink(missing_ok=True)
        shutil.rmtree(temporary_bundle, ignore_errors=True)


def binary_path(name: str) -> Path:
    """
    Return the path to a ready-to-run, cached copy of the `name` binary,
    downloading and extracting it first if this is the first run.

    Each release asset is a zip holding one self-contained executable
    (about 2 MB) named after the asset. It is extracted once, here, and
    every later launch runs the cached executable directly.
    """
    cache_root = Path.home() / ".cache" / "blackbox-ml-game" / RELEASE_TAG
    cache_root.mkdir(parents=True, exist_ok=True)

    bundle_dir = cache_root / name
    executable_name = f"{name}.exe" if os.name == "nt" else name
    executable = bundle_dir / executable_name

    if executable.exists() and not executable.is_file():
        # A previous run was interrupted mid-extraction; rebuild from scratch.
        shutil.rmtree(bundle_dir, ignore_errors=True)

    if not executable.is_file():
        _download_and_extract(name, cache_root, bundle_dir)
        if os.name != "nt":
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return executable


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
