# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this branch is

`binaries` is the **player distribution branch** for the Blackbox ML Game, a command-line puzzle game (used in an ARIES competition) where players guess the hidden function behind an input/output dataset. This branch intentionally contains almost nothing:

- `launch.py` — the only file players run. It detects the OS/arch, downloads the matching prebuilt binary from a GitHub release, caches it under `~/.cache/blackbox-ml-game/<tag>/`, and `exec`s it with the player's CLI arguments.
- `.github/workflows/build-binaries.yml` — builds those binaries from the Rust game with `cargo build --release`, one per OS/arch, and publishes them to a GitHub release when a `v*` tag is pushed.
- `README.md` — the only setup instructions a player needs (`uv run launch.py list`).

The actual game — puzzles, dataset generation, image transforms, plotting — is a Rust crate on the **`rust`** branch (`Cargo.toml`, `src/*.rs`, `assets/`); `plan.md` there is the design record and answer key. That is the branch to edit for gameplay/puzzle changes. The older Python version is on `code` and is no longer built. This branch should only ever contain the launcher + the build workflow; don't add game source files here.

## Commands

```bash
uv run launch.py list                  # what a player runs; no other setup
uv run launch.py show puzzle_08 --plot
uv run launch.py apply --input photo.jpg --apply invert
```

There is nothing to build, lint, or test on *this* branch — `launch.py` is a single dependency-free script. To work on game logic, use the `rust` branch (`cargo build --release`, `cargo test`).

## Releasing a new binary build

1. Make the gameplay change on `rust`, commit and push it there.
2. On `binaries`, bump `RELEASE_TAG` in `launch.py` to the new version.
3. Push a tag matching that version (`git tag vX.Y.Z && git push --tags`). The workflow triggers on any `v*` tag push, checks out the **current tip of `rust`** (not a pinned commit), builds four binaries with cargo, smoke-tests each, zips each, and publishes them to a GitHub release named after the tag.
4. Confirm the release assets exist before telling players to update — `launch.py` will 404 instead of falling back if the tag/assets don't exist yet.
5. Fast-forward `main` to `binaries`: `git push origin binaries:main`. `main` is the GitHub default branch, so a plain `git clone` gets `main`'s `launch.py` and its `RELEASE_TAG`. If this step is skipped, players who clone without `-b binaries` keep downloading the previous release.

## Architecture gotchas (read before touching `launch.py` or the workflow)

- **Each release asset is a zip containing exactly one file: the executable, named after the asset** (`blackbox-ml-game-windows-x86_64.exe` on Windows, no extension elsewhere). `launch.py` extracts the zip into `~/.cache/blackbox-ml-game/<tag>/<asset>/` and runs `<asset>/<asset>` (plus `.exe` on Windows). Rename one side and the other breaks.
- **The asset name in `launch.py`'s `_ASSET_NAMES` must exactly match the `asset:` value in the workflow's build matrix** — no extensions baked in. `binary_path()` is the only place that appends `.exe`. A mismatch silently 404s the download on that platform.
- **Everything the game reads is compiled into the binary** (`include_bytes!` for the curated PNGs and the plot font on `rust`). There are no data files to bundle; a new runtime file must be embedded the same way.
- **Linux builds target `x86_64-unknown-linux-musl`** so the binary is static and runs on any distro regardless of glibc. The JPEG decoder (mozjpeg) is C compiled by the `cc` crate, which is why the workflow installs `musl-tools`. Windows links the C runtime statically (`+crt-static`) so players need no VC++ redistributable. macOS builds set `MACOSX_DEPLOYMENT_TARGET=11.0`.
- **`RELEASE_TAG` pins the exact release `launch.py` downloads.** It will never auto-update to a newer tag; that's deliberate (players shouldn't get a surprise mid-competition binary swap), but it means bumping the game requires both a new tag/release *and* a `launch.py` edit on this branch.
- **Cache layout:** bumping `RELEASE_TAG` points at a new cache directory; old versions' cached copies are left behind (harmless).
- **`os.execv` is used on macOS/Linux** so the launcher process is replaced in-place. Windows has no `execv` equivalent for this, hence the `subprocess.call` branch there.
