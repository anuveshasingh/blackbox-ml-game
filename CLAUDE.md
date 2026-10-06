# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this branch is

`preesha-binaries` is the **player distribution branch** for the Blackbox ML Game, a command-line puzzle game (used in an ARIES competition) where players guess the hidden function behind an input/output dataset. This branch intentionally contains almost nothing:

- `launch.py` — the only file players run. It detects the OS/arch, downloads the matching prebuilt binary from a GitHub release, caches it under `~/.cache/blackbox-ml-game/<tag>/`, and `exec`s it with the player's CLI arguments.
- `.github/workflows/build-binaries.yml` — builds those binaries with PyInstaller, one per OS/arch, and publishes them to a GitHub release when a `v*` tag is pushed.
- `README.md` — the only setup instructions a player needs (`uv run launch.py list`).

The actual game — puzzles, dataset generation, image transforms, plotting (`play.py` and `src/blackbox_game/`) — lives on the **`preesha`** branch with its own `pyproject.toml`. That branch has no test suite; `plan.md` there is the design record and answer key. That is the branch to edit for gameplay/puzzle changes. This branch should only ever contain the launcher + the build workflow; don't add game source files or a `pyproject.toml` here.

## Commands

```bash
uv run launch.py list                  # what a player runs; no other setup
uv run launch.py show puzzle_08 --plot
uv run launch.py submit my_answers.json
```

There is nothing to build, lint, or test on *this* branch — `launch.py` is a single dependency-free script. To work on game logic, use the `preesha` branch (or `git show origin/preesha:<path>`), which has its own `pyproject.toml` and `requirements.txt`.

## Releasing a new binary build

1. Make the gameplay change on `preesha`, merge/commit it there.
2. On `preesha-binaries`, bump `RELEASE_TAG` in `launch.py` to the new version.
3. Push a tag matching that version (`git tag vX.Y.Z && git push --tags`). The workflow triggers on any `v*` tag push, checks out the **current tip of `preesha`** (not a pinned commit — see gotcha below), builds four binaries with PyInstaller, zips each, and publishes them to a GitHub release named after the tag.
4. Confirm the release assets exist before telling players to update — `launch.py` will 404 instead of falling back if the tag/assets don't exist yet.

## Architecture gotchas (read before touching `launch.py` or the workflow)

- **PyInstaller must build `--onedir`, never `--onefile`.** A onefile binary re-extracts its entire payload (numpy/pandas/scikit-learn/matplotlib, ~100MB+) into a fresh temp directory on *every single launch* — this was the original "every command is way too slow" bug. Onedir + the cache in `binary_path()` means extraction happens once per machine, ever.
- **The PyInstaller step must keep `--collect-data blackbox_game`.** It bundles the package's non-Python files — the curated image-puzzle PNGs in `src/blackbox_game/curated/`. Without it the build succeeds but every image puzzle crashes at runtime. Any new data file the game reads must live inside the `blackbox_game` package (and be listed under `[tool.setuptools.package-data]` on `preesha`) to be bundled.
- **The asset name in `launch.py`'s `_ASSET_NAMES` must exactly match the `asset:` value in the workflow's build matrix** — no extensions baked in (including on Windows: the asset is `blackbox-ml-game-windows-x86_64`, not `...x86_64.exe`). `binary_path()` is the only place that appends `.exe`, and only for the extracted executable path. A mismatch here silently 404s the download on that platform.
- **`RELEASE_TAG` pins the exact release `launch.py` downloads.** It will never auto-update to a newer tag; that's deliberate (players shouldn't get a surprise mid-competition binary swap), but it means bumping the game requires both a new tag/release *and* a `launch.py` edit on this branch.
- **Cache layout:** `~/.cache/blackbox-ml-game/<RELEASE_TAG>/<asset-name>/` holds the extracted onedir bundle; the executable inside shares the same name (plus `.exe` on Windows). Bumping `RELEASE_TAG` naturally invalidates old caches by pointing at a new path — old versions' cached bundles are simply left behind (harmless, just disk space).
- **`os.execv` is used on macOS/Linux** so the launcher process is replaced in-place (no parent Python process hanging around). Windows has no `execv` equivalent for this, hence the `subprocess.call` branch there.
- **PyInstaller's bundled matplotlib runtime hook sets `MPLCONFIGDIR` to a fresh `tempfile.mkdtemp()` directory before `play.py` ever runs.** `play.py`'s `_plot_output()` must set `os.environ["MPLCONFIGDIR"]` with a plain assignment, never `setdefault()` — `setdefault()` is always a no-op here because the var is already set, which silently defeats the font-cache persistence and makes every `--plot` command rebuild (and discard) the whole font cache from scratch, costing several seconds to tens of seconds every single time. This is easy to reintroduce by "cleaning up" that line back to `setdefault()`.
- **The build workflow pre-builds matplotlib's font cache and bundles it** (the "Prebuild matplotlib font cache" step → `dist/<asset>/_internal/mplcache/`), so a player's very first `--plot` ever doesn't pay a live font scan either. `play.py`'s `_plot_output()` seeds the player's persistent cache from that bundled file on first run. If you ever change how/where PyInstaller lays out `_internal/`, or rename the matplotlib cache step, update both halves together — they're coupled by that exact relative path.
