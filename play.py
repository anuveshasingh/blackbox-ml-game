#!/usr/bin/env python3
"""
play.py — Blackbox ML Game  (run from the repo root)

COMMANDS
--------
  python play.py list [--difficulty 1|2|3]
      List all puzzle IDs (optionally filter by difficulty).

  python play.py show <puzzle_id>
      Show a puzzle's description and input data.

  python play.py transforms
      Print all available transformation keys.

QUICK START
-----------
  1.  pip install -e .
  2.  python play.py list
  3.  python play.py show puzzle_03
  4.  # record your result on the course leaderboard

INPUT FILE FORMAT
-----------------
  See README.md → "Input File Format" section.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import textwrap
from pathlib import Path

# ── Make the package importable even without `pip install -e .` ──────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from blackbox_game import (
    list_puzzles, get_puzzle, POWER_MAP,
)


# ── ANSI colours (VS Code terminal supports these) ────────────────────────
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_RED    = "\033[91m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

def _b(s):  return f"{_BOLD}{s}{_RESET}"
def _g(s):  return f"{_GREEN}{s}{_RESET}"
def _y(s):  return f"{_YELLOW}{s}{_RESET}"
def _c(s):  return f"{_CYAN}{s}{_RESET}"
def _r(s):  return f"{_RED}{s}{_RESET}"
def _d(s):  return f"{_DIM}{s}{_RESET}"


SEP  = _d("─" * 64)
SEP2 = _d("═" * 64)
_PLOT_EPSILON = 1.0
_PLOT_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────

def _diff_label(d: int) -> str:
    return {1: _g("Beginner"), 2: _y("Intermediate"), 3: _r("Challenge")}.get(d, str(d))


def _wrap(text: str, width: int = 60, indent: int = 2) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix,
                         subsequent_indent=prefix)


# ── Command: list ─────────────────────────────────────────────────────────

def cmd_list(args):
    difficulty = args.difficulty
    puzzles    = list_puzzles(difficulty=difficulty)

    label = ("All" if difficulty is None
             else {1: "Beginner", 2: "Intermediate", 3: "Challenge"}.get(difficulty, ""))
    print(f"\n{_b('Blackbox ML Game')} — {_b(label)} puzzles  ({len(puzzles)} total)\n")
    print(SEP)
    print(f"  {'ID':<22} Diff")
    print(SEP)
    for p in puzzles:
        print(
            f"  {_c(_puzzle_label(p['id'])):<31} "
            f"{_diff_label(p['difficulty'])}"
        )
    print(SEP)
    print(_d("  Run: python play.py show <puzzle_id>"))
    print()


# ── Command: show ─────────────────────────────────────────────────────────

def cmd_show(args):
    puzzle_id = _resolve_puzzle_id(args.puzzle_id)
    try:
        puzzle = get_puzzle(puzzle_id)
    except KeyError as e:
        print(_r(str(e)))
        sys.exit(1)

    from blackbox_game import generate_dataset

    dataset = generate_dataset(puzzle, n_samples=20, seed=42)
    X = dataset["X"]
    y = dataset["y"]
    base_plot = None
    if args.plot:
        _ensure_output_dirs()
        base_plot = _plot_output(
            X.assign(y=y),
            _puzzle_output_dir(puzzle_id) / f"{_puzzle_label(puzzle_id)}_base.csv",
            "points",
        )

    diff_label = {1: "Beginner", 2: "Intermediate", 3: "Challenge"}.get(
        puzzle.difficulty, str(puzzle.difficulty)
    )

    print(f"\n{SEP2}")
    print(f"  {_b(puzzle.title)}  [{_diff_label(puzzle.difficulty)} · {_d(puzzle.category)}]")
    print(f"  ID: {_c(_puzzle_label(puzzle_id))}")
    print(SEP2)
    print()
    print(_b("  Description"))
    print(_wrap(puzzle.description, width=70))
    print()
    print(_b("  Input features: ") + _c(", ".join(puzzle.input_features)))
    print()

    # Data preview
    print(_b(f"  Sample data ({len(X)} rows):"))
    print()
    header = "  " + "  ".join(f"{col:<10}" for col in X.columns) + "  y"
    print(_d(header))
    print(_d("  " + "─" * (len(header) - 2)))
    for i in range(len(X)):
        row_vals = "  ".join(f"{X.iloc[i][col]:>10.4f}" for col in X.columns)
        print(f"  {row_vals}  {y[i]:>10.4f}")
    print()

    print(_b("  Models you can try:"))
    print(f"    linear_regression")
    print(f"    decision_tree")
    print()
    print(SEP)
    print(_d("  Record your result on the course leaderboard."))
    if base_plot:
        print(f"  Base plot written: {_b(str(base_plot))}")
    print()


# ── Command: transforms ───────────────────────────────────────────────────

def cmd_transforms(args):
    from blackbox_game import list_binary_transforms, list_transforms

    print(f"\n{_b('Unary transforms')} — use as \"transform:column\" in your features list\n")
    print(f"  {'Key':<18} {'Description':<28} Power")
    print(SEP)
    for t in list_transforms():
        k    = t["key"]
        desc = t["description"]
        pw   = POWER_MAP.get(k, "")
        pw_s = f"  x^{pw}" if pw != "" else ""
        print(f"  {_c(k):<27} {desc:<28}{_d(pw_s)}")

    print(f"\n{_b('Binary transforms')} — use as dict in your features list\n")
    print(f"  {'Key':<18} Description")
    print(SEP)
    for t in list_binary_transforms():
        print(f"  {_c(t['key']):<27} {t['description']}")

    print()
    print(_b("  Feature spec formats:"))
    print('    "identity:x"                          → x (unary)')
    print('    "square:x"                            → x²  (unary)')
    print('    {"binary": "multiply", "a": "x1", "b": "x2"}  → x1 × x2')
    print()


def _read_points(path: str, puzzle):
    """Read one whitespace- or comma-separated input row per line."""
    import pandas as pd

    rows = []
    with open(path) as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values = [value for value in line.replace(",", " ").split() if value]
            try:
                row = [float(value) for value in values]
            except ValueError as exc:
                if not rows and [value.lower() for value in values] == puzzle.input_features:
                    continue
                raise ValueError(f"Line {line_number} contains a non-numeric value.") from exc
            if len(row) != len(puzzle.input_features):
                expected = ", ".join(puzzle.input_features)
                raise ValueError(
                    f"Line {line_number} has {len(row)} values; expected {len(puzzle.input_features)} ({expected})."
                )
            rows.append(row)

    if not rows:
        raise ValueError("Input file contains no data rows.")
    return pd.DataFrame(rows, columns=puzzle.input_features)


def _output_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path.cwd() / "outputs"
    return Path(__file__).resolve().parent / "outputs"


def _puzzle_number(puzzle_id: str) -> int:
    for number, puzzle in enumerate(list_puzzles(), 1):
        if puzzle["id"] == puzzle_id:
            return number
    raise KeyError(f"Unknown puzzle ID: {puzzle_id}")


def _puzzle_label(puzzle_id: str) -> str:
    return f"puzzle_{_puzzle_number(puzzle_id):02d}"


def _resolve_puzzle_id(identifier: str) -> str:
    if identifier.startswith("puzzle_"):
        try:
            number = int(identifier.removeprefix("puzzle_"))
        except ValueError:
            return identifier
        puzzles = list_puzzles()
        if 1 <= number <= len(puzzles):
            return puzzles[number - 1]["id"]
    return identifier


def _puzzle_output_dir(puzzle_id: str) -> Path:
    return _output_root() / _puzzle_label(puzzle_id)


def _ensure_output_dirs() -> Path:
    root = _output_root()
    root.mkdir(exist_ok=True)
    for number in range(1, len(list_puzzles()) + 1):
        (root / f"puzzle_{number:02d}").mkdir(exist_ok=True)
    return root


def _points_output_path(input_path: str, puzzle_id: str, suffix: str) -> Path:
    source = Path(input_path)
    label = _puzzle_label(puzzle_id)
    return _ensure_output_dirs() / label / f"{source.stem}_{label}_{suffix}.csv"


def _plot_output(
    data,
    csv_path: Path,
    kind: str,
) -> Path:
    """Save a non-interactive plot beside a generated CSV."""
    import numpy as np

    matplotlib_cache = Path.home() / ".cache" / "blackbox-ml-game" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)

    # Seed the persistent cache from the copy built at release time (see
    # build-binaries.yml) so a player's very first --plot ever doesn't pay
    # a full system font scan (several seconds to tens of seconds) — only
    # the bundle's download does. Harmless no-op once a real cache exists.
    if getattr(sys, "frozen", False) and not any(matplotlib_cache.glob("fontlist-*.json")):
        bundled_cache = Path(sys.executable).parent / "_internal" / "mplcache"
        if bundled_cache.is_dir():
            for cached_file in bundled_cache.glob("fontlist-*.json"):
                shutil.copy2(cached_file, matplotlib_cache / cached_file.name)

    # Must be a forced assignment, not setdefault(): PyInstaller's bundled
    # matplotlib runtime hook already sets MPLCONFIGDIR to a fresh
    # tempfile.mkdtemp() directory before this script runs, so setdefault()
    # is always a no-op in the frozen binary — the font cache was silently
    # rebuilding (and being discarded) on every single launch.
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = csv_path.with_suffix(".png")
    rng = np.random.default_rng(_PLOT_SEED)
    plot_noise = lambda size: rng.uniform(-_PLOT_EPSILON, _PLOT_EPSILON, size)
    if kind == "points":
        input_columns = [
            column for column in data.columns
            if column not in {"y", "prediction", "residual"}
        ]
        figure, axes = plt.subplots(
            len(input_columns), 1,
            figsize=(7, max(4, 3.5 * len(input_columns))),
            squeeze=False,
        )
        plotted_y = data["y"].to_numpy() + plot_noise(len(data))
        for axis, column in zip(axes.flat, input_columns):
            axis.scatter(data[column], plotted_y, alpha=0.7, s=24)
            axis.set_xlabel(column)
            axis.set_ylabel("y")
            axis.set_title(f"{column} vs y")
            axis.grid(True, alpha=0.3)
    else:
        figure, axis = plt.subplots(figsize=(7, 4))
        plotted_residual = data["residual"].to_numpy() + plot_noise(len(data))
        axis.scatter(data["prediction"], plotted_residual, alpha=0.7, s=24)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("prediction")
        axis.set_ylabel("residual (y - prediction) + plot noise")
        axis.set_title(f"Residuals vs prediction (epsilon <= {_PLOT_EPSILON:g})")
        axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return plot_path


def cmd_points(args):
    try:
        from blackbox_game import evaluate_points

        puzzle = get_puzzle(_resolve_puzzle_id(args.puzzle_id))
        points = _read_points(args.input, puzzle)
        result = evaluate_points(puzzle, points, include_noise=not args.no_noise)
        output = Path(args.output) if args.output else _points_output_path(args.input, puzzle.id, "output")
        values = result["X"].assign(y=result["y"])
        values.to_csv(output, index=False)
        plot = _plot_output(values, output, "points") if args.plot else None
    except (OSError, ValueError, KeyError) as exc:
        print(_r(f"Error: {exc}"))
        sys.exit(1)
    print(f"\n{_g('✓')} Points evaluated: {_b(str(output))}")
    print(f"  {len(result['X'])} rows, columns: {', '.join(result['X'].columns)} + y\n")
    if plot:
        print(f"  Plot written: {_b(str(plot))}\n")


def cmd_residuals(args):
    try:
        from blackbox_game import evaluate_points
        from blackbox_game.evaluator import build_feature_matrix, predict_model

        puzzle = get_puzzle(_resolve_puzzle_id(args.puzzle_id))
        points = _read_points(args.input, puzzle)
        result = evaluate_points(puzzle, points, include_noise=not args.no_noise)
        X_feat = build_feature_matrix(result["X"], args.features)
        task = "classification" if puzzle.function.type == "circle_classify" else "regression"
        predictions = predict_model(X_feat, result["y"], args.model, task=task)
        output = Path(args.output) if args.output else _points_output_path(args.input, puzzle.id, "residuals")
        residuals = result["X"].assign(
            y=result["y"], prediction=predictions, residual=result["y"] - predictions
        )
        residuals.to_csv(output, index=False)
        plot = _plot_output(residuals, output, "residuals") if args.plot else None
    except (OSError, ValueError, KeyError) as exc:
        print(_r(f"Error: {exc}"))
        sys.exit(1)
    print(f"\n{_g('✓')} Residuals written: {_b(str(output))}")
    print(f"  {len(residuals)} rows, model: {args.model}\n")
    if plot:
        print(f"  Plot written: {_b(str(plot))}\n")


# ── Argument parser ───────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="play.py",
        description="Blackbox ML Game — discover hidden functions through experimentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="List all puzzles")
    p_list.add_argument("--difficulty", type=int, choices=[1, 2, 3], default=None,
                        help="Filter: 1=Beginner, 2=Intermediate, 3=Challenge")

    # show
    p_show = sub.add_parser("show", help="Show a puzzle and its data")
    p_show.add_argument("puzzle_id", help="Puzzle ID (e.g. puzzle_03)")
    p_show.add_argument("--plot", action="store_true",
                        help="Also save the generated points as <puzzle_id>_base.png")

    # transforms
    sub.add_parser("transforms", help="List all available transforms")

    # points and residuals
    for command, handler_help in (
        ("points", "Evaluate a puzzle at user-supplied input points"),
        ("residuals", "Write residuals for a model fitted to supplied points"),
    ):
        point_parser = sub.add_parser(command, help=handler_help)
        point_parser.add_argument("puzzle_id", help="Puzzle ID (e.g. puzzle_03)")
        point_parser.add_argument("--input", required=True, help="Text file with one input point per line")
        point_parser.add_argument("--output", help="CSV path (default: input stem plus puzzle ID and command)")
        point_parser.add_argument("--no-noise", action="store_true",
                                  help="Do not add the puzzle's configured output noise")
        point_parser.add_argument("--plot", action="store_true",
                      help="Also save a PNG plot beside the CSV")
        if command == "residuals":
            point_parser.add_argument("--features", nargs="+", required=True,
                                      help="Feature specifications, e.g. identity:x sin:x")
            point_parser.add_argument("--model", choices=["linear_regression", "decision_tree"],
                                      default="linear_regression")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "list":       cmd_list,
        "show":       cmd_show,
        "transforms": cmd_transforms,
        "points":     cmd_points,
        "residuals":  cmd_residuals,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
