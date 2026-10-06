"""
generator.py — Reproducible dataset generation for the numerical puzzles.

Usage
-----
    from blackbox_game.generator import generate_dataset, sample_count
    from blackbox_game.puzzles import get_puzzle

    puzzle  = get_puzzle("square_01")
    dataset = generate_dataset(puzzle, n_samples=sample_count(puzzle), seed=42)
    X, y    = dataset["X"], dataset["y"]

Outputs are exact function values: there is no noise of any kind.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import Puzzle
from .physics import evaluate_formula

#: Rows shown by ``play.py show``: 25 for beginner puzzles, 100 for physics
#: (their 3D plots need more points to show a surface).
SAMPLE_COUNTS = {1: 25, 2: 100}


def sample_count(puzzle: Puzzle) -> int:
    """Number of data points ``show`` generates for a numerical puzzle."""
    return SAMPLE_COUNTS[puzzle.difficulty]


# ---------------------------------------------------------------------------
# Input sampling  (rng, n_samples, params) → column dict
# ---------------------------------------------------------------------------

def _sample_x(rng, n, p):
    return {"x": rng.uniform(p["x_min"], p["x_max"], n)}


def _sample_physics(rng, n, p):
    return {name: rng.uniform(lo, hi, n) for name, (lo, hi) in p["ranges"].items()}


_SAMPLERS = {
    "linear": _sample_x,
    "quadratic": _sample_x,
    "sqrt_fn": _sample_x,
    "log_fn": _sample_x,
    "linear_plus_sin": _sample_x,
    "physics": _sample_physics,
}


# ---------------------------------------------------------------------------
# Hidden functions  (params, columns) → y
# ---------------------------------------------------------------------------

def _hidden_function(fn_type: str, p: dict, c: dict) -> np.ndarray:
    if fn_type == "linear":
        y = p["slope"] * c["x"] + p["intercept"]
    elif fn_type == "quadratic":
        y = p["a"] * c["x"] ** 2 + p["b"] * c["x"] + p["c"]
    elif fn_type == "sqrt_fn":
        y = p["a"] * np.sqrt(c["x"])
    elif fn_type == "log_fn":
        y = p["a"] * np.log(c["x"]) + p["b"]
    elif fn_type == "linear_plus_sin":
        y = p["slope"] * c["x"] + p["amplitude"] * np.sin(c["x"])
    elif fn_type == "physics":
        y = evaluate_formula(p["formula"], {**p.get("fixed", {}), **c})
    else:
        raise ValueError(f"Unknown function type '{fn_type}'.")
    return np.asarray(y, dtype=float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(
    puzzle: Puzzle,
    n_samples: int = 100,
    seed: int = 42,
) -> dict:
    """
    Generate a reproducible dataset for a numerical puzzle.

    Returns
    -------
    dict
        ``"X"`` — pandas DataFrame (n_samples × n_features)
        ``"y"`` — numpy array (n_samples,)
    """
    fn_type = puzzle.function.type
    if fn_type not in _SAMPLERS:
        raise ValueError(
            f"Unknown function type '{fn_type}'. Available: {sorted(_SAMPLERS)}"
        )

    rng = np.random.default_rng(seed)
    columns = _SAMPLERS[fn_type](rng, n_samples, puzzle.function.parameters)
    y = _hidden_function(fn_type, puzzle.function.parameters, columns)
    return {"X": pd.DataFrame(columns), "y": y}


def evaluate_points(puzzle: Puzzle, X: pd.DataFrame) -> dict:
    """Evaluate a puzzle's hidden function at user-supplied input points.

    ``X`` must contain the columns named by ``puzzle.input_features``. The
    returned ``y`` uses the same function as :func:`generate_dataset`, but
    keeps the caller's input rows.
    """
    missing = [column for column in puzzle.input_features if column not in X]
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(missing)}")

    values = X[puzzle.input_features].astype(float).copy()
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Input points must contain only finite numbers.")

    columns = {name: values[name].to_numpy() for name in values.columns}
    y = _hidden_function(puzzle.function.type, puzzle.function.parameters, columns)
    return {"X": values, "y": y}
