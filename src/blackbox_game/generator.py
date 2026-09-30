"""
generator.py — Reproducible dataset generation for every puzzle.

Each puzzle has a ``FunctionSpec`` that describes the hidden function.
This module maps those specs to numpy expressions and generates (X, y) pairs.

Usage
-----
    from blackbox_game.generator import generate_dataset
    from blackbox_game.puzzles import get_puzzle

    puzzle = get_puzzle("square_01")
    dataset = generate_dataset(puzzle, n_samples=100, seed=42)

    X = dataset["X"]   # pandas DataFrame  (n_samples × n_features)
    y = dataset["y"]   # numpy array        (n_samples,)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from .models import Puzzle, FunctionSpec


# ---------------------------------------------------------------------------
# Internal function implementations
# Each function takes rng, n, params → (X dict, y array)
# ---------------------------------------------------------------------------

def _fn_linear(rng, n, p):
    """y = slope * x + intercept"""
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 10.0)
    x = rng.uniform(lo, hi, n)
    y = p.get("slope", 1.0) * x + p.get("intercept", 0.0)
    return {"x": x}, y


def _fn_quadratic(rng, n, p):
    """y = a * x^2 + b * x + c"""
    lo, hi = p.get("x_min", -5.0), p.get("x_max", 5.0)
    x = rng.uniform(lo, hi, n)
    a = p.get("a", 1.0)
    b = p.get("b", 0.0)
    c = p.get("c", 0.0)
    y = a * x ** 2 + b * x + c
    return {"x": x}, y


def _fn_cubic(rng, n, p):
    """y = a * x^3"""
    lo, hi = p.get("x_min", -3.0), p.get("x_max", 3.0)
    x = rng.uniform(lo, hi, n)
    a = p.get("a", 1.0)
    y = a * x ** 3
    return {"x": x}, y


def _fn_sqrt_fn(rng, n, p):
    """y = a * sqrt(x)"""
    lo, hi = p.get("x_min", 0.5), p.get("x_max", 25.0)
    x = rng.uniform(lo, hi, n)
    a = p.get("a", 1.0)
    y = a * np.sqrt(x)
    return {"x": x}, y


def _fn_log_fn(rng, n, p):
    """y = a * log(x) + b"""
    lo, hi = p.get("x_min", 1.0), p.get("x_max", 100.0)
    x = rng.uniform(lo, hi, n)
    a = p.get("a", 1.0)
    b = p.get("b", 0.0)
    y = a * np.log(x) + b
    return {"x": x}, y


def _fn_reciprocal_fn(rng, n, p):
    """y = a / x"""
    lo, hi = p.get("x_min", 1.0), p.get("x_max", 20.0)
    x = rng.uniform(lo, hi, n)
    a = p.get("a", 10.0)
    y = a / x
    return {"x": x}, y


def _fn_sinusoidal(rng, n, p):
    """y = amplitude * sin(2*pi*x / period) + offset"""
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 20.0)
    x = rng.uniform(lo, hi, n)
    amp = p.get("amplitude", 1.0)
    period = p.get("period", 7.0)
    offset = p.get("offset", 0.0)
    y = amp * np.sin(2 * np.pi * x / period) + offset
    return {"x": x}, y


def _fn_sinusoidal_sum(rng, n, p):
    """y = sin(x) + 0.5 * sin(3x)"""
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 4 * np.pi)
    x = rng.uniform(lo, hi, n)
    a1 = p.get("a1", 1.0)
    f1 = p.get("f1", 1.0)
    a2 = p.get("a2", 0.5)
    f2 = p.get("f2", 3.0)
    y = a1 * np.sin(f1 * x) + a2 * np.sin(f2 * x)
    return {"x": x}, y


def _fn_almost_linear(rng, n, p):
    """y = slope * x + amp * sin(x) — looks nearly linear"""
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 10.0)
    x = rng.uniform(lo, hi, n)
    slope = p.get("slope", 1.0)
    amp = p.get("amplitude", 0.5)
    y = slope * x + amp * np.sin(x)
    return {"x": x}, y


def _fn_piecewise_linear(rng, n, p):
    """
    Two-slope piecewise:
        y = slope1 * x + c1    if x < threshold
        y = slope2 * x + c2   otherwise
    c2 is chosen so the function is continuous at threshold.
    """
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 20.0)
    x = rng.uniform(lo, hi, n)
    thr = p.get("threshold", 10.0)
    s1 = p.get("slope1", 2.0)
    c1 = p.get("c1", 0.0)
    s2 = p.get("slope2", 0.5)
    # Ensure continuity at threshold
    c2 = (s1 - s2) * thr + c1
    y = np.where(x < thr, s1 * x + c1, s2 * x + c2)
    return {"x": x}, y


def _fn_piecewise_flat(rng, n, p):
    """
    y = slope * x  if x < threshold, else constant
    constant chosen so the function is continuous at threshold.
    """
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 10.0)
    x = rng.uniform(lo, hi, n)
    thr = p.get("threshold", 5.0)
    slope = p.get("slope", 2.0)
    constant = slope * thr
    y = np.where(x < thr, slope * x, constant)
    return {"x": x}, y


def _fn_boss_piecewise(rng, n, p):
    """
    Three-segment piecewise (boss puzzle):
        y = s1*x + c1            if x < t1
        y = s2*x + c2            if t1 <= x < t2
        y = s3*x + c3            otherwise
    Continuity enforced at t1 and t2.
    """
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 15.0)
    x = rng.uniform(lo, hi, n)
    t1, t2 = p.get("t1", 3.0), p.get("t2", 7.0)
    s1, c1 = p.get("slope1", 2.0), p.get("c1", 0.0)
    s2 = p.get("slope2", 1.0)
    c2 = (s1 - s2) * t1 + c1
    s3 = p.get("slope3", 0.5)
    c3 = (s2 - s3) * t2 + c2
    y = np.where(x < t1, s1 * x + c1,
        np.where(x < t2, s2 * x + c2, s3 * x + c3))
    return {"x": x}, y


def _fn_product_2d(rng, n, p):
    """y = x1 * x2"""
    lo, hi = p.get("x_min", 1.0), p.get("x_max", 10.0)
    x1 = rng.uniform(lo, hi, n)
    x2 = rng.uniform(lo, hi, n)
    y = x1 * x2
    return {"x1": x1, "x2": x2}, y


def _fn_ratio_2d(rng, n, p):
    """y = x1 / x2  (x2 kept positive and away from zero)"""
    lo, hi = p.get("x_min", 1.0), p.get("x_max", 10.0)
    x1 = rng.uniform(lo, hi, n)
    x2 = rng.uniform(lo, hi, n)
    y = x1 / x2
    return {"x1": x1, "x2": x2}, y


def _fn_distance_2d(rng, n, p):
    """y = sqrt(x1^2 + x2^2) — distance from origin"""
    lo, hi = p.get("x_min", -5.0), p.get("x_max", 5.0)
    x1 = rng.uniform(lo, hi, n)
    x2 = rng.uniform(lo, hi, n)
    y = np.sqrt(x1 ** 2 + x2 ** 2)
    return {"x1": x1, "x2": x2}, y


def _fn_circle_classify(rng, n, p):
    """y = 1 if x1^2 + x2^2 < r^2 else 0"""
    lo, hi = p.get("x_min", -6.0), p.get("x_max", 6.0)
    x1 = rng.uniform(lo, hi, n)
    x2 = rng.uniform(lo, hi, n)
    r = p.get("radius", 4.0)
    y = (x1 ** 2 + x2 ** 2 < r ** 2).astype(float)
    return {"x1": x1, "x2": x2}, y


def _fn_staircase(rng, n, p):
    """y = floor(x / step) * step"""
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 50.0)
    x = rng.uniform(lo, hi, n)
    step = p.get("step", 10.0)
    y = np.floor(x / step) * step
    return {"x": x}, y


def _fn_linear_distractor(rng, n, p):
    """
    y = slope * x1 + intercept
    but the player also sees x2, x3, x4 (random noise features).
    """
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 10.0)
    x1 = rng.uniform(lo, hi, n)
    x2 = rng.uniform(0, 10, n)
    x3 = rng.uniform(0, 10, n)
    x4 = rng.uniform(0, 10, n)
    slope = p.get("slope", 3.0)
    intercept = p.get("intercept", 5.0)
    y = slope * x1 + intercept
    return {"x1": x1, "x2": x2, "x3": x3, "x4": x4}, y


def _fn_multi_transform(rng, n, p):
    """y = a * sin(x1) + b * x2^2 + c * x3"""
    x1 = rng.uniform(0.0, 2 * np.pi, n)
    x2 = rng.uniform(-3.0, 3.0, n)
    x3 = rng.uniform(0.0, 5.0, n)
    a = p.get("a", 1.0)
    b = p.get("b", 1.0)
    c = p.get("c", 3.0)
    y = a * np.sin(x1) + b * x2 ** 2 + c * x3
    return {"x1": x1, "x2": x2, "x3": x3}, y


def _fn_boss_multi(rng, n, p):
    """y = x1 * x2 + c * x3"""
    lo, hi = p.get("x_min", 1.0), p.get("x_max", 8.0)
    x1 = rng.uniform(lo, hi, n)
    x2 = rng.uniform(lo, hi, n)
    x3 = rng.uniform(0.0, 5.0, n)
    c = p.get("c", 2.0)
    y = x1 * x2 + c * x3
    return {"x1": x1, "x2": x2, "x3": x3}, y


# ---------------------------------------------------------------------------
# Registry mapping FunctionSpec.type → generator callable
# ---------------------------------------------------------------------------

_FUNCTION_REGISTRY: dict[str, Any] = {
    "linear":           _fn_linear,
    "quadratic":        _fn_quadratic,
    "cubic":            _fn_cubic,
    "sqrt_fn":          _fn_sqrt_fn,
    "log_fn":           _fn_log_fn,
    "reciprocal_fn":    _fn_reciprocal_fn,
    "sinusoidal":       _fn_sinusoidal,
    "sinusoidal_sum":   _fn_sinusoidal_sum,
    "almost_linear":    _fn_almost_linear,
    "piecewise_linear": _fn_piecewise_linear,
    "piecewise_flat":   _fn_piecewise_flat,
    "boss_piecewise":   _fn_boss_piecewise,
    "product_2d":       _fn_product_2d,
    "ratio_2d":         _fn_ratio_2d,
    "distance_2d":      _fn_distance_2d,
    "circle_classify":  _fn_circle_classify,
    "staircase":        _fn_staircase,
    "linear_distractor":"_fn_linear_distractor",  # resolved below
    "multi_transform":  _fn_multi_transform,
    "boss_multi":       _fn_boss_multi,
}
# Patch string that slipped through above
_FUNCTION_REGISTRY["linear_distractor"] = _fn_linear_distractor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(
    puzzle: Puzzle,
    n_samples: int = 100,
    seed: int = 42,
) -> dict:
    """
    Generate a reproducible dataset for a given puzzle.

    Parameters
    ----------
    puzzle : Puzzle
        The puzzle object whose ``function`` field describes the relationship.
    n_samples : int
        Number of rows to generate.
    seed : int
        Random seed for full reproducibility.

    Returns
    -------
    dict with keys:
        ``"X"`` — pandas DataFrame of input features (n_samples × n_features)
        ``"y"`` — numpy array of output values (n_samples,)

    Raises
    ------
    ValueError
        If the puzzle's function type is not in the registry.
    """
    fn_type = puzzle.function.type
    if fn_type not in _FUNCTION_REGISTRY:
        raise ValueError(
            f"Unknown function type '{fn_type}'. "
            f"Available: {sorted(_FUNCTION_REGISTRY.keys())}"
        )

    rng = np.random.default_rng(seed)
    generator = _FUNCTION_REGISTRY[fn_type]
    feature_dict, y = generator(rng, n_samples, puzzle.function.parameters)

    # Optional additive Gaussian noise
    if puzzle.noise_std > 0.0:
        y = y + rng.normal(0.0, puzzle.noise_std, n_samples)

    X = pd.DataFrame(feature_dict)
    return {"X": X, "y": np.array(y, dtype=float)}
