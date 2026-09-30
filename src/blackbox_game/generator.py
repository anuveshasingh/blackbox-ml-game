"""
generator.py — Reproducible dataset generation for every puzzle.

Usage
-----
    from blackbox_game.generator import generate_dataset
    from blackbox_game.puzzles import get_puzzle

    puzzle  = get_puzzle("bend_in_road")
    dataset = generate_dataset(puzzle, n_samples=100, seed=42)
    X, y    = dataset["X"], dataset["y"]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from .models import Puzzle, FunctionSpec


# ---------------------------------------------------------------------------
# Internal generators  (rng, n_samples, params) → (feature_dict, y_array)
# ---------------------------------------------------------------------------

def _fn_linear(rng, n, p):
    lo, hi = p.get("x_min", -5.0), p.get("x_max", 5.0)
    x = rng.uniform(lo, hi, n)
    y = p.get("slope", 1.0) * x + p.get("intercept", 0.0)
    return {"x": x}, y


def _fn_quadratic(rng, n, p):
    lo, hi = p.get("x_min", -5.0), p.get("x_max", 5.0)
    x = rng.uniform(lo, hi, n)
    y = p.get("a", 1.0) * x**2 + p.get("b", 0.0) * x + p.get("c", 0.0)
    return {"x": x}, y


def _fn_cubic(rng, n, p):
    lo, hi = p.get("x_min", -3.0), p.get("x_max", 3.0)
    x = rng.uniform(lo, hi, n)
    y = p.get("a", 1.0) * x**3
    return {"x": x}, y


def _fn_sqrt_fn(rng, n, p):
    lo, hi = p.get("x_min", 0.5), p.get("x_max", 25.0)
    x = rng.uniform(lo, hi, n)
    y = p.get("a", 1.0) * np.sqrt(x)
    return {"x": x}, y


def _fn_log_fn(rng, n, p):
    lo, hi = p.get("x_min", 1.0), p.get("x_max", 100.0)
    x = rng.uniform(lo, hi, n)
    y = p.get("a", 1.0) * np.log(x) + p.get("b", 0.0)
    return {"x": x}, y


def _fn_reciprocal_fn(rng, n, p):
    lo, hi = p.get("x_min", 1.0), p.get("x_max", 20.0)
    x = rng.uniform(lo, hi, n)
    y = p.get("a", 10.0) / x
    return {"x": x}, y


def _fn_abs_fn(rng, n, p):
    """y = a * |x|"""
    lo, hi = p.get("x_min", -5.0), p.get("x_max", 5.0)
    x = rng.uniform(lo, hi, n)
    y = p.get("a", 1.0) * np.abs(x)
    return {"x": x}, y


def _fn_cosine_fn(rng, n, p):
    """y = A * cos(freq * x)"""
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 4 * np.pi)
    x = rng.uniform(lo, hi, n)
    A = p.get("amplitude", 1.0)
    freq = p.get("freq", 1.0)
    y = A * np.cos(freq * x)
    return {"x": x}, y


def _fn_polynomial2_fn(rng, n, p):
    """y = a*x² + b*x  (needs BOTH square:x and identity:x features)"""
    lo, hi = p.get("x_min", -5.0), p.get("x_max", 5.0)
    x = rng.uniform(lo, hi, n)
    a = p.get("a", 1.0)
    b = p.get("b", -3.0)
    y = a * x**2 + b * x
    return {"x": x}, y


def _fn_phase_sum(rng, n, p):
    """y = sin(x) + cos(x) = sqrt(2) * sin(x + pi/4)"""
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 4 * np.pi)
    x = rng.uniform(lo, hi, n)
    y = np.sin(x) + np.cos(x)
    return {"x": x}, y


def _fn_sinusoidal(rng, n, p):
    """y = amplitude * sin(2π x / period) + offset"""
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 20.0)
    x = rng.uniform(lo, hi, n)
    amp    = p.get("amplitude", 1.0)
    period = p.get("period", 7.0)
    offset = p.get("offset", 0.0)
    y = amp * np.sin(2 * np.pi * x / period) + offset
    return {"x": x}, y


def _fn_sinusoidal_sum(rng, n, p):
    """y = a1*sin(f1*x) + a2*sin(f2*x)"""
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 4 * np.pi)
    x = rng.uniform(lo, hi, n)
    y = (p.get("a1", 1.0) * np.sin(p.get("f1", 1.0) * x) +
         p.get("a2", 0.5) * np.sin(p.get("f2", 3.0) * x))
    return {"x": x}, y


def _fn_almost_linear(rng, n, p):
    """y = slope*x + amplitude*sin(x)"""
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 10.0)
    x = rng.uniform(lo, hi, n)
    y = p.get("slope", 1.0) * x + p.get("amplitude", 0.5) * np.sin(x)
    return {"x": x}, y


def _fn_piecewise_flat(rng, n, p):
    """y = slope*x  if x < threshold, else constant"""
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 10.0)
    x   = rng.uniform(lo, hi, n)
    thr = p.get("threshold", 5.0)
    slp = p.get("slope", 2.0)
    y   = np.where(x < thr, slp * x, slp * thr)
    return {"x": x}, y


def _fn_boss_piecewise(rng, n, p):
    """Three-segment piecewise with continuity at t1 and t2."""
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 15.0)
    x  = rng.uniform(lo, hi, n)
    t1, t2 = p.get("t1", 3.0), p.get("t2", 7.0)
    s1, c1 = p.get("slope1", 2.0), p.get("c1", 0.0)
    s2     = p.get("slope2", 1.0)
    c2     = (s1 - s2) * t1 + c1
    s3     = p.get("slope3", 0.5)
    c3     = (s2 - s3) * t2 + c2
    y = np.where(x < t1, s1 * x + c1,
        np.where(x < t2, s2 * x + c2, s3 * x + c3))
    return {"x": x}, y


def _fn_product_2d(rng, n, p):
    """y = x1 * x2"""
    lo, hi = p.get("x_min", 1.0), p.get("x_max", 10.0)
    x1 = rng.uniform(lo, hi, n)
    x2 = rng.uniform(lo, hi, n)
    return {"x1": x1, "x2": x2}, x1 * x2


def _fn_ratio_2d(rng, n, p):
    """y = x1 / x2  (x2 safely positive)"""
    lo, hi = p.get("x_min", 1.0), p.get("x_max", 10.0)
    x1 = rng.uniform(lo, hi, n)
    x2 = rng.uniform(lo, hi, n)
    return {"x1": x1, "x2": x2}, x1 / x2


def _fn_distance_2d(rng, n, p):
    """y = sqrt(x1² + x2²)"""
    lo, hi = p.get("x_min", -5.0), p.get("x_max", 5.0)
    x1 = rng.uniform(lo, hi, n)
    x2 = rng.uniform(lo, hi, n)
    return {"x1": x1, "x2": x2}, np.sqrt(x1**2 + x2**2)


def _fn_circle_classify(rng, n, p):
    """y = 1 if x1² + x2² < r² else 0"""
    lo, hi = p.get("x_min", -6.0), p.get("x_max", 6.0)
    x1 = rng.uniform(lo, hi, n)
    x2 = rng.uniform(lo, hi, n)
    r  = p.get("radius", 4.0)
    return {"x1": x1, "x2": x2}, (x1**2 + x2**2 < r**2).astype(float)


def _fn_linear_distractor(rng, n, p):
    """y = slope*x1 + intercept, with x2/x3/x4 as random distractors."""
    lo, hi = p.get("x_min", -5.0), p.get("x_max", 5.0)
    x1 = rng.uniform(lo, hi, n)
    x2 = rng.uniform(-5, 5, n)
    x3 = rng.uniform(-5, 5, n)
    x4 = rng.uniform(-5, 5, n)
    y  = p.get("slope", 3.0) * x1 + p.get("intercept", 0.0)
    return {"x1": x1, "x2": x2, "x3": x3, "x4": x4}, y


def _fn_multi_transform(rng, n, p):
    """y = a*sin(x1) + b*x2² + c*x3"""
    x1 = rng.uniform(0.0, 2 * np.pi, n)
    x2 = rng.uniform(-3.0, 3.0, n)
    x3 = rng.uniform(0.0, 5.0, n)
    y  = (p.get("a", 1.0) * np.sin(x1) +
          p.get("b", 1.0) * x2**2 +
          p.get("c", 3.0) * x3)
    return {"x1": x1, "x2": x2, "x3": x3}, y


def _fn_boss_multi(rng, n, p):
    """y = x1*x2 + c*x3"""
    lo, hi = p.get("x_min", 1.0), p.get("x_max", 8.0)
    x1 = rng.uniform(lo, hi, n)
    x2 = rng.uniform(lo, hi, n)
    x3 = rng.uniform(0.0, 5.0, n)
    return {"x1": x1, "x2": x2, "x3": x3}, x1 * x2 + p.get("c", 2.0) * x3


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_FUNCTION_REGISTRY: dict[str, Any] = {
    "linear":           _fn_linear,
    "quadratic":        _fn_quadratic,
    "cubic":            _fn_cubic,
    "sqrt_fn":          _fn_sqrt_fn,
    "log_fn":           _fn_log_fn,
    "reciprocal_fn":    _fn_reciprocal_fn,
    "abs_fn":           _fn_abs_fn,
    "cosine_fn":        _fn_cosine_fn,
    "polynomial2_fn":   _fn_polynomial2_fn,
    "phase_sum":        _fn_phase_sum,
    "sinusoidal":       _fn_sinusoidal,
    "sinusoidal_sum":   _fn_sinusoidal_sum,
    "almost_linear":    _fn_almost_linear,
    "piecewise_flat":   _fn_piecewise_flat,
    "boss_piecewise":   _fn_boss_piecewise,
    "product_2d":       _fn_product_2d,
    "ratio_2d":         _fn_ratio_2d,
    "distance_2d":      _fn_distance_2d,
    "circle_classify":  _fn_circle_classify,
    "linear_distractor":"_fn_linear_distractor",  # resolved below
    "multi_transform":  _fn_multi_transform,
    "boss_multi":       _fn_boss_multi,
}
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
    puzzle    : Puzzle object
    n_samples : number of rows (default 100)
    seed      : random seed — same seed always gives same data

    Returns
    -------
    dict
        ``"X"`` — pandas DataFrame (n_samples × n_features)
        ``"y"`` — numpy array (n_samples,)
    """
    fn_type = puzzle.function.type
    if fn_type not in _FUNCTION_REGISTRY:
        raise ValueError(
            f"Unknown function type '{fn_type}'. "
            f"Available: {sorted(_FUNCTION_REGISTRY.keys())}"
        )

    rng = np.random.default_rng(seed)
    feature_dict, y = _FUNCTION_REGISTRY[fn_type](
        rng, n_samples, puzzle.function.parameters
    )

    if puzzle.noise_std > 0.0:
        y = y + rng.normal(0.0, puzzle.noise_std, n_samples)

    return {"X": pd.DataFrame(feature_dict), "y": np.array(y, dtype=float)}
