"""
transforms.py — Safe, vectorised feature transformations.

All transforms operate on 1-D numpy arrays and handle invalid domains
gracefully (returning NaN rather than crashing). The evaluator will
reject features containing NaN values.

Usage
-----
    from blackbox_game.transforms import apply_transform, TRANSFORM_REGISTRY

    x = np.array([1.0, 2.0, 3.0, 4.0])
    x_sq = apply_transform("square", x)          # [1, 4, 9, 16]
    x_log = apply_transform("log", x)            # [0, 0.693, ...]

    # Binary (two-feature) transforms
    result = apply_binary_transform("multiply", x, x)  # x * x
"""

from __future__ import annotations

import numpy as np
from typing import Callable

# ---------------------------------------------------------------------------
# Unary transforms
# ---------------------------------------------------------------------------

def _identity(x: np.ndarray) -> np.ndarray:
    return x.copy()

def _square(x: np.ndarray) -> np.ndarray:
    return x ** 2

def _cube(x: np.ndarray) -> np.ndarray:
    return x ** 3

def _sqrt(x: np.ndarray) -> np.ndarray:
    """Returns NaN for negative values — caller decides what to do."""
    return np.where(x >= 0, np.sqrt(np.maximum(x, 0.0)), np.nan)

def _abs(x: np.ndarray) -> np.ndarray:
    return np.abs(x)

def _log(x: np.ndarray) -> np.ndarray:
    """Natural log. Returns NaN for x <= 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, np.log(x), np.nan)

def _log2(x: np.ndarray) -> np.ndarray:
    """Base-2 log. Returns NaN for x <= 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, np.log2(x), np.nan)

def _reciprocal(x: np.ndarray) -> np.ndarray:
    """1/x. Returns NaN where |x| < 1e-10."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x) > 1e-10, 1.0 / x, np.nan)

def _sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x)

def _cos(x: np.ndarray) -> np.ndarray:
    return np.cos(x)

def _sin_2pi(x: np.ndarray) -> np.ndarray:
    """sin(2π x) — period 1."""
    return np.sin(2 * np.pi * x)

def _cos_2pi(x: np.ndarray) -> np.ndarray:
    """cos(2π x) — period 1."""
    return np.cos(2 * np.pi * x)

def _sin_period7(x: np.ndarray) -> np.ndarray:
    """sin(2π x / 7) — captures weekly periodicity."""
    return np.sin(2 * np.pi * x / 7.0)

def _cos_period7(x: np.ndarray) -> np.ndarray:
    """cos(2π x / 7) — captures weekly periodicity."""
    return np.cos(2 * np.pi * x / 7.0)

def _exp(x: np.ndarray) -> np.ndarray:
    """Exponential. Clips large inputs to avoid overflow."""
    return np.exp(np.clip(x, -100, 100))

def _floor10(x: np.ndarray) -> np.ndarray:
    """floor(x / 10) * 10 — creates a staircase with step size 10."""
    return np.floor(x / 10.0) * 10.0

def _step(x: np.ndarray) -> np.ndarray:
    """Binary step: 1 if x >= 0, else 0."""
    return (x >= 0).astype(float)

# ---------------------------------------------------------------------------
# Public unary transform registry
# ---------------------------------------------------------------------------

#: Maps string key → callable transform.
TRANSFORM_REGISTRY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity":     _identity,
    "square":       _square,
    "cube":         _cube,
    "sqrt":         _sqrt,
    "abs":          _abs,
    "log":          _log,
    "log2":         _log2,
    "reciprocal":   _reciprocal,
    "sin":          _sin,
    "cos":          _cos,
    "sin_2pi":      _sin_2pi,
    "cos_2pi":      _cos_2pi,
    "sin_period7":  _sin_period7,
    "cos_period7":  _cos_period7,
    "exp":          _exp,
    "floor10":      _floor10,
    "step":         _step,
}

# Human-readable description for each transform (shown to the player).
TRANSFORM_DESCRIPTIONS: dict[str, str] = {
    "identity":     "x  (no change)",
    "square":       "x²",
    "cube":         "x³",
    "sqrt":         "√x",
    "abs":          "|x|",
    "log":          "log(x)",
    "log2":         "log₂(x)",
    "reciprocal":   "1/x",
    "sin":          "sin(x)",
    "cos":          "cos(x)",
    "sin_2pi":      "sin(2π·x)",
    "cos_2pi":      "cos(2π·x)",
    "sin_period7":  "sin(2π·x/7)",
    "cos_period7":  "cos(2π·x/7)",
    "exp":          "eˣ",
    "floor10":      "floor(x/10)·10",
    "step":         "step(x)  [1 if x≥0 else 0]",
}


def apply_transform(key: str, x: np.ndarray) -> np.ndarray:
    """
    Apply a named unary transform to a 1-D numpy array.

    Parameters
    ----------
    key : str
        One of the keys in ``TRANSFORM_REGISTRY``.
    x : np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        Transformed values (may contain NaN for invalid inputs).

    Raises
    ------
    ValueError
        If ``key`` is not recognised.
    """
    if key not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform '{key}'. "
            f"Available: {sorted(TRANSFORM_REGISTRY.keys())}"
        )
    return TRANSFORM_REGISTRY[key](np.asarray(x, dtype=float))


# ---------------------------------------------------------------------------
# Binary transforms
# ---------------------------------------------------------------------------

def _multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b

def _divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > 1e-10, a / b, np.nan)

def _add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b

def _subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b

def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance from origin: sqrt(a² + b²)."""
    return np.sqrt(a ** 2 + b ** 2)

BINARY_TRANSFORM_REGISTRY: dict[
    str, Callable[[np.ndarray, np.ndarray], np.ndarray]
] = {
    "multiply":  _multiply,
    "divide":    _divide,
    "add":       _add,
    "subtract":  _subtract,
    "distance":  _distance,
}

BINARY_TRANSFORM_DESCRIPTIONS: dict[str, str] = {
    "multiply":  "x1 × x2",
    "divide":    "x1 / x2",
    "add":       "x1 + x2",
    "subtract":  "x1 − x2",
    "distance":  "√(x1² + x2²)",
}


def apply_binary_transform(
    key: str,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Apply a named binary transform to two 1-D numpy arrays.

    Parameters
    ----------
    key : str
        One of the keys in ``BINARY_TRANSFORM_REGISTRY``.
    a, b : np.ndarray
        Input arrays (same length).

    Returns
    -------
    np.ndarray
        Result array (may contain NaN for invalid inputs).

    Raises
    ------
    ValueError
        If ``key`` is not recognised.
    """
    if key not in BINARY_TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown binary transform '{key}'. "
            f"Available: {sorted(BINARY_TRANSFORM_REGISTRY.keys())}"
        )
    return BINARY_TRANSFORM_REGISTRY[key](
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
    )


def list_transforms() -> list[dict]:
    """
    Return all available unary transforms with their descriptions.

    Useful for the frontend to display the transform palette to the player.
    """
    return [
        {"key": k, "description": TRANSFORM_DESCRIPTIONS.get(k, k)}
        for k in TRANSFORM_REGISTRY
    ]


def list_binary_transforms() -> list[dict]:
    """Return all available binary transforms with their descriptions."""
    return [
        {"key": k, "description": BINARY_TRANSFORM_DESCRIPTIONS.get(k, k)}
        for k in BINARY_TRANSFORM_REGISTRY
    ]
