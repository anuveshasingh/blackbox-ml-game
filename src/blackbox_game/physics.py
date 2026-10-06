"""
physics.py — Physical constants and exact formulas for the Physics round.

Physics puzzles are deterministic: their outputs are exact function values.

Each formula takes a mapping of input column name → 1-D array and returns
the output array. Generator and point-evaluation code both call
``evaluate_formula`` so they can never disagree. Fixed values from a puzzle
are merged into the mapping before the formula is called.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Standard values, fixed and never input columns.
G_ACCEL = 9.80665                 # m/s^2 — gravitational acceleration (g)
K_COULOMB = 8.9875517923e9        # N m^2 C^-2 — Coulomb's constant (k_e)


def _projectile_y(c):
    return c["v0"] * np.sin(c["theta"]) * c["t"] - 0.5 * G_ACCEL * c["t"] ** 2


def _shm_energy(c):
    return 0.5 * c["k"] * c["x"] ** 2 + 0.5 * c["m"] * c["v"] ** 2


def _wave(c):
    return c["A"] * np.sin(c["k"] * c["x"] - c["omega"] * c["t"] + c["phi"])


def _coulomb(count: int) -> Callable[[dict], np.ndarray]:
    def formula(c):
        return K_COULOMB * sum(c[f"q{i}"] / c[f"r{i}"] for i in range(1, count + 1))
    return formula


FORMULAS: dict[str, Callable[[dict], np.ndarray]] = {
    "projectile_y": _projectile_y,
    "shm_energy": _shm_energy,
    "travelling_wave": _wave,
    "coulomb_2": _coulomb(2),
}


def evaluate_formula(name: str, columns: dict) -> np.ndarray:
    """Return the exact output of physics formula ``name`` for ``columns``."""
    if name not in FORMULAS:
        raise ValueError(f"Unknown physics formula '{name}'.")
    return np.asarray(FORMULAS[name](columns), dtype=float)
