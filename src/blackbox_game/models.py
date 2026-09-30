"""
models.py — Pydantic-style dataclasses for the puzzle schema.

These models define the structure of a Puzzle. They are plain Python dataclasses
so the entire game engine has zero mandatory framework dependencies beyond
scikit-learn and numpy.

JSON serialization is handled via `dataclasses.asdict()` plus a small helper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# FunctionSpec — a serialisable description of a black-box function
# ---------------------------------------------------------------------------

@dataclass
class FunctionSpec:
    """
    Describes the hidden function inside a puzzle without storing a raw lambda.

    The ``type`` field is a string key that ``generator.py`` maps to an actual
    numpy expression.  ``parameters`` carries any numeric constants the
    expression needs.

    Supported types (see generator.py for full list):
        linear, quadratic, cubic, sqrt_fn, log_fn, reciprocal_fn,
        sinusoidal, sinusoidal_sum, piecewise_linear, product_2d,
        ratio_2d, distance_2d, circle_classify, staircase,
        linear_distractor, multi_transform, boss_piecewise, boss_multi
    """
    type: str
    parameters: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Puzzle — the central dataclass
# ---------------------------------------------------------------------------

@dataclass
class Puzzle:
    """
    A single game puzzle.

    Attributes
    ----------
    id : str
        Unique identifier used in API calls.
    title : str
        Human-readable title shown to the player.
    category : str
        Broad concept category (e.g. "linear", "feature_transform").
    difficulty : int
        1 = beginner, 2 = intermediate, 3 = challenge / boss.
    description : str
        Player-facing description of the scenario.
    input_features : list[str]
        Names of the raw input features (e.g. ["x"] or ["x1", "x2"]).
    function : FunctionSpec
        Serialisable description of the hidden relationship.
    noise_std : float
        Standard deviation of Gaussian noise added during data generation.
        Use 0.0 for beginner puzzles.
    allowed_transforms : list[str]
        Transformation keys the player may apply (see transforms.py).
    allowed_binary_transforms : list[str]
        Binary (two-feature) transformation keys the player may use.
    intended_model : str
        "linear_regression" or "decision_tree".
    solution_features : list[str]
        The canonical correct feature(s). Used for scoring.
    hints : list[str]
        Progressive hints, from vague to specific.
    explanation : str
        Post-solve educational explanation (≤ ~100 words).
    real_world_connection : str
        One sentence relating the puzzle to a real-world scenario.
    """
    id: str
    title: str
    category: str
    difficulty: int
    description: str
    input_features: list[str]
    function: FunctionSpec
    noise_std: float
    allowed_transforms: list[str]
    allowed_binary_transforms: list[str]
    intended_model: str
    solution_features: list[str]
    hints: list[str]
    explanation: str
    real_world_connection: str

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain dict (all values JSON-serialisable)."""
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "Puzzle":
        """Reconstruct a Puzzle from a plain dict (e.g. loaded from JSON)."""
        fn_data = data.pop("function", {})
        function = FunctionSpec(
            type=fn_data["type"],
            parameters=fn_data.get("parameters", {}),
        )
        return cls(function=function, **data)

    @classmethod
    def from_json(cls, json_str: str) -> "Puzzle":
        """Reconstruct a Puzzle from a JSON string."""
        return cls.from_dict(json.loads(json_str))
