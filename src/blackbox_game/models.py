"""
models.py — Pydantic-style dataclasses for the puzzle schema.

These models define the structure of a Puzzle. They are plain Python
dataclasses so the engine has zero mandatory framework dependencies
beyond scikit-learn and numpy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class FunctionSpec:
    """
    Serialisable description of the hidden function inside a puzzle.

    ``type`` maps to an implementation in ``generator.py``.
    ``parameters`` carries numeric constants needed by that implementation.
    """
    type: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class Puzzle:
    """
    A single game puzzle.

    Attributes
    ----------
    id, title, category, difficulty : basic metadata
    description : player-facing scenario description
    input_features : names of raw input columns
    function : serialisable hidden function specification
    noise_std : Gaussian noise added during generation (0 for beginner)
    allowed_transforms : unary transform keys the player may use
    allowed_binary_transforms : binary transform keys the player may use
    intended_model : "linear_regression" or "decision_tree"
    solution_features : canonical correct features (used server-side only)
    explanation : post-solve educational text
    real_world_connection : one-sentence real-world link
    hints : optional progressive hints (empty list = disabled)
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
    solution_features: list[str | dict]
    explanation: str
    real_world_connection: str
    hints: list[str] = field(default_factory=list)  # disabled by default

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "Puzzle":
        fn_data = data.pop("function", {})
        function = FunctionSpec(
            type=fn_data["type"],
            parameters=fn_data.get("parameters", {}),
        )
        # hints is optional in older data
        data.setdefault("hints", [])
        return cls(function=function, **data)

    @classmethod
    def from_json(cls, json_str: str) -> "Puzzle":
        return cls.from_dict(json.loads(json_str))
