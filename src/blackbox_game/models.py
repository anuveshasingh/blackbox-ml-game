"""
models.py — Dataclasses for the puzzle schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FunctionSpec:
    """
    The hidden function inside a puzzle.

    ``type`` selects the implementation: a numerical type in ``generator.py``,
    or ``"image"`` for ``images.py``. ``parameters`` carries its constants.
    """
    type: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class Puzzle:
    """
    One game puzzle.

    id             : internal name (never shown; players see puzzle_NN)
    difficulty     : 1 beginner, 2 physics, 3 image
    description    : text shown by ``show`` (may be empty)
    input_features : input column names (empty for image puzzles)
    function       : the hidden function
    """
    id: str
    difficulty: int
    description: str
    input_features: list[str]
    function: FunctionSpec
