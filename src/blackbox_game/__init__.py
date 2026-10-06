"""
blackbox_game — engine for the Blackbox ML Game.

    from blackbox_game import get_puzzle, list_puzzles, generate_dataset
"""

from __future__ import annotations

from .models  import FunctionSpec, Puzzle
from .puzzles import PUZZLE_REGISTRY, get_puzzle, list_puzzles


def __getattr__(name: str):
    """Load numerical and model modules only when their API is requested."""
    if name in {"generate_dataset", "evaluate_points", "sample_count"}:
        from . import generator
        value = getattr(generator, name)
    elif name in {
        "apply_transform", "apply_binary_transform",
        "list_transforms", "list_binary_transforms",
    }:
        from . import transforms
        value = getattr(transforms, name)
    elif name in {"build_feature_matrix", "predict_model"}:
        from . import fitting
        value = getattr(fitting, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = [
    "get_puzzle", "list_puzzles", "PUZZLE_REGISTRY",
    "generate_dataset", "evaluate_points", "sample_count",
    "apply_transform", "apply_binary_transform",
    "list_transforms", "list_binary_transforms",
    "build_feature_matrix", "predict_model",
    "Puzzle", "FunctionSpec",
]
