"""
__init__.py — Public API for blackbox_game.

Import everything a frontend developer needs from this single module:

    from blackbox_game import (
        get_puzzle,
        list_puzzles,
        generate_dataset,
        apply_transform,
        apply_binary_transform,
        evaluate_submission,
        compare_models,
        get_hint,
        get_explanation,
        list_transforms,
        list_binary_transforms,
        PlayerSession,
    )

The internal modules (evaluator, scoring, generator, transforms, puzzles,
hints, models) are importable directly for advanced usage but are not
required for normal game operation.
"""

from __future__ import annotations

from typing import Union
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Re-export the clean public surface
# ---------------------------------------------------------------------------

from .puzzles import get_puzzle, list_puzzles, PUZZLE_REGISTRY
from .generator import generate_dataset
from .transforms import (
    apply_transform,
    apply_binary_transform,
    list_transforms,
    list_binary_transforms,
)
from .evaluator import (
    build_feature_matrix,
    evaluate_linear_regression,
    evaluate_decision_tree,
    compare_models as _compare_models_raw,
)
from .scoring import (
    PlayerSession,
    compute_score,
    CORRECT_R2_THRESHOLD,
    CORRECT_ACCURACY_THRESHOLD,
)
from .hints import get_hint, get_explanation, get_all_hints
from .models import Puzzle, FunctionSpec

# Keep a simple in-memory session store so callers don't need to manage
# sessions themselves (stateless web backends should manage sessions externally).
_SESSIONS: dict[str, PlayerSession] = {}


# ---------------------------------------------------------------------------
# High-level submission API
# ---------------------------------------------------------------------------

def evaluate_submission(
    puzzle_id: str,
    model: str,
    features: list[Union[str, dict]],
    session: PlayerSession | None = None,
    n_samples: int = 200,
    seed: int = 42,
) -> dict:
    """
    Evaluate a player's attempt at solving a puzzle.

    This is the **primary entry point** for the frontend.

    Parameters
    ----------
    puzzle_id : str
        The puzzle to attempt (e.g. ``"square_01"``).
    model : str
        ``"linear_regression"`` or ``"decision_tree"``.
    features : list
        Feature specifications.  Each element can be:
        - a plain string matching a raw feature name (e.g. ``"x"``)
        - a transform shorthand string (e.g. ``"square:x"``)
        - a binary transform dict (e.g. ``{"binary": "multiply", "a": "x1", "b": "x2"}``)
    session : PlayerSession | None
        Pass an existing session to track attempts/hints/score across
        multiple submissions for the same puzzle.
        If None, a fresh session is created (score starts at 0).
    n_samples : int
        Dataset size used for evaluation (default 200).
    seed : int
        Random seed — keep at 42 for reproducible results.

    Returns
    -------
    dict with keys:
        puzzle_id, model, is_correct, r2 (or accuracy), mse,
        score_result (full scoring breakdown), message, session

    Example
    -------
    >>> result = evaluate_submission(
    ...     puzzle_id="square_01",
    ...     model="linear_regression",
    ...     features=["square:x"],
    ... )
    >>> print(result["is_correct"])   # True
    >>> print(result["r2"])           # ~1.0
    """
    puzzle = get_puzzle(puzzle_id)

    if session is None:
        session = PlayerSession(puzzle_id=puzzle_id)

    # Generate dataset
    dataset = generate_dataset(puzzle, n_samples=n_samples, seed=seed)
    X_df: pd.DataFrame = dataset["X"]
    y: np.ndarray = dataset["y"]

    # Build design matrix
    try:
        X_feat = build_feature_matrix(X_df, features)
    except (ValueError, KeyError) as exc:
        return {
            "puzzle_id": puzzle_id,
            "error": str(exc),
            "is_correct": False,
            "message": f"Could not build features: {exc}",
            "session": session.to_dict(),
        }

    # Determine task type
    task = "classification" if puzzle.function.type == "circle_classify" else "regression"

    # Evaluate chosen model
    model_correct = model == puzzle.intended_model

    if model == "linear_regression":
        eval_result = evaluate_linear_regression(X_feat, y)
        fit_quality = eval_result.get("r2", -999.0)
    elif model == "decision_tree":
        eval_result = evaluate_decision_tree(X_feat, y, task=task)
        fit_quality = eval_result.get(
            "accuracy" if task == "classification" else "r2", -999.0
        )
    else:
        return {
            "puzzle_id": puzzle_id,
            "error": f"Unknown model '{model}'. Use 'linear_regression' or 'decision_tree'.",
            "is_correct": False,
            "message": "Unknown model.",
            "session": session.to_dict(),
        }

    # Score
    score_result = compute_score(
        session=session,
        model_correct=model_correct,
        fit_quality=fit_quality if np.isfinite(fit_quality) else -999.0,
        task=task,
    )

    response = {
        "puzzle_id": puzzle_id,
        "model": model,
        "features_used": features,
        "is_correct": score_result["is_correct"],
        "message": score_result["message"],
        "score_result": score_result,
        "session": session.to_dict(),
        **eval_result,
    }

    # Include explanation only when the puzzle is solved
    if score_result["is_correct"]:
        response["explanation"] = get_explanation(puzzle)

    return response


def compare_models(
    puzzle_id: str,
    features: list[Union[str, dict]],
    n_samples: int = 200,
    seed: int = 42,
) -> dict:
    """
    Compare linear regression vs decision tree on a set of features.

    Useful for teaching the player to understand why one model is better.

    Parameters
    ----------
    puzzle_id : str
        The puzzle whose data to use.
    features : list
        Feature specifications (same format as ``evaluate_submission``).
    n_samples : int
        Dataset size.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Side-by-side comparison with ``recommended_model`` and a
        human-readable ``summary``.
    """
    puzzle = get_puzzle(puzzle_id)
    dataset = generate_dataset(puzzle, n_samples=n_samples, seed=seed)
    X_df: pd.DataFrame = dataset["X"]
    y: np.ndarray = dataset["y"]

    X_feat = build_feature_matrix(X_df, features)
    task = "classification" if puzzle.function.type == "circle_classify" else "regression"

    return _compare_models_raw(X_feat, y, task=task)


__all__ = [
    # Puzzle catalogue
    "get_puzzle",
    "list_puzzles",
    "PUZZLE_REGISTRY",
    # Data generation
    "generate_dataset",
    # Transforms
    "apply_transform",
    "apply_binary_transform",
    "list_transforms",
    "list_binary_transforms",
    # Evaluation
    "evaluate_submission",
    "compare_models",
    "build_feature_matrix",
    "evaluate_linear_regression",
    "evaluate_decision_tree",
    # Scoring
    "PlayerSession",
    "compute_score",
    # Hints & explanations
    "get_hint",
    "get_explanation",
    # Models
    "Puzzle",
    "FunctionSpec",
]
