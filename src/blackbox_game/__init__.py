"""
__init__.py — Public API for blackbox_game.

    from blackbox_game import (
        get_puzzle, list_puzzles, generate_dataset,
        apply_transform, apply_binary_transform,
        evaluate_submission, compare_models,
        get_explanation, list_transforms, list_binary_transforms,
        PlayerSession,
    )
"""

from __future__ import annotations

from typing import Union

from .puzzles    import get_puzzle, list_puzzles, PUZZLE_REGISTRY
from .scoring    import (
    PlayerSession, compute_score, check_fuzzy_match,
    CORRECT_R2_THRESHOLD, CORRECT_ACCURACY_THRESHOLD,
    POWER_MAP,
)
from .hints      import get_explanation
from .models     import Puzzle, FunctionSpec


def __getattr__(name: str):
    """Load numerical and model modules only when their API is requested."""
    if name in {"generate_dataset", "evaluate_points"}:
        from . import generator
        value = getattr(generator, name)
    elif name in {
        "apply_transform", "apply_binary_transform",
        "list_transforms", "list_binary_transforms",
    }:
        from . import transforms
        value = getattr(transforms, name)
    elif name in {
        "build_feature_matrix", "evaluate_linear_regression",
        "evaluate_decision_tree", "predict_model",
    }:
        from . import evaluator
        value = getattr(evaluator, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


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

    This is the **primary entry point** for the frontend / CLI.

    Parameters
    ----------
    puzzle_id : str
        The puzzle to attempt (e.g. ``"square_01"``).
    model : str
        ``"linear_regression"`` or ``"decision_tree"``.
    features : list
        Feature specifications.  Each element can be:

        * a plain string column name  (e.g. ``"x"``)
        * a transform shorthand       (e.g. ``"square:x"``)
        * a binary transform dict     (e.g. ``{"binary":"multiply","a":"x1","b":"x2"}``)

    session : PlayerSession | None
        Existing session for multi-attempt tracking.  Pass None to auto-create.
    n_samples : int
        Dataset size used for evaluation (default 200).
    seed : int
        Random seed (default 42 — keep consistent for reproducibility).

    Returns
    -------
    dict
        puzzle_id, model, is_correct, is_fuzzy, r2/accuracy, mse,
        score_result, message, session, [explanation if correct]
    """
    import numpy as np
    import pandas as pd
    from .evaluator import (
        build_feature_matrix, evaluate_decision_tree,
        evaluate_linear_regression,
    )
    from .generator import generate_dataset

    puzzle = get_puzzle(puzzle_id)

    if session is None:
        session = PlayerSession(puzzle_id=puzzle_id)

    dataset = generate_dataset(puzzle, n_samples=n_samples, seed=seed)
    X_df: pd.DataFrame = dataset["X"]
    y: np.ndarray = dataset["y"]

    try:
        X_feat = build_feature_matrix(X_df, features)
    except (ValueError, KeyError) as exc:
        return {
            "puzzle_id": puzzle_id,
            "error": str(exc),
            "is_correct": False,
            "is_fuzzy": False,
            "message": f"Could not build features: {exc}",
            "session": session.to_dict(),
        }

    task = "classification" if puzzle.function.type == "circle_classify" else "regression"
    model_correct = (model == puzzle.intended_model)

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
            "is_fuzzy": False,
            "message": "Unknown model.",
            "session": session.to_dict(),
        }

    # Fuzzy matching (only when not already correct)
    threshold = (
        CORRECT_R2_THRESHOLD if task == "regression"
        else CORRECT_ACCURACY_THRESHOLD
    )
    fq = fit_quality if np.isfinite(fit_quality) else -999.0
    if fq >= threshold:
        fuzzy_info = None   # exact correct — skip fuzzy
    else:
        frac, desc = check_fuzzy_match(features, puzzle.solution_features)
        fuzzy_info = (frac, desc) if frac > 0 else None

    score_result = compute_score(
        session=session,
        model_correct=model_correct,
        fit_quality=fq,
        task=task,
        fuzzy_info=fuzzy_info,
    )

    response = {
        "puzzle_id":     puzzle_id,
        "model":         model,
        "features_used": features,
        "is_correct":    score_result["is_correct"],
        "is_fuzzy":      score_result["is_fuzzy"],
        "message":       score_result["message"],
        "score_result":  score_result,
        "session":       session.to_dict(),
        **eval_result,
    }

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
    Run both models on the same features and return a side-by-side comparison.

    Useful for teaching why one model is better than another.
    """
    from .evaluator import build_feature_matrix, compare_models as compare_models_raw
    from .generator import generate_dataset

    puzzle  = get_puzzle(puzzle_id)
    dataset = generate_dataset(puzzle, n_samples=n_samples, seed=seed)
    X_df    = dataset["X"]
    y       = dataset["y"]
    X_feat  = build_feature_matrix(X_df, features)
    task    = "classification" if puzzle.function.type == "circle_classify" else "regression"
    return compare_models_raw(X_feat, y, task=task)


__all__ = [
    "get_puzzle", "list_puzzles", "PUZZLE_REGISTRY",
    "generate_dataset",
    "apply_transform", "apply_binary_transform",
    "list_transforms", "list_binary_transforms",
    "evaluate_submission", "compare_models",
    "build_feature_matrix",
    "evaluate_linear_regression", "evaluate_decision_tree",
    "PlayerSession", "compute_score", "check_fuzzy_match",
    "get_explanation",
    "Puzzle", "FunctionSpec",
    "POWER_MAP",
]
