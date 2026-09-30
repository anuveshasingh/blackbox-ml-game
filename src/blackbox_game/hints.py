"""
hints.py — Hint retrieval and educational explanations.

Hints are stored directly in each Puzzle object.  This module provides
a clean API so the frontend never needs to access puzzle internals.

Usage
-----
    from blackbox_game.hints import get_hint, get_explanation

    hint_text = get_hint(puzzle, hint_index=0)
    explanation = get_explanation(puzzle)
"""

from __future__ import annotations

from .models import Puzzle


def get_hint(puzzle: Puzzle, hint_index: int) -> dict:
    """
    Return a single progressive hint for the given puzzle.

    Parameters
    ----------
    puzzle : Puzzle
        The puzzle whose hints to retrieve.
    hint_index : int
        0-based index of the hint to return.
        Hint 0 is the vaguest; the last hint is the most specific.

    Returns
    -------
    dict with keys:
        ``hint_index``   — the index returned
        ``total_hints``  — total number of hints available
        ``text``         — the hint text
        ``cost``         — score penalty for requesting this hint (-20)
    """
    hints = puzzle.hints
    n = len(hints)

    if hint_index < 0 or hint_index >= n:
        return {
            "hint_index": hint_index,
            "total_hints": n,
            "text": (
                f"No more hints available (only {n} hint(s) for this puzzle)."
            ),
            "cost": 0,
        }

    return {
        "hint_index": hint_index,
        "total_hints": n,
        "text": hints[hint_index],
        "cost": -20,
    }


def get_explanation(puzzle: Puzzle) -> dict:
    """
    Return the post-solve educational explanation for a puzzle.

    The frontend should display this only AFTER the player has solved
    the puzzle, so as not to spoil the answer.

    Returns
    -------
    dict with keys:
        ``explanation``           — the main explanation text
        ``real_world_connection`` — one-line real-world relevance
        ``solution_features``     — the canonical correct feature(s)
        ``intended_model``        — "linear_regression" or "decision_tree"
    """
    return {
        "explanation": puzzle.explanation,
        "real_world_connection": puzzle.real_world_connection,
        "solution_features": puzzle.solution_features,
        "intended_model": puzzle.intended_model,
    }


def get_all_hints(puzzle: Puzzle) -> list[dict]:
    """
    Return all hints for a puzzle (for debugging / admin use only).

    Do NOT expose this to the player during the game.
    """
    return [
        {
            "hint_index": i,
            "total_hints": len(puzzle.hints),
            "text": h,
            "cost": -20,
        }
        for i, h in enumerate(puzzle.hints)
    ]
