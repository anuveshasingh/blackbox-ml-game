"""
hints.py — Explanation retrieval (hints disabled).

Hints have been removed from this version of the game.
Only post-solve explanations are available.
"""

from __future__ import annotations

from .models import Puzzle


def get_explanation(puzzle: Puzzle) -> dict:
    """
    Return the post-solve explanation for a puzzle.

    Only call this after the player has submitted a correct answer.
    """
    return {
        "explanation":          puzzle.explanation,
        "real_world_connection": puzzle.real_world_connection,
        "solution_features":    puzzle.solution_features,
        "intended_model":       puzzle.intended_model,
    }
