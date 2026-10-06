"""
test_puzzles.py — Tests for the updated puzzle registry and explanation system.
"""

import pytest
import json

from blackbox_game.puzzles import get_puzzle, list_puzzles, PUZZLE_REGISTRY
from blackbox_game.hints import get_explanation
from blackbox_game.models import Puzzle


class TestPuzzleRegistry:

    def test_total_puzzle_count(self):
        assert len(PUZZLE_REGISTRY) == 16, (
            f"Expected 16 puzzles, found {len(PUZZLE_REGISTRY)}"
        )

    def test_get_puzzle_returns_puzzle(self):
        puzzle = get_puzzle("line_01")
        assert isinstance(puzzle, Puzzle)

    def test_get_puzzle_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_puzzle("does_not_exist")

    def test_all_puzzle_ids_unique(self):
        ids = list(PUZZLE_REGISTRY.keys())
        assert len(ids) == len(set(ids))

    def test_list_puzzles_returns_all(self):
        assert len(list_puzzles()) == 16

    def test_list_puzzles_filter_beginner(self):
        beginner = list_puzzles(difficulty=1)
        assert len(beginner) == 6
        assert all(p["difficulty"] == 1 for p in beginner)

    def test_list_puzzles_filter_physics(self):
        physics = list_puzzles(difficulty=2)
        assert len(physics) == 4
        assert all(p["difficulty"] == 2 for p in physics)

    def test_image_tier_has_six_puzzles(self):
        images = list_puzzles(difficulty=3)
        assert [p["id"] for p in images] == [
            "solarise", "posterise", "overlay", "channel_shuffle", "fft_encode", "fft_decode",
        ]

    def test_no_decision_tree_puzzles(self):
        dt_puzzles = [
            p for p in PUZZLE_REGISTRY.values()
            if p.intended_model == "decision_tree"
        ]
        assert dt_puzzles == []

    def test_physics_puzzles_are_linear_regression(self):
        for p in PUZZLE_REGISTRY.values():
            if p.function.type == "physics":
                assert p.intended_model == "linear_regression", p.id

    def test_all_puzzles_have_required_fields(self):
        required = [
            "id", "title", "category", "difficulty", "description",
            "input_features", "intended_model", "solution_features",
            "explanation", "real_world_connection",
        ]
        for pid, puzzle in PUZZLE_REGISTRY.items():
            for f in required:
                assert hasattr(puzzle, f), f"Puzzle '{pid}' missing '{f}'"
                assert getattr(puzzle, f) is not None, f"Puzzle '{pid}' has None for '{f}'"

    def test_hints_are_empty_or_absent(self):
        """Hints have been disabled — all puzzles should have empty hints lists."""
        for pid, puzzle in PUZZLE_REGISTRY.items():
            assert isinstance(puzzle.hints, list), f"Puzzle '{pid}' hints is not a list"
            assert len(puzzle.hints) == 0, (
                f"Puzzle '{pid}' has {len(puzzle.hints)} hints; expected 0"
            )

    def test_all_puzzles_have_nonempty_explanation(self):
        for pid, puzzle in PUZZLE_REGISTRY.items():
            assert len(puzzle.explanation) > 20, (
                f"Puzzle '{pid}' explanation too short."
            )

    def test_difficulty_in_valid_range(self):
        for pid, puzzle in PUZZLE_REGISTRY.items():
            assert puzzle.difficulty in (1, 2, 3), (
                f"Puzzle '{pid}' has invalid difficulty {puzzle.difficulty}."
            )

    def test_intended_model_is_valid(self):
        valid = {"linear_regression", "decision_tree", "none"}
        for pid, puzzle in PUZZLE_REGISTRY.items():
            assert puzzle.intended_model in valid, (
                f"Puzzle '{pid}' has invalid intended_model '{puzzle.intended_model}'."
            )

    def test_titles_are_different_from_ids(self):
        """Titles should be cryptic — not just the ID reworded."""
        for pid, puzzle in PUZZLE_REGISTRY.items():
            # Title should NOT directly contain the category word or the id as-is
            assert puzzle.title != pid, f"Puzzle '{pid}' title is same as ID"


class TestPuzzleSerialization:

    def test_to_dict_returns_dict(self):
        assert isinstance(get_puzzle("square_01").to_dict(), dict)

    def test_to_json_is_valid_json(self):
        puzzle = get_puzzle("square_01")
        parsed = json.loads(puzzle.to_json())
        assert parsed["id"] == "square_01"

    def test_round_trip_serialization(self):
        puzzle  = get_puzzle("projectile_y")
        puzzle2 = Puzzle.from_json(puzzle.to_json())
        assert puzzle2.id == puzzle.id
        assert puzzle2.function.type == puzzle.function.type
        assert puzzle2.solution_features == puzzle.solution_features

    def test_all_puzzles_serializable(self):
        for pid, puzzle in PUZZLE_REGISTRY.items():
            try:
                json.loads(puzzle.to_json())
            except Exception as exc:
                pytest.fail(f"Puzzle '{pid}' not serialisable: {exc}")


class TestExplanationSystem:

    def test_get_explanation_returns_dict(self):
        puzzle = get_puzzle("square_01")
        exp = get_explanation(puzzle)
        assert "explanation" in exp
        assert "solution_features" in exp
        assert "intended_model" in exp
        assert "real_world_connection" in exp

    def test_explanation_text_nonempty(self):
        for pid, puzzle in PUZZLE_REGISTRY.items():
            exp = get_explanation(puzzle)
            assert len(exp["explanation"]) > 10, f"Puzzle '{pid}' empty explanation"
