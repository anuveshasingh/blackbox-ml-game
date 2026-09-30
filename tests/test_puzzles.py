"""
test_puzzles.py — Tests for the puzzle registry and hint/explanation system.
"""

import pytest

from blackbox_game.puzzles import get_puzzle, list_puzzles, PUZZLE_REGISTRY
from blackbox_game.hints import get_hint, get_explanation, get_all_hints
from blackbox_game.models import Puzzle
import json


class TestPuzzleRegistry:

    def test_total_puzzle_count(self):
        assert len(PUZZLE_REGISTRY) == 25, (
            f"Expected 25 puzzles, found {len(PUZZLE_REGISTRY)}"
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
        puzzles = list_puzzles()
        assert len(puzzles) == 25

    def test_list_puzzles_filter_beginner(self):
        beginner = list_puzzles(difficulty=1)
        assert len(beginner) == 6
        assert all(p["difficulty"] == 1 for p in beginner)

    def test_list_puzzles_filter_intermediate(self):
        intermediate = list_puzzles(difficulty=2)
        assert len(intermediate) == 9
        assert all(p["difficulty"] == 2 for p in intermediate)

    def test_list_puzzles_filter_challenge(self):
        challenge = list_puzzles(difficulty=3)
        assert len(challenge) == 10
        assert all(p["difficulty"] == 3 for p in challenge)

    def test_all_puzzles_have_required_fields(self):
        required = [
            "id", "title", "category", "difficulty", "description",
            "input_features", "intended_model", "solution_features",
            "hints", "explanation", "real_world_connection",
        ]
        for pid, puzzle in PUZZLE_REGISTRY.items():
            for field in required:
                assert hasattr(puzzle, field), f"Puzzle '{pid}' missing field '{field}'"
                val = getattr(puzzle, field)
                assert val is not None, f"Puzzle '{pid}' has None for '{field}'"

    def test_all_puzzles_have_at_least_two_hints(self):
        for pid, puzzle in PUZZLE_REGISTRY.items():
            assert len(puzzle.hints) >= 2, (
                f"Puzzle '{pid}' has only {len(puzzle.hints)} hint(s). Need at least 2."
            )

    def test_all_puzzles_have_nonempty_explanation(self):
        for pid, puzzle in PUZZLE_REGISTRY.items():
            assert len(puzzle.explanation) > 20, (
                f"Puzzle '{pid}' explanation is too short."
            )

    def test_difficulty_in_valid_range(self):
        for pid, puzzle in PUZZLE_REGISTRY.items():
            assert puzzle.difficulty in (1, 2, 3), (
                f"Puzzle '{pid}' has invalid difficulty {puzzle.difficulty}."
            )

    def test_intended_model_is_valid(self):
        valid_models = {"linear_regression", "decision_tree"}
        for pid, puzzle in PUZZLE_REGISTRY.items():
            assert puzzle.intended_model in valid_models, (
                f"Puzzle '{pid}' has invalid intended_model '{puzzle.intended_model}'."
            )


class TestPuzzleSerialization:

    def test_to_dict_returns_dict(self):
        puzzle = get_puzzle("square_01")
        d = puzzle.to_dict()
        assert isinstance(d, dict)

    def test_to_json_is_valid_json(self):
        puzzle = get_puzzle("square_01")
        json_str = puzzle.to_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "square_01"

    def test_round_trip_serialization(self):
        puzzle = get_puzzle("product_01")
        json_str = puzzle.to_json()
        puzzle2 = Puzzle.from_json(json_str)
        assert puzzle2.id == puzzle.id
        assert puzzle2.function.type == puzzle.function.type
        assert puzzle2.solution_features == puzzle.solution_features

    def test_all_puzzles_serializable(self):
        for pid, puzzle in PUZZLE_REGISTRY.items():
            try:
                json_str = puzzle.to_json()
                json.loads(json_str)
            except Exception as exc:
                pytest.fail(f"Puzzle '{pid}' could not be serialised to JSON: {exc}")


class TestHintSystem:

    def test_get_first_hint(self):
        puzzle = get_puzzle("square_01")
        hint = get_hint(puzzle, 0)
        assert hint["hint_index"] == 0
        assert isinstance(hint["text"], str)
        assert len(hint["text"]) > 5
        assert hint["cost"] == -20

    def test_hint_index_out_of_bounds(self):
        puzzle = get_puzzle("square_01")
        hint = get_hint(puzzle, 999)
        assert "No more hints" in hint["text"]
        assert hint["cost"] == 0

    def test_total_hints_reported_correctly(self):
        puzzle = get_puzzle("line_01")
        hint = get_hint(puzzle, 0)
        assert hint["total_hints"] == len(puzzle.hints)

    def test_get_explanation_returns_dict(self):
        puzzle = get_puzzle("square_01")
        exp = get_explanation(puzzle)
        assert "explanation" in exp
        assert "solution_features" in exp
        assert "intended_model" in exp
        assert "real_world_connection" in exp

    def test_get_all_hints_returns_all(self):
        puzzle = get_puzzle("line_01")
        hints = get_all_hints(puzzle)
        assert len(hints) == len(puzzle.hints)
        for i, h in enumerate(hints):
            assert h["hint_index"] == i
