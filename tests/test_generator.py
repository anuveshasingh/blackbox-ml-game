"""
test_generator.py — Reproducible dataset generation tests.
"""

import numpy as np
import pandas as pd
import pytest

from blackbox_game.generator import generate_dataset
from blackbox_game.puzzles import get_puzzle, PUZZLE_REGISTRY


class TestReproducibility:

    def test_same_seed_same_output(self):
        puzzle = get_puzzle("square_01")
        d1 = generate_dataset(puzzle, n_samples=50, seed=42)
        d2 = generate_dataset(puzzle, n_samples=50, seed=42)
        np.testing.assert_array_equal(d1["y"], d2["y"])
        pd.testing.assert_frame_equal(d1["X"], d2["X"])

    def test_different_seed_different_output(self):
        puzzle = get_puzzle("square_01")
        d1 = generate_dataset(puzzle, n_samples=50, seed=42)
        d2 = generate_dataset(puzzle, n_samples=50, seed=99)
        assert not np.array_equal(d1["y"], d2["y"])


class TestDatasetShape:

    def test_n_samples_respected(self):
        dataset = generate_dataset(get_puzzle("line_01"), n_samples=123)
        assert len(dataset["y"]) == 123
        assert len(dataset["X"]) == 123

    def test_single_feature_column(self):
        dataset = generate_dataset(get_puzzle("line_01"), n_samples=10)
        assert list(dataset["X"].columns) == ["x"]

    def test_two_feature_columns(self):
        dataset = generate_dataset(get_puzzle("product_01"), n_samples=10)
        assert "x1" in dataset["X"].columns
        assert "x2" in dataset["X"].columns

    def test_distractor_has_four_columns(self):
        dataset = generate_dataset(get_puzzle("distractor_01"), n_samples=10)
        for col in ["x1", "x2", "x3", "x4"]:
            assert col in dataset["X"].columns

    def test_y_is_numpy(self):
        assert isinstance(generate_dataset(get_puzzle("sqrt_01"))["y"], np.ndarray)

    def test_X_is_dataframe(self):
        assert isinstance(generate_dataset(get_puzzle("sqrt_01"))["X"], pd.DataFrame)


class TestDomainConstraints:

    def test_sqrt_puzzle_x_nonnegative(self):
        dataset = generate_dataset(get_puzzle("sqrt_01"), n_samples=200)
        assert np.all(dataset["X"]["x"].values >= 0)

    def test_log_puzzle_x_positive(self):
        dataset = generate_dataset(get_puzzle("log_01"), n_samples=200)
        assert np.all(dataset["X"]["x"].values > 0)

    def test_reciprocal_puzzle_x_away_from_zero(self):
        dataset = generate_dataset(get_puzzle("reciprocal_01"), n_samples=200)
        assert np.all(np.abs(dataset["X"]["x"].values) > 0.5)

    def test_no_nan_in_y_for_all_puzzles(self):
        for pid, puzzle in PUZZLE_REGISTRY.items():
            dataset = generate_dataset(puzzle, n_samples=50, seed=0)
            assert np.all(np.isfinite(dataset["y"])), (
                f"Puzzle '{pid}' produced NaN/inf in y."
            )

    def test_no_nan_in_X_for_all_puzzles(self):
        for pid, puzzle in PUZZLE_REGISTRY.items():
            dataset = generate_dataset(puzzle, n_samples=50, seed=0)
            assert dataset["X"].notna().all().all(), (
                f"Puzzle '{pid}' produced NaN in X."
            )


class TestNewPuzzleData:
    """Specific checks for the four new LR puzzles."""

    def test_abs_puzzle_y_nonnegative(self):
        """y = |x| is always non-negative."""
        dataset = generate_dataset(get_puzzle("abs_01"), n_samples=200)
        assert np.all(dataset["y"] >= 0)

    def test_cos_puzzle_y_bounded(self):
        """y = cos(x) stays in [-1, 1]."""
        dataset = generate_dataset(get_puzzle("cos_01"), n_samples=200)
        assert np.all(dataset["y"] >= -1.1)
        assert np.all(dataset["y"] <= 1.1)

    def test_polynomial_puzzle_two_features_needed(self):
        """polynomial_01 has only 'x' as input feature."""
        dataset = generate_dataset(get_puzzle("polynomial_01"), n_samples=50)
        assert list(dataset["X"].columns) == ["x"]

    def test_phase_puzzle_y_bounded(self):
        """y = sin(x)+cos(x) has amplitude sqrt(2) ≈ 1.41."""
        dataset = generate_dataset(get_puzzle("phase_01"), n_samples=300)
        assert np.all(np.abs(dataset["y"]) <= 1.5)


class TestNoise:

    def test_noise_changes_y(self):
        puzzle = get_puzzle("almost_linear_02")  # noise_std=0.1
        d1 = generate_dataset(puzzle, n_samples=100, seed=1)
        d2 = generate_dataset(puzzle, n_samples=100, seed=2)
        assert not np.array_equal(d1["y"], d2["y"])

    def test_zero_noise_is_deterministic_function(self):
        puzzle = get_puzzle("square_01")  # noise_std=0
        assert puzzle.noise_std == 0.0
        d = generate_dataset(puzzle, n_samples=50, seed=42)
        x = d["X"]["x"].values
        np.testing.assert_array_almost_equal(d["y"], x**2, decimal=6)
