"""
test_generator.py — Tests for reproducible dataset generation.
"""

import numpy as np
import pandas as pd
import pytest

from blackbox_game.generator import generate_dataset
from blackbox_game.puzzles import get_puzzle, PUZZLE_REGISTRY


class TestReproducibility:
    """Same seed should always produce the same dataset."""

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
    """Returned datasets should have the correct shape and columns."""

    def test_n_samples_respected(self):
        puzzle = get_puzzle("line_01")
        dataset = generate_dataset(puzzle, n_samples=123, seed=0)
        assert len(dataset["y"]) == 123
        assert len(dataset["X"]) == 123

    def test_correct_columns_single_feature(self):
        puzzle = get_puzzle("line_01")
        dataset = generate_dataset(puzzle, n_samples=50)
        assert list(dataset["X"].columns) == ["x"]

    def test_correct_columns_two_features(self):
        puzzle = get_puzzle("product_01")
        dataset = generate_dataset(puzzle, n_samples=50)
        assert "x1" in dataset["X"].columns
        assert "x2" in dataset["X"].columns

    def test_correct_columns_four_features_distractor(self):
        puzzle = get_puzzle("distractor_01")
        dataset = generate_dataset(puzzle, n_samples=50)
        for col in ["x1", "x2", "x3", "x4"]:
            assert col in dataset["X"].columns

    def test_y_is_numpy_array(self):
        puzzle = get_puzzle("sqrt_01")
        dataset = generate_dataset(puzzle)
        assert isinstance(dataset["y"], np.ndarray)

    def test_X_is_dataframe(self):
        puzzle = get_puzzle("sqrt_01")
        dataset = generate_dataset(puzzle)
        assert isinstance(dataset["X"], pd.DataFrame)


class TestDomainConstraints:
    """Puzzle data should stay in valid domains so transforms work without NaN."""

    def test_sqrt_puzzle_x_nonnegative(self):
        puzzle = get_puzzle("sqrt_01")
        dataset = generate_dataset(puzzle, n_samples=200)
        assert np.all(dataset["X"]["x"].values >= 0)

    def test_log_puzzle_x_positive(self):
        puzzle = get_puzzle("log_01")
        dataset = generate_dataset(puzzle, n_samples=200)
        assert np.all(dataset["X"]["x"].values > 0)

    def test_reciprocal_puzzle_x_away_from_zero(self):
        puzzle = get_puzzle("reciprocal_01")
        dataset = generate_dataset(puzzle, n_samples=200)
        assert np.all(np.abs(dataset["X"]["x"].values) > 0.5)

    def test_no_nan_in_y_for_all_puzzles(self):
        """All puzzle datasets should produce finite y values."""
        for pid, puzzle in PUZZLE_REGISTRY.items():
            dataset = generate_dataset(puzzle, n_samples=50, seed=0)
            assert np.all(np.isfinite(dataset["y"])), (
                f"Puzzle '{pid}' produced NaN/inf in y."
            )

    def test_no_nan_in_X_for_all_puzzles(self):
        """All puzzle datasets should produce finite X values."""
        for pid, puzzle in PUZZLE_REGISTRY.items():
            dataset = generate_dataset(puzzle, n_samples=50, seed=0)
            assert dataset["X"].notna().all().all(), (
                f"Puzzle '{pid}' produced NaN in X."
            )


class TestNoiseApplication:
    def test_noise_changes_y(self):
        """Puzzles with noise_std > 0 should have non-deterministic y."""
        puzzle = get_puzzle("almost_linear_02")
        d1 = generate_dataset(puzzle, n_samples=100, seed=1)
        d2 = generate_dataset(puzzle, n_samples=100, seed=2)
        # Different seeds → different noise → different y
        assert not np.array_equal(d1["y"], d2["y"])

    def test_zero_noise_puzzle_deterministic_y(self):
        """Puzzles with noise_std=0 should give reproducible (seed-determined) datasets."""
        puzzle = get_puzzle("square_01")
        assert puzzle.noise_std == 0.0
        d1 = generate_dataset(puzzle, n_samples=50, seed=42)
        d2 = generate_dataset(puzzle, n_samples=50, seed=42)
        # Same seed → same X and same y
        np.testing.assert_array_equal(d1["X"].values, d2["X"].values)
        np.testing.assert_array_equal(d1["y"], d2["y"])
        # Confirm y = x² for the noise-free quadratic puzzle
        x_vals = d1["X"]["x"].values
        y_expected = x_vals ** 2
        np.testing.assert_array_almost_equal(d1["y"], y_expected, decimal=6)
