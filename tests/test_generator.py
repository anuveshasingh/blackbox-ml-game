"""
test_generator.py — Reproducible dataset generation tests.
"""

import dataclasses

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
        dataset = generate_dataset(get_puzzle("coulomb_2"), n_samples=10)
        assert list(dataset["X"].columns) == ["r1", "r2"]

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


    def test_no_nan_in_y_for_all_puzzles(self):
        for pid, puzzle in PUZZLE_REGISTRY.items():
            if puzzle.function.type == "image":
                continue
            dataset = generate_dataset(puzzle, n_samples=50, seed=0)
            assert np.all(np.isfinite(dataset["y"])), (
                f"Puzzle '{pid}' produced NaN/inf in y."
            )

    def test_no_nan_in_X_for_all_puzzles(self):
        for pid, puzzle in PUZZLE_REGISTRY.items():
            if puzzle.function.type == "image":
                continue
            dataset = generate_dataset(puzzle, n_samples=50, seed=0)
            assert dataset["X"].notna().all().all(), (
                f"Puzzle '{pid}' produced NaN in X."
            )


class TestPhysicsData:
    """Physics puzzles are exact: y equals the formula with no added noise."""

    def test_projectile_y_matches_formula(self):
        d = generate_dataset(get_puzzle("projectile_y"), n_samples=50)
        X = d["X"]
        np.testing.assert_allclose(d["y"], 10.0 * np.sin(X.theta) * X.t - 0.5 * 9.80665 * X.t**2)

    def test_coulomb_2_matches_formula(self):
        d = generate_dataset(get_puzzle("coulomb_2"), n_samples=50)
        X = d["X"]
        expected = 8.9875517923e9 * (2e-6 / X.r1 - 3e-6 / X.r2)
        np.testing.assert_allclose(d["y"], expected, rtol=1e-12)

    def test_inputs_within_declared_ranges(self):
        for pid in ("projectile_y", "shm_energy", "coulomb_2"):
            puzzle = get_puzzle(pid)
            d = generate_dataset(puzzle, n_samples=200)
            for name, (lo, hi) in puzzle.function.parameters["ranges"].items():
                col = d["X"][name].values
                assert np.all((col >= lo) & (col <= hi)), f"{pid}.{name} out of range"

    def test_all_physics_outputs_finite_and_nonzero_scale(self):
        for pid, puzzle in PUZZLE_REGISTRY.items():
            if puzzle.function.type != "physics":
                continue
            d = generate_dataset(puzzle, n_samples=25, seed=42)
            assert np.all(np.isfinite(d["y"])), pid
            assert np.std(d["y"]) > 0, pid


class TestNoise:

    def test_noise_changes_y(self):
        puzzle = dataclasses.replace(get_puzzle("line_01"), noise_std=0.1)
        d1 = generate_dataset(puzzle, n_samples=100, seed=1)
        d2 = generate_dataset(puzzle, n_samples=100, seed=2)
        assert not np.array_equal(d1["y"], d2["y"])

    def test_zero_noise_is_deterministic_function(self):
        puzzle = get_puzzle("square_01")  # noise_std=0
        assert puzzle.noise_std == 0.0
        d = generate_dataset(puzzle, n_samples=50, seed=42)
        x = d["X"]["x"].values
        np.testing.assert_array_almost_equal(d["y"], x**2, decimal=6)
