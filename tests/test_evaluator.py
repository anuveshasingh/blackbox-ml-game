"""
test_evaluator.py — Tests for model evaluation and feature matrix building.
"""

import numpy as np
import pytest

from blackbox_game.evaluator import (
    build_feature_matrix,
    evaluate_linear_regression,
    evaluate_decision_tree,
    compare_models,
)
from blackbox_game.generator import generate_dataset
from blackbox_game.puzzles import get_puzzle
import pandas as pd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def linear_dataset(n=200, seed=42):
    """y = 3x + 5 — perfect linear relationship."""
    puzzle = get_puzzle("line_01")
    return generate_dataset(puzzle, n_samples=n, seed=seed)


def quadratic_dataset(n=200, seed=42):
    """y = x² — quadratic relationship."""
    puzzle = get_puzzle("square_01")
    return generate_dataset(puzzle, n_samples=n, seed=seed)


def piecewise_dataset(n=200, seed=42):
    """Piecewise function — better captured by decision tree."""
    puzzle = get_puzzle("piecewise_01")
    return generate_dataset(puzzle, n_samples=n, seed=seed)


# ---------------------------------------------------------------------------
# Build feature matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:

    def test_plain_column(self):
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        mat = build_feature_matrix(X, ["x"])
        assert mat.shape == (3, 1)
        np.testing.assert_array_equal(mat[:, 0], [1.0, 2.0, 3.0])

    def test_transform_shorthand(self):
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        mat = build_feature_matrix(X, ["square:x"])
        np.testing.assert_array_almost_equal(mat[:, 0], [1.0, 4.0, 9.0])

    def test_binary_transform_dict(self):
        X = pd.DataFrame({"x1": [2.0, 4.0], "x2": [3.0, 5.0]})
        mat = build_feature_matrix(
            X, [{"binary": "multiply", "a": "x1", "b": "x2"}]
        )
        np.testing.assert_array_almost_equal(mat[:, 0], [6.0, 20.0])

    def test_multiple_features(self):
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        mat = build_feature_matrix(X, ["identity:x", "square:x"])
        assert mat.shape == (3, 2)

    def test_unknown_column_raises(self):
        X = pd.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="not in dataset"):
            build_feature_matrix(X, ["z"])

    def test_empty_features_raises(self):
        X = pd.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="must not be empty"):
            build_feature_matrix(X, [])


# ---------------------------------------------------------------------------
# Linear regression evaluator
# ---------------------------------------------------------------------------

class TestLinearRegression:

    def test_linear_feature_gets_high_r2(self):
        ds = linear_dataset()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])
        result = evaluate_linear_regression(X_feat, ds["y"])
        assert result["r2"] > 0.999, f"Expected R² > 0.999, got {result['r2']}"

    def test_quadratic_raw_x_gets_poor_linear_r2(self):
        """Raw x should give meaningfully lower R² than x² on y = x².

        On x in [0, 5], raw x is positively correlated with x² so R² is
        moderate (≈0.94).  x² (the correct feature) gives R² = 1.0, which is
        clearly better.  We just assert that raw x is NOT a perfect fit.
        """
        ds = quadratic_dataset()
        X_feat_raw = build_feature_matrix(ds["X"], ["identity:x"])
        X_feat_sq  = build_feature_matrix(ds["X"], ["square:x"])
        r2_raw = evaluate_linear_regression(X_feat_raw, ds["y"])["r2"]
        r2_sq  = evaluate_linear_regression(X_feat_sq,  ds["y"])["r2"]
        assert r2_raw < 0.999, (
            f"Raw x fit should be imperfect on y=x², got R²={r2_raw}"
        )
        assert r2_sq > r2_raw + 0.05, (
            f"x² (R²={r2_sq}) should be meaningfully better than raw x (R²={r2_raw})"
        )

    def test_quadratic_transformed_x_gets_high_r2(self):
        """x² feature should fit y = x² perfectly."""
        ds = quadratic_dataset()
        X_feat = build_feature_matrix(ds["X"], ["square:x"])
        result = evaluate_linear_regression(X_feat, ds["y"])
        assert result["r2"] > 0.999, f"Expected R² > 0.999, got {result['r2']}"

    def test_result_contains_expected_keys(self):
        ds = linear_dataset()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])
        result = evaluate_linear_regression(X_feat, ds["y"])
        for key in ["model", "r2", "mse", "cv_r2_mean", "verdict"]:
            assert key in result, f"Missing key '{key}' in result"

    def test_model_key_is_linear_regression(self):
        ds = linear_dataset()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])
        result = evaluate_linear_regression(X_feat, ds["y"])
        assert result["model"] == "linear_regression"

    def test_nan_feature_returns_nan_r2(self):
        """Feeding NaN values should return a graceful NaN result, not crash."""
        X_feat = np.array([[1.0], [np.nan], [3.0]])
        y = np.array([1.0, 2.0, 3.0])
        result = evaluate_linear_regression(X_feat, y)
        assert np.isnan(result["r2"])

    def test_mse_non_negative(self):
        ds = quadratic_dataset()
        X_feat = build_feature_matrix(ds["X"], ["square:x"])
        result = evaluate_linear_regression(X_feat, ds["y"])
        assert result["mse"] >= 0


# ---------------------------------------------------------------------------
# Decision tree evaluator
# ---------------------------------------------------------------------------

class TestDecisionTree:

    def test_piecewise_better_with_tree_than_linear(self):
        """Piecewise function should be better captured by a tree."""
        ds = piecewise_dataset()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])

        lr = evaluate_linear_regression(X_feat, ds["y"])
        dt = evaluate_decision_tree(X_feat, ds["y"])

        assert dt["r2"] > lr["r2"], (
            f"Decision tree R²={dt['r2']} should exceed linear R²={lr['r2']} "
            "for piecewise data."
        )
        assert dt["r2"] > 0.98, f"Decision tree should nearly perfectly fit piecewise, got {dt['r2']}"

    def test_linear_data_linear_wins(self):
        """Linear data should be better fit by linear regression than a shallow tree."""
        ds = linear_dataset()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])

        lr = evaluate_linear_regression(X_feat, ds["y"])
        dt = evaluate_decision_tree(X_feat, ds["y"])

        # Linear regression should achieve near-perfect R² on linear data
        assert lr["r2"] > 0.999
        # Tree may also do well on large datasets but linear should be at least as good
        assert lr["r2"] >= dt["r2"] - 0.01

    def test_classification_task(self):
        puzzle = get_puzzle("circle_01")
        ds = generate_dataset(puzzle, n_samples=300, seed=42)
        from blackbox_game.transforms import apply_binary_transform
        dist = apply_binary_transform("distance", ds["X"]["x1"].values, ds["X"]["x2"].values)
        X_feat = dist.reshape(-1, 1)
        result = evaluate_decision_tree(X_feat, ds["y"], task="classification")
        assert "accuracy" in result
        assert result["accuracy"] > 0.95

    def test_result_has_expected_keys_regression(self):
        ds = linear_dataset()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])
        result = evaluate_decision_tree(X_feat, ds["y"])
        for key in ["model", "r2", "mse", "verdict"]:
            assert key in result

    def test_result_has_expected_keys_classification(self):
        X_feat = np.random.rand(100, 1)
        y = (X_feat[:, 0] > 0.5).astype(float)
        result = evaluate_decision_tree(X_feat, y, task="classification")
        assert "accuracy" in result


# ---------------------------------------------------------------------------
# Compare models
# ---------------------------------------------------------------------------

class TestCompareModels:

    def test_returns_both_models(self):
        ds = linear_dataset()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])
        result = compare_models(X_feat, ds["y"])
        assert "linear_regression" in result
        assert "decision_tree" in result

    def test_recommended_model_is_valid(self):
        ds = linear_dataset()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])
        result = compare_models(X_feat, ds["y"])
        assert result["recommended_model"] in ("linear_regression", "decision_tree")

    def test_summary_is_string(self):
        ds = linear_dataset()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])
        result = compare_models(X_feat, ds["y"])
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0
