"""
test_evaluator.py — Tests for model evaluation and feature matrix building.
"""

import numpy as np
import pandas as pd
import pytest

from blackbox_game.evaluator import (
    build_feature_matrix, evaluate_linear_regression,
    evaluate_decision_tree, compare_models,
)
from blackbox_game.generator import generate_dataset
from blackbox_game.puzzles import get_puzzle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def linear_ds(n=200, seed=42):
    return generate_dataset(get_puzzle("line_01"), n_samples=n, seed=seed)

def quadratic_ds(n=200, seed=42):
    return generate_dataset(get_puzzle("square_01"), n_samples=n, seed=seed)

def piecewise_ds(n=200, seed=42):
    """Flat-then-rising piecewise function: a step-friendly shape for a tree."""
    x = np.random.default_rng(seed).uniform(0.0, 10.0, n)
    y = np.where(x < 5.0, 2.0 * x, 10.0)
    return {"X": pd.DataFrame({"x": x}), "y": y}


# ---------------------------------------------------------------------------
# build_feature_matrix
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
        X = pd.DataFrame({"x": [1.0]})
        with pytest.raises(ValueError, match="not in dataset"):
            build_feature_matrix(X, ["z"])

    def test_empty_features_raises(self):
        X = pd.DataFrame({"x": [1.0]})
        with pytest.raises(ValueError, match="must not be empty"):
            build_feature_matrix(X, [])


# ---------------------------------------------------------------------------
# Linear regression evaluator
# ---------------------------------------------------------------------------

class TestLinearRegression:

    def test_linear_feature_r2_near_one(self):
        ds     = linear_ds()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])
        result = evaluate_linear_regression(X_feat, ds["y"])
        assert result["r2"] > 0.999

    def test_square_feature_on_quadratic_r2_near_one(self):
        ds     = quadratic_ds()
        X_feat = build_feature_matrix(ds["X"], ["square:x"])
        result = evaluate_linear_regression(X_feat, ds["y"])
        assert result["r2"] > 0.999

    def test_square_feature_clearly_beats_raw_x_on_quadratic(self):
        ds        = quadratic_ds()
        X_raw     = build_feature_matrix(ds["X"], ["identity:x"])
        X_sq      = build_feature_matrix(ds["X"], ["square:x"])
        r2_raw    = evaluate_linear_regression(X_raw, ds["y"])["r2"]
        r2_sq     = evaluate_linear_regression(X_sq,  ds["y"])["r2"]
        assert r2_sq > r2_raw + 0.05
        assert r2_raw < 0.999

    def test_result_keys(self):
        ds     = linear_ds()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])
        result = evaluate_linear_regression(X_feat, ds["y"])
        for key in ["model", "r2", "mse", "cv_r2_mean", "verdict"]:
            assert key in result

    def test_model_key(self):
        ds     = linear_ds()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])
        assert evaluate_linear_regression(X_feat, ds["y"])["model"] == "linear_regression"

    def test_nan_feature_returns_nan_r2(self):
        X_feat = np.array([[1.0], [np.nan], [3.0]])
        y      = np.array([1.0, 2.0, 3.0])
        result = evaluate_linear_regression(X_feat, y)
        assert np.isnan(result["r2"])

    def test_mse_non_negative(self):
        ds     = quadratic_ds()
        X_feat = build_feature_matrix(ds["X"], ["square:x"])
        result = evaluate_linear_regression(X_feat, ds["y"])
        assert result["mse"] >= 0


# ---------------------------------------------------------------------------
# Decision tree evaluator
# ---------------------------------------------------------------------------

class TestDecisionTree:

    def test_piecewise_tree_beats_linear(self):
        ds     = piecewise_ds()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])
        lr     = evaluate_linear_regression(X_feat, ds["y"])
        dt     = evaluate_decision_tree(X_feat, ds["y"])
        assert dt["r2"] > lr["r2"]
        assert dt["r2"] > 0.98

    def test_classification_returns_accuracy(self):
        rng    = np.random.default_rng(42)
        x1, x2 = rng.uniform(-6, 6, 300), rng.uniform(-6, 6, 300)
        y      = (x1**2 + x2**2 < 4.0**2).astype(float)
        X_feat = np.sqrt(x1**2 + x2**2).reshape(-1, 1)
        result = evaluate_decision_tree(X_feat, y, task="classification")
        assert "accuracy" in result
        assert result["accuracy"] > 0.90

    def test_result_keys_regression(self):
        ds     = linear_ds()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])
        result = evaluate_decision_tree(X_feat, ds["y"])
        for key in ["model", "r2", "mse", "verdict"]:
            assert key in result


# ---------------------------------------------------------------------------
# Compare models
# ---------------------------------------------------------------------------

class TestCompareModels:

    def test_returns_both_models(self):
        ds     = linear_ds()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])
        result = compare_models(X_feat, ds["y"])
        assert "linear_regression" in result
        assert "decision_tree" in result

    def test_recommended_model_valid(self):
        ds     = linear_ds()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])
        result = compare_models(X_feat, ds["y"])
        assert result["recommended_model"] in ("linear_regression", "decision_tree")

    def test_summary_is_string(self):
        ds     = linear_ds()
        X_feat = build_feature_matrix(ds["X"], ["identity:x"])
        result = compare_models(X_feat, ds["y"])
        assert isinstance(result["summary"], str) and len(result["summary"]) > 0
