"""
evaluator.py — Model fitting and scoring helpers.

Internally uses scikit-learn. Students never interact with this module
directly; the game engine calls it on their behalf.

Key functions
-------------
evaluate_linear_regression(X_feat, y)
    Fit linear regression, return R², MSE, and a quick verdict.

evaluate_decision_tree(X_feat, y, task)
    Fit a shallow decision-tree regressor or classifier.

compare_models(X_feat, y, task)
    Run both models and return a side-by-side comparison.

build_feature_matrix(puzzle, dataset, features)
    Convert a list of feature keys (possibly with transforms) into a
    numpy design matrix the models can consume.

evaluate_submission(puzzle_id, model, features)
    High-level end-to-end call used by the frontend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

from .models import Puzzle
from .transforms import (
    apply_transform,
    apply_binary_transform,
    TRANSFORM_REGISTRY,
    BINARY_TRANSFORM_REGISTRY,
)

# R² threshold above which we consider a fit "correct"
_CORRECT_R2_THRESHOLD = 0.92
_CORRECT_ACCURACY_THRESHOLD = 0.90

# Decision-tree depth kept shallow so it stays interpretable
_DT_MAX_DEPTH = 4


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    X: pd.DataFrame,
    features: list[Union[str, dict]],
) -> np.ndarray:
    """
    Construct a 2-D numpy design matrix from a list of feature specifications.

    Each element of ``features`` can be:

    * a plain string matching a column in ``X``
      → uses that column as-is
    * a string of the form ``"transform:column"``
      → applies the named unary transform to that column
      (e.g. ``"square:x"``)
    * a dict ``{"binary": "multiply", "a": "x1", "b": "x2"}``
      → applies a binary transform to two columns

    Parameters
    ----------
    X : pd.DataFrame
        Raw feature DataFrame from ``generate_dataset``.
    features : list
        Feature specifications as described above.

    Returns
    -------
    np.ndarray
        Shape (n_samples, len(features)).

    Raises
    ------
    ValueError
        If a specified column or transform is not found.
    """
    cols = []
    for spec in features:
        if isinstance(spec, dict):
            # Binary transform
            binary_key = spec["binary"]
            a = X[spec["a"]].values
            b = X[spec["b"]].values
            cols.append(apply_binary_transform(binary_key, a, b))
        elif ":" in spec:
            # "transform:column" shorthand
            transform_key, col_name = spec.split(":", 1)
            col_name = col_name.strip()
            transform_key = transform_key.strip()
            if col_name not in X.columns:
                raise ValueError(f"Column '{col_name}' not in dataset.")
            cols.append(apply_transform(transform_key, X[col_name].values))
        else:
            # Plain column reference
            if spec not in X.columns:
                raise ValueError(f"Column '{spec}' not in dataset.")
            cols.append(X[spec].values.astype(float))

    if not cols:
        raise ValueError("features list must not be empty.")

    matrix = np.column_stack(cols)
    return matrix


def _has_nan(arr: np.ndarray) -> bool:
    return bool(np.any(~np.isfinite(arr)))


# ---------------------------------------------------------------------------
# Linear regression evaluator
# ---------------------------------------------------------------------------

def evaluate_linear_regression(
    X_feat: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
) -> dict:
    """
    Fit a linear regression model and return quality metrics.

    Parameters
    ----------
    X_feat : np.ndarray
        Design matrix (n_samples × n_features).
    y : np.ndarray
        Target values.
    cv_folds : int
        Number of cross-validation folds (default 5).

    Returns
    -------
    dict with keys:
        model, r2, mse, cv_r2_mean, cv_r2_std, verdict
    """
    if _has_nan(X_feat):
        return {
            "model": "linear_regression",
            "r2": float("nan"),
            "mse": float("nan"),
            "cv_r2_mean": float("nan"),
            "cv_r2_std": float("nan"),
            "verdict": "Feature contains invalid values (NaN). "
                       "Check the transform domain.",
        }

    model = LinearRegression()
    model.fit(X_feat, y)
    y_pred = model.predict(X_feat)

    r2 = float(r2_score(y, y_pred))
    mse = float(mean_squared_error(y, y_pred))

    n = len(y)
    folds = min(cv_folds, n // 5) if n >= 10 else 2
    folds = max(folds, 2)
    cv_scores = cross_val_score(
        LinearRegression(), X_feat, y, cv=folds, scoring="r2"
    )
    cv_r2_mean = float(cv_scores.mean())
    cv_r2_std = float(cv_scores.std())

    verdict = _r2_verdict(r2)

    return {
        "model": "linear_regression",
        "r2": round(r2, 4),
        "mse": round(mse, 4),
        "cv_r2_mean": round(cv_r2_mean, 4),
        "cv_r2_std": round(cv_r2_std, 4),
        "verdict": verdict,
    }


def _r2_verdict(r2: float) -> str:
    if r2 >= 0.97:
        return "Excellent fit! This feature almost perfectly explains the output."
    elif r2 >= 0.90:
        return "Very good fit. You are on the right track!"
    elif r2 >= 0.70:
        return "Moderate fit. There is still some unexplained structure."
    elif r2 >= 0.40:
        return "Weak fit. The feature captures some pattern but misses a lot."
    elif r2 >= 0.0:
        return "Poor fit. This feature barely explains the output."
    else:
        return "Negative R²: this feature is worse than just predicting the mean."


# ---------------------------------------------------------------------------
# Decision tree evaluator
# ---------------------------------------------------------------------------

def evaluate_decision_tree(
    X_feat: np.ndarray,
    y: np.ndarray,
    task: str = "regression",
    max_depth: int = _DT_MAX_DEPTH,
    cv_folds: int = 5,
) -> dict:
    """
    Fit a shallow decision tree and return quality metrics.

    Parameters
    ----------
    X_feat : np.ndarray
        Design matrix.
    y : np.ndarray
        Target values.
    task : str
        "regression" (default) or "classification".
    max_depth : int
        Maximum tree depth (kept shallow for interpretability).
    cv_folds : int
        Cross-validation folds.

    Returns
    -------
    dict with keys:
        model, task, r2/accuracy, mse (regression only), verdict
    """
    if _has_nan(X_feat):
        return {
            "model": "decision_tree",
            "task": task,
            "error": "Feature contains invalid values (NaN).",
        }

    result = {"model": "decision_tree", "task": task, "max_depth": max_depth}

    if task == "classification":
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        clf.fit(X_feat, y)
        y_pred = clf.predict(X_feat)
        acc = float(accuracy_score(y, y_pred))

        n = len(y)
        folds = min(cv_folds, n // 5) if n >= 10 else 2
        folds = max(folds, 2)
        cv_scores = cross_val_score(
            DecisionTreeClassifier(max_depth=max_depth, random_state=0),
            X_feat, y, cv=folds, scoring="accuracy",
        )
        result["accuracy"] = round(acc, 4)
        result["cv_accuracy_mean"] = round(float(cv_scores.mean()), 4)
        result["cv_accuracy_std"] = round(float(cv_scores.std()), 4)
        result["verdict"] = _accuracy_verdict(acc)
    else:
        reg = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
        reg.fit(X_feat, y)
        y_pred = reg.predict(X_feat)
        r2 = float(r2_score(y, y_pred))
        mse = float(mean_squared_error(y, y_pred))

        n = len(y)
        folds = min(cv_folds, n // 5) if n >= 10 else 2
        folds = max(folds, 2)
        cv_scores = cross_val_score(
            DecisionTreeRegressor(max_depth=max_depth, random_state=0),
            X_feat, y, cv=folds, scoring="r2",
        )
        result["r2"] = round(r2, 4)
        result["mse"] = round(mse, 4)
        result["cv_r2_mean"] = round(float(cv_scores.mean()), 4)
        result["cv_r2_std"] = round(float(cv_scores.std()), 4)
        result["verdict"] = _r2_verdict(r2)

    return result


def _accuracy_verdict(acc: float) -> str:
    if acc >= 0.95:
        return "Excellent! The decision tree classifies almost perfectly."
    elif acc >= 0.85:
        return "Very good classification accuracy."
    elif acc >= 0.70:
        return "Moderate accuracy. Some points are misclassified."
    else:
        return "Poor accuracy. Try a different feature or model."


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    X_feat: np.ndarray,
    y: np.ndarray,
    task: str = "regression",
) -> dict:
    """
    Evaluate both linear regression and decision tree on the same features.

    Useful for the player to understand which model better captures the
    hidden relationship.

    Returns
    -------
    dict with keys:
        "linear_regression": {...}
        "decision_tree": {...}
        "recommended_model": "linear_regression" | "decision_tree"
        "summary": human-readable comparison string
    """
    lr = evaluate_linear_regression(X_feat, y)
    dt = evaluate_decision_tree(X_feat, y, task=task)

    if task == "regression":
        lr_score = lr.get("r2", -999)
        dt_score = dt.get("r2", -999)
    else:
        lr_score = lr.get("r2", -999)
        dt_score = dt.get("accuracy", 0)

    recommended = (
        "linear_regression" if lr_score >= dt_score else "decision_tree"
    )

    summary = (
        f"Linear regression R²={lr.get('r2', 'N/A')}. "
        f"Decision tree R²={dt.get('r2', dt.get('accuracy', 'N/A'))}. "
        f"Recommended: {recommended.replace('_', ' ')}."
    )

    return {
        "linear_regression": lr,
        "decision_tree": dt,
        "recommended_model": recommended,
        "summary": summary,
    }
