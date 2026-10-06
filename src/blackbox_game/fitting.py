"""
fitting.py — Feature building and model fitting for the ``residuals`` command.

build_feature_matrix(X, features)
    Turn feature specifications (columns, transforms, products, sums) into
    a numpy design matrix.

predict_model(X_feat, y, model)
    Fit linear regression or a shallow decision tree and return predictions
    for the same rows.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from .transforms import apply_binary_transform, apply_transform

# Decision-tree depth kept shallow so it stays interpretable
_DT_MAX_DEPTH = 4


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
    * a dict ``{"product": [spec, ...]}``
      → multiplies the element-wise values of nested specs
    * a dict ``{"sum": [spec, {"term": spec, "sign": -1}, ...]}``
      → adds (or subtracts, with ``sign: -1``) nested specs

    Any ``product`` or ``sum`` dict may also carry ``"transform": key`` to
    apply a unary transform to its result (e.g. ``exp_neg`` or ``sin``).

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
    cols = [_eval_spec(X, spec) for spec in features]

    if not cols:
        raise ValueError("features list must not be empty.")

    matrix = np.column_stack(cols)
    return matrix


def _eval_spec(X: pd.DataFrame, spec: Union[str, dict]) -> np.ndarray:
    """Evaluate one feature specification to a 1-D array (see build_feature_matrix)."""
    if isinstance(spec, dict) and "product" in spec:
        result = np.prod([_eval_spec(X, s) for s in spec["product"]], axis=0)
    elif isinstance(spec, dict) and "sum" in spec:
        terms = []
        for item in spec["sum"]:
            sign = 1.0
            if isinstance(item, dict) and "term" in item:
                sign = float(item.get("sign", 1))
                item = item["term"]
            terms.append(sign * _eval_spec(X, item))
        result = np.sum(terms, axis=0)
    elif isinstance(spec, dict):
        # Binary transform
        if not {"binary", "a", "b"} <= spec.keys():
            raise ValueError(f"Feature dict must have 'product', 'sum', or 'binary'/'a'/'b': {spec}")
        for column in (spec["a"], spec["b"]):
            if column not in X.columns:
                raise ValueError(f"Column '{column}' not in dataset.")
        a = X[spec["a"]].values
        b = X[spec["b"]].values
        return apply_binary_transform(spec["binary"], a, b)
    elif ":" in spec:
        # "transform:column" shorthand
        transform_key, col_name = spec.split(":", 1)
        col_name = col_name.strip()
        transform_key = transform_key.strip()
        if col_name not in X.columns:
            raise ValueError(f"Column '{col_name}' not in dataset.")
        return apply_transform(transform_key, X[col_name].values)
    else:
        # Plain column reference
        if spec not in X.columns:
            raise ValueError(f"Column '{spec}' not in dataset.")
        return X[spec].values.astype(float)

    if isinstance(spec, dict) and "transform" in spec:
        result = apply_transform(spec["transform"], result)
    return result


def predict_model(X_feat: np.ndarray, y: np.ndarray, model: str) -> np.ndarray:
    """Fit a supported model and return predictions for the same rows."""
    if not np.all(np.isfinite(X_feat)):
        raise ValueError("A feature contains invalid values (NaN). Check the transform domain.")
    if model == "linear_regression":
        # Scale each column to unit max-magnitude. Predictions are unchanged
        # (linear regression is scale-invariant), but without it a column
        # near 1e10 makes the least-squares solver drop the O(1) columns.
        scale = np.max(np.abs(X_feat), axis=0)
        scale[scale == 0] = 1.0
        X_feat = X_feat / scale
        estimator = LinearRegression()
    elif model == "decision_tree":
        estimator = DecisionTreeRegressor(max_depth=_DT_MAX_DEPTH, random_state=0)
    else:
        raise ValueError(
            f"Unknown model '{model}'. Use 'linear_regression' or 'decision_tree'."
        )
    estimator.fit(X_feat, y)
    return np.asarray(estimator.predict(X_feat), dtype=float)
