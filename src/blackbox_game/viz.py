"""
viz.py — Optional visualization helpers for notebooks and examples.

These helpers make it easy for a frontend or notebook to plot:
    1. Raw feature vs output
    2. Transformed feature vs output
    3. Residuals after model fit
    4. Decision-tree predictions

These functions are OPTIONAL — the core game engine never calls them.
They are provided as utilities for the frontend developer or notebook user.

Usage
-----
    import matplotlib.pyplot as plt
    from blackbox_game.viz import plot_xy, plot_residuals, plot_tree_predictions

    dataset = generate_dataset(puzzle)
    fig = plot_xy(dataset["X"]["x"].values, dataset["y"])
    plt.show()
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; override if needed
import matplotlib.pyplot as plt
from typing import Optional


def plot_xy(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str = "Feature vs Output",
    colour: str = "steelblue",
) -> plt.Figure:
    """
    Scatter plot of one feature against the output y.

    Parameters
    ----------
    x, y : np.ndarray
        Feature and output arrays.
    xlabel, ylabel, title : str
        Axis labels and plot title.
    colour : str
        Marker colour (any matplotlib colour string).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure (call ``plt.show()`` or ``fig.savefig(...)``).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, alpha=0.6, s=20, color=colour, edgecolors="none")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residuals(
    x_feat: np.ndarray,
    y: np.ndarray,
    title: str = "Residuals",
) -> plt.Figure:
    """
    Fit a linear model on ``x_feat`` and plot the residuals.

    Residuals are the leftover errors after the linear fit.
    If residuals show a pattern, the model is missing structure.

    Parameters
    ----------
    x_feat : np.ndarray
        Design matrix (n_samples × n_features) or 1-D feature array.
    y : np.ndarray
        Target values.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.linear_model import LinearRegression

    X = x_feat.reshape(-1, 1) if x_feat.ndim == 1 else x_feat
    model = LinearRegression()
    model.fit(X, y)
    residuals = y - model.predict(X)
    y_pred = model.predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Predicted vs actual
    axes[0].scatter(y_pred, y, alpha=0.5, s=20, color="steelblue", edgecolors="none")
    mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="y = ŷ")
    axes[0].set_xlabel("Predicted ŷ", fontsize=11)
    axes[0].set_ylabel("Actual y", fontsize=11)
    axes[0].set_title("Predicted vs Actual", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Residuals vs predicted
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20, color="coral", edgecolors="none")
    axes[1].axhline(0, color="grey", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Predicted ŷ", fontsize=11)
    axes[1].set_ylabel("Residual (y − ŷ)", fontsize=11)
    axes[1].set_title(title, fontsize=12)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_tree_predictions(
    x: np.ndarray,
    y: np.ndarray,
    max_depth: int = 4,
    xlabel: str = "x",
    title: str = "Decision Tree Predictions",
) -> plt.Figure:
    """
    Overlay a shallow decision tree's step-function predictions on a scatter.

    Only works for 1-D input x.

    Parameters
    ----------
    x, y : np.ndarray
        Single feature and output.
    max_depth : int
        Maximum tree depth.
    xlabel, title : str
        Axis labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.tree import DecisionTreeRegressor

    X = x.reshape(-1, 1)
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
    tree.fit(X, y)

    x_line = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
    y_line = tree.predict(x_line)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x, y, alpha=0.5, s=20, color="steelblue",
               edgecolors="none", label="Data")
    ax.plot(x_line, y_line, color="firebrick", linewidth=2,
            label=f"Decision tree (depth={max_depth})")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_2d_classification(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    title: str = "2D Classification",
) -> plt.Figure:
    """
    Scatter of (x1, x2) coloured by binary label y.

    Useful for the circle puzzle or any 2D classification problem.

    Parameters
    ----------
    x1, x2 : np.ndarray
        Coordinate arrays.
    y : np.ndarray
        Binary labels (0 or 1).
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, colour, marker in [(0, "steelblue", "o"), (1, "firebrick", "^")]:
        mask = y == label
        ax.scatter(
            x1[mask], x2[mask],
            alpha=0.6, s=25, color=colour, marker=marker,
            edgecolors="none", label=f"y = {int(label)}",
        )
    ax.set_xlabel("x1", fontsize=12)
    ax.set_ylabel("x2", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_transform_comparison(
    x: np.ndarray,
    y: np.ndarray,
    transform_keys: list[str],
    ncols: int = 3,
) -> plt.Figure:
    """
    Grid of scatter plots: y vs each transform of x.

    Helps the player visually identify which transformation
    produces a linear-looking relationship.

    Parameters
    ----------
    x : np.ndarray
        Raw input feature.
    y : np.ndarray
        Output values.
    transform_keys : list[str]
        Keys from ``TRANSFORM_REGISTRY`` to evaluate.
    ncols : int
        Number of columns in the grid.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .transforms import apply_transform, TRANSFORM_DESCRIPTIONS

    n = len(transform_keys)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, key in enumerate(transform_keys):
        ax = axes[i]
        try:
            x_t = apply_transform(key, x)
            valid = np.isfinite(x_t)
            ax.scatter(
                x_t[valid], y[valid],
                alpha=0.5, s=15, color="steelblue", edgecolors="none",
            )
            ax.set_xlabel(TRANSFORM_DESCRIPTIONS.get(key, key), fontsize=10)
            ax.set_ylabel("y", fontsize=10)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Error:\n{exc}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        ax.set_title(f"y vs {TRANSFORM_DESCRIPTIONS.get(key, key)}", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Transform Comparison", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig
