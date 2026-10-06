"""
plots.py — 3D plot files for ``play.py show --plot``.

Each pair of inputs gets one PLY point cloud: the first input on the x-axis,
the second on the z-axis, and the output y on the vertical axis. Each data
point is drawn as a small sphere of points, coloured by y. Axes are drawn as
coloured lines (red = x, green = y, blue = z) from the lower corner of the box.

Axis names and end values are drawn as text made of points, because PLY has
no text. The text is rasterised from matplotlib's own font outlines.

PLY is plain text, so no 3D library is needed. Open the files in VS Code with
the PLY viewer extension (see viewer.py) to rotate them with the mouse.

Coordinates are normalised to [-1, 1] per axis, so the original ranges are
written into the PLY header as comments.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd

_MARKER_POINTS = 120       # points per data-point sphere
_MARKER_RADIUS = 0.04      # sphere radius in normalised units
_AXIS_POINTS = 80          # points per axis line
_AXIS_COLOURS = {"x": (230, 60, 60), "y": (60, 180, 60), "z": (60, 110, 230)}
_LABEL_COLOUR = (60, 60, 60)
_LABEL_STEP = 0.03         # spacing of sample points inside a glyph (glyph units)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def _normalise(values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    lo, hi = float(values.min()), float(values.max())
    span = (hi - lo) or 1.0
    return 2.0 * (values - lo) / span - 1.0, (lo, hi)


def _sphere_offsets(n: int) -> np.ndarray:
    """Roughly even points on a unit sphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / n)
    azimuth = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([
        np.cos(azimuth) * np.sin(polar),
        np.cos(polar),
        np.sin(azimuth) * np.sin(polar),
    ])


def _axes() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(-1.0, 1.0, _AXIS_POINTS)
    corner = np.full_like(t, -1.0)
    points = np.vstack([
        np.column_stack([t, corner, corner]),        # x-axis
        np.column_stack([corner, t, corner]),        # y-axis (vertical)
        np.column_stack([corner, corner, t]),        # z-axis
    ])
    colours = np.vstack([
        np.tile(_AXIS_COLOURS["x"], (_AXIS_POINTS, 1)),
        np.tile(_AXIS_COLOURS["y"], (_AXIS_POINTS, 1)),
        np.tile(_AXIS_COLOURS["z"], (_AXIS_POINTS, 1)),
    ])
    return points, colours


def _text_points(text: str, centre, along, up, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Points that spell ``text`` as a flat sheet centred on ``centre``.

    ``along`` is the 3D direction the text reads in, ``up`` the direction of
    its top, and ``height`` the glyph height in normalised units.
    """
    from matplotlib.textpath import TextPath

    path = TextPath((0.0, 0.0), text, size=1.0)
    vertices = path.vertices
    if len(vertices) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    xs = np.arange(lo[0], hi[0], _LABEL_STEP)
    ys = np.arange(lo[1], hi[1], _LABEL_STEP)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    local = grid[path.contains_points(grid)] - (lo + hi) / 2.0
    local *= height / max(hi[1] - lo[1], 1e-9)

    points = (np.asarray(centre, dtype=float)
              + local[:, :1] * np.asarray(along, dtype=float)
              + local[:, 1:] * np.asarray(up, dtype=float))
    colours = np.tile(_LABEL_COLOUR, (len(points), 1)).astype(np.uint8)
    return points, colours


def _axis_labels(x_name: str, x_range, z_name: str, z_range, y_range) -> tuple[np.ndarray, np.ndarray]:
    """Axis names and end values, placed just outside the box."""
    x_axis, z_axis = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    up = (0.0, 1.0, 0.0)
    pieces = [
        _text_points(x_name, (0.0, -1.35, -1.0), x_axis, up, 0.12),
        _text_points(z_name, (-1.25, -1.7, 0.0), z_axis, up, 0.12),
        _text_points("y", (-1.75, 0.0, -1.0), x_axis, up, 0.12),
        _text_points(f"{x_range[0]:.3g}", (-1.0, -1.25, -1.0), x_axis, up, 0.09),
        _text_points(f"{x_range[1]:.3g}", (1.0, -1.25, -1.0), x_axis, up, 0.09),
        _text_points(f"{z_range[0]:.3g}", (-1.25, -1.55, -1.0), z_axis, up, 0.09),
        _text_points(f"{z_range[1]:.3g}", (-1.25, -1.55, 1.0), z_axis, up, 0.09),
        _text_points(f"{y_range[0]:.3g}", (-1.45, -1.0, -1.0), x_axis, up, 0.09),
        _text_points(f"{y_range[1]:.3g}", (-1.45, 1.0, -1.0), x_axis, up, 0.09),
    ]
    points = np.vstack([p for p, _ in pieces])
    colours = np.vstack([c for _, c in pieces])
    return points, colours


def write_ply(path: Path, points: np.ndarray, colours: np.ndarray, comments: list[str]) -> Path:
    """Write an ASCII PLY point cloud with per-point RGB colours."""
    header = ["ply", "format ascii 1.0"]
    header += [f"comment {line}" for line in comments]
    header += [
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    with path.open("w") as handle:
        handle.write("\n".join(header) + "\n")
        for (x, y, z), (r, g, b) in zip(points, colours):
            handle.write(f"{x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)}\n")
    return path


def save_3d_plots(X: pd.DataFrame, y: np.ndarray, out_dir: Path, label: str) -> list[Path]:
    """Write one PLY file per pair of input columns. Returns the file paths."""
    import matplotlib

    out_dir.mkdir(parents=True, exist_ok=True)
    y_values = np.asarray(y, dtype=float)
    y_norm, y_range = _normalise(y_values)
    y_unit = (y_values - y_range[0]) / ((y_range[1] - y_range[0]) or 1.0)
    colours = (matplotlib.colormaps["viridis"](y_unit)[:, :3] * 255).astype(np.uint8)

    sphere = _sphere_offsets(_MARKER_POINTS) * _MARKER_RADIUS
    axis_points, axis_colours = _axes()
    paths = []

    for first, second in itertools.combinations(X.columns, 2):
        x_norm, x_range = _normalise(X[first].to_numpy(dtype=float))
        z_norm, z_range = _normalise(X[second].to_numpy(dtype=float))
        centres = np.column_stack([x_norm, y_norm, z_norm])

        markers = (centres[:, None, :] + sphere[None, :, :]).reshape(-1, 3)
        marker_colours = np.repeat(colours, _MARKER_POINTS, axis=0)

        label_points, label_colours = _axis_labels(first, x_range, second, z_range, y_range)
        points = np.vstack([markers, axis_points, label_points])
        all_colours = np.vstack([marker_colours, axis_colours, label_colours])

        comments = [
            f"{label}: y vs {first} and {second}",
            f"x axis (red): {first} in [{x_range[0]:.6g}, {x_range[1]:.6g}]",
            f"y axis (green, vertical): y in [{y_range[0]:.6g}, {y_range[1]:.6g}]",
            f"z axis (blue): {second} in [{z_range[0]:.6g}, {z_range[1]:.6g}]",
            "each axis is normalised to [-1, 1] for display",
        ]
        path = out_dir / f"{label}_y_vs_{safe_name(first)}_{safe_name(second)}.ply"
        paths.append(write_ply(path, points, all_colours, comments))

    return paths
