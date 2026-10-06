"""
test_plots.py — 3D PLY plot files and the VS Code viewer fallback.
"""

import numpy as np
import pandas as pd

from blackbox_game import viewer
from blackbox_game.plots import save_3d_plots, _axis_labels, _MARKER_POINTS, _AXIS_POINTS


def _ply_vertex_count(path):
    with open(path) as handle:
        for line in handle:
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line.strip() == "end_header":
                break
    raise AssertionError("no vertex count in header")


def _sample(n_inputs=3, n=25):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(1.0, 5.0, (n, n_inputs)),
                     columns=[f"x{i}" for i in range(1, n_inputs + 1)])
    y = rng.uniform(0.0, 10.0, n)
    return X, y


class TestSave3DPlots:

    def test_one_file_per_pair_of_inputs(self, tmp_path):
        X, y = _sample(n_inputs=3)
        paths = save_3d_plots(X, y, tmp_path, "puzzle_10")
        assert len(paths) == 3  # C(3, 2)
        assert {p.name for p in paths} == {
            "puzzle_10_y_vs_x1_x2.ply",
            "puzzle_10_y_vs_x1_x3.ply",
            "puzzle_10_y_vs_x2_x3.ply",
        }

    def test_vertex_count_is_markers_axes_and_labels(self, tmp_path):
        X, y = _sample(n_inputs=2, n=25)
        (path,) = save_3d_plots(X, y, tmp_path, "p")
        x_range = (float(X["x1"].min()), float(X["x1"].max()))
        z_range = (float(X["x2"].min()), float(X["x2"].max()))
        y_range = (float(y.min()), float(y.max()))
        label_points, _ = _axis_labels("x1", x_range, "x2", z_range, y_range)
        expected = 25 * _MARKER_POINTS + 3 * _AXIS_POINTS + len(label_points)
        assert _ply_vertex_count(path) == expected

    def test_header_declares_ascii_ply_with_colours(self, tmp_path):
        X, y = _sample(n_inputs=2)
        (path,) = save_3d_plots(X, y, tmp_path, "p")
        lines = path.read_text().splitlines()
        assert lines[0] == "ply"
        assert lines[1] == "format ascii 1.0"
        assert "property uchar red" in lines
        assert lines[lines.index("end_header") - 1].startswith("property uchar blue")

    def test_colours_are_valid_bytes(self, tmp_path):
        X, y = _sample(n_inputs=2)
        (path,) = save_3d_plots(X, y, tmp_path, "p")
        body = path.read_text().split("end_header\n", 1)[1].splitlines()
        rgb = np.array([[int(v) for v in row.split()[3:]] for row in body if row])
        assert rgb.min() >= 0 and rgb.max() <= 255

    def test_single_input_gives_no_3d_files(self, tmp_path):
        X, y = _sample(n_inputs=1)
        assert save_3d_plots(X, y, tmp_path, "p") == []

    def test_header_records_original_ranges(self, tmp_path):
        X, y = _sample(n_inputs=2)
        (path,) = save_3d_plots(X, y, tmp_path, "p")
        text = path.read_text()
        assert f"{X['x1'].min():.6g}" in text or f"{X['x1'].min():.5g}" in text


class TestViewer:

    def test_missing_code_command_reports_no_vscode(self, monkeypatch):
        monkeypatch.setattr(viewer.shutil, "which", lambda name: None)
        assert viewer.ensure_viewer_extension() == "no-vscode"

    def test_open_without_code_command_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(viewer.shutil, "which", lambda name: None)
        assert viewer.open_in_vscode([tmp_path / "a.ply"]) is False
