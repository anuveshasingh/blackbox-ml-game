//! Plots for `show --plot`, `points --plot` and `residuals --plot`.
//!
//! PNG scatter plots follow matplotlib's default style (figure size, DPI, fonts,
//! tick placement, colours) so they look like the Python game's plots.
//!
//! 3D plots are ASCII PLY point clouds: the first input on the x-axis, the
//! second on the z-axis, and the output y on the vertical axis. Each data point
//! is a small sphere of points, coloured by y (viridis). Axes are coloured lines
//! (red = x, green = y, blue = z) from the lower corner of the box. Axis names
//! and end values are text made of points, because PLY has no text.
//! Coordinates are normalised to [-1, 1] per axis, so the original ranges are
//! written into the PLY header as comments. Open the files in VS Code with the
//! PLY viewer extension (see viewer.rs) to rotate them with the mouse.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::canvas::{hex, inside, text_polygons, text_width_em, Align, Canvas, Colour, BLACK};
use crate::fmt::g;
use crate::viridis::VIRIDIS;

// ---------------------------------------------------------------------------
// PNG scatter plots (matplotlib defaults: 150 dpi, DejaVu Sans 10 pt)
// ---------------------------------------------------------------------------

const DPI: f64 = 150.0;
const PT: f64 = DPI / 72.0; // pixels per point

const LABEL_SIZE: f64 = 10.0 * PT;
const TITLE_SIZE: f64 = 12.0 * PT;
const TICK_LENGTH: f64 = 3.5 * PT;
const TICK_PAD: f64 = 3.5 * PT;
const LABEL_PAD: f64 = 4.0 * PT;
const TITLE_PAD: f64 = 6.0 * PT;
const LINE_WIDTH: f64 = 0.8 * PT; // spines, ticks and grid lines
const LAYOUT_PAD: f64 = 1.08 * 10.0 * PT; // tight_layout(pad=1.08)
const MARGIN: f64 = 0.05; // axes.xmargin / axes.ymargin
const MARKER_RADIUS: f64 = 0.5 * 4.898_979_485_566_356 * PT; // scatter s=24 → sqrt(24) pt across
const MARKER_EDGE: f64 = 1.0 * PT;
const MARKER_ALPHA: f64 = 0.7;
const GRID_ALPHA: f64 = 0.3;

fn c0() -> Colour {
    hex(0x1f77b4)
}

fn grid_colour() -> Colour {
    hex(0xb0b0b0)
}

/// One scatter panel.
pub struct Panel<'a> {
    pub x: &'a [f64],
    pub y: &'a [f64],
    pub x_label: String,
    pub y_label: String,
    pub title: String,
    /// Draw a dashed black line at y = 0 (the residual plot).
    pub zero_line: bool,
}

/// Python's float `divmod`.
fn divmod(x: f64, y: f64) -> (f64, f64) {
    let mut m = x % y;
    let mut d = (x - m) / y;
    if m != 0.0 {
        if (y < 0.0) != (m < 0.0) {
            m += y;
            d -= 1.0;
        }
    } else {
        m = 0.0f64.copysign(y);
    }
    let floordiv = if d != 0.0 {
        let f = d.floor();
        if d - f > 0.5 { f + 1.0 } else { f }
    } else {
        0.0f64.copysign(x / y)
    };
    (floordiv, m)
}

/// matplotlib's `nonsingular`: widen an empty or tiny interval.
fn nonsingular(lo: f64, hi: f64, expander: f64, tiny: f64) -> (f64, f64) {
    let (mut lo, mut hi) = (lo, hi);
    if !lo.is_finite() || !hi.is_finite() {
        return (-expander, expander);
    }
    let maxabs = lo.abs().max(hi.abs());
    if maxabs < 1e6 * f64::MIN_POSITIVE {
        return (-expander, expander);
    }
    if hi - lo <= maxabs * tiny {
        if hi == 0.0 && lo == 0.0 {
            lo = -expander;
            hi = expander;
        } else {
            lo -= expander * lo.abs();
            hi += expander * hi.abs();
        }
    }
    (lo, hi)
}

/// matplotlib's `MaxNLocator(nbins='auto', steps=[1, 2, 2.5, 5, 10])`.
fn ticks(vmin: f64, vmax: f64, nbins: usize) -> Vec<f64> {
    let (vmin, vmax) = nonsingular(vmin, vmax, 1e-13, 1e-14);
    let nbins = nbins as f64;
    let dv = (vmax - vmin).abs();
    let meanv = (vmax + vmin) / 2.0;
    let offset = if meanv.abs() / dv < 100.0 { 0.0 } else { 10f64.powf(meanv.abs().log10().floor()).copysign(meanv) };
    let scale = 10f64.powf((dv / nbins).log10().floor());
    let (lo, hi) = (vmin - offset, vmax - offset);
    let steps: Vec<f64> = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0].iter().map(|s| s * scale).collect();
    let raw_step = (hi - lo) / nbins;
    let istep = steps.iter().position(|&s| s >= raw_step).unwrap_or(steps.len() - 1);

    let mut result = vec![];
    for &step in steps[..=istep].iter().rev() {
        let tol = if offset > 0.0 { (10f64.powf((offset / step).log10() - 12.0)).max(1e-10).min(0.4999) } else { 1e-10 };
        let best_vmin = divmod(lo, step).0 * step;
        let (d, m) = divmod(lo - best_vmin, step);
        let low = if (m / step - 1.0).abs() < tol { d + 1.0 } else { d };
        let (d, m) = divmod(hi - best_vmin, step);
        let high = if (m / step).abs() < tol { d } else { d + 1.0 };
        result = (low as i64..=high as i64).map(|i| i as f64 * step + best_vmin).collect();
        let visible = result.iter().filter(|&&t| t >= lo && t <= hi).count();
        if visible >= 2 {
            break;
        }
    }
    result.into_iter().map(|t| t + offset).collect()
}

/// matplotlib's `ScalarFormatter`: labels for `locs` and the "1eN" text, if any.
fn tick_labels(locs: &[f64], vmin: f64, vmax: f64) -> (Vec<String>, Option<String>) {
    let visible: Vec<f64> = locs.iter().copied().filter(|&t| t >= vmin && t <= vmax).collect();
    let max_abs = visible.iter().fold(0.0f64, |m, t| m.max(t.abs()));
    let oom = if max_abs == 0.0 { 0 } else { max_abs.log10().floor() as i32 };
    let oom = if oom <= -5 || oom >= 6 { oom } else { 0 };
    let scaled: Vec<f64> = locs.iter().map(|t| t / 10f64.powi(oom)).collect();

    let mut range = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - scaled.iter().cloned().fold(f64::INFINITY, f64::min);
    if range == 0.0 {
        range = scaled.iter().fold(0.0f64, |m, t| m.max(t.abs()));
    }
    if range == 0.0 {
        range = 1.0;
    }
    let range_oom = range.log10().floor() as i32;
    let thresh = 1e-3 * 10f64.powi(range_oom);
    let mut sigfigs = (3 - range_oom).max(0);
    while sigfigs >= 0 {
        let p = 10f64.powi(sigfigs);
        let worst = scaled.iter().map(|t| (t - (t * p).round_ties_even() / p).abs()).fold(0.0, f64::max);
        if worst < thresh { sigfigs -= 1 } else { break }
    }
    let decimals = (sigfigs + 1) as usize;
    let labels = scaled
        .iter()
        .map(|&t| {
            let t = if t.abs() < 1e-8 { 0.0 } else { t };
            format!("{:.*}", decimals, t).replace('-', "\u{2212}")
        })
        .collect();
    let offset_text = (oom != 0).then(|| format!("1e{oom}").replace('-', "\u{2212}"));
    (labels, offset_text)
}

fn data_limits(values: &[f64], include_zero: bool) -> (f64, f64) {
    let mut lo = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut hi = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if include_zero {
        lo = lo.min(0.0);
        hi = hi.max(0.0);
    }
    let (lo, hi) = nonsingular(lo, hi, 0.05, 1e-15);
    let span = hi - lo;
    (lo - MARGIN * span, hi + MARGIN * span)
}

struct Axis {
    lo: f64,
    hi: f64,
    ticks: Vec<f64>,
    labels: Vec<String>,
    offset_text: Option<String>,
}

impl Axis {
    fn new(limits: (f64, f64), length_px: f64, label_spacing_pt: f64) -> Self {
        let nbins = ((length_px / PT) / label_spacing_pt).floor().clamp(1.0, 9.0) as usize;
        let ticks = ticks(limits.0, limits.1, nbins);
        let (labels, offset_text) = tick_labels(&ticks, limits.0, limits.1);
        Axis { lo: limits.0, hi: limits.1, ticks, labels, offset_text }
    }

    fn visible(&self) -> impl Iterator<Item = (f64, &String)> {
        let tol = (self.hi - self.lo) * 1e-10;
        self.ticks.iter().zip(&self.labels).filter(move |(t, _)| **t >= self.lo - tol && **t <= self.hi + tol).map(|(t, l)| (*t, l))
    }

    fn widest_label(&self) -> f64 {
        self.visible().map(|(_, l)| text_width_em(l) * LABEL_SIZE).fold(0.0, f64::max)
    }
}

/// Draw one or more panels stacked vertically, then save the PNG.
pub fn save_scatter(path: &Path, panels: &[Panel], width_in: f64, height_in: f64) -> Result<PathBuf, String> {
    let (w, h) = ((width_in * DPI).round(), (height_in * DPI).round());
    let mut canvas = Canvas::new(w as usize, h as usize);
    // Text heights (in ems) that reproduce where tight_layout puts the axes.
    let (tick_h, label_h, title_h) = (1.0 * LABEL_SIZE, 1.03 * LABEL_SIZE, 0.76 * TITLE_SIZE);

    let limits: Vec<((f64, f64), (f64, f64))> =
        panels.iter().map(|p| (data_limits(p.x, false), data_limits(p.y, p.zero_line))).collect();

    // tight_layout: decorations around each axes box, found by two passes since
    // the tick labels depend on the box size and the box on the labels.
    let n = panels.len() as f64;
    let top_deco = title_h + TITLE_PAD;
    let bottom_deco = TICK_LENGTH + TICK_PAD + tick_h + LABEL_PAD + label_h;
    let mut left = LAYOUT_PAD + 3.0 * LABEL_SIZE;
    let mut right = w - LAYOUT_PAD - LABEL_SIZE;
    let panel_h = (h - 2.0 * LAYOUT_PAD - (n - 1.0) * LAYOUT_PAD - n * (top_deco + bottom_deco)) / n;
    let mut axes: Vec<(Axis, Axis)> = vec![];
    for _ in 0..3 {
        axes = limits
            .iter()
            .map(|&(xl, yl)| (Axis::new(xl, right - left, 3.0 * 10.0), Axis::new(yl, panel_h, 2.0 * 10.0)))
            .collect();
        let widest = axes.iter().map(|(_, ya)| ya.widest_label()).fold(0.0, f64::max);
        left = LAYOUT_PAD + label_h + LABEL_PAD + widest + TICK_PAD + TICK_LENGTH;
        // The last x tick label may stick out past the right edge of the box.
        let overflow = axes
            .iter()
            .filter_map(|(xa, _)| {
                let (t, l) = xa.visible().last()?;
                let x = left + (t - xa.lo) / (xa.hi - xa.lo) * (right - left);
                Some(x + text_width_em(l) * LABEL_SIZE / 2.0 - right)
            })
            .fold(0.0, f64::max);
        right = w - LAYOUT_PAD - overflow.max(0.0);
    }

    for (i, (panel, (xa, ya))) in panels.iter().zip(&axes).enumerate() {
        let top = LAYOUT_PAD + top_deco + i as f64 * (panel_h + top_deco + bottom_deco + LAYOUT_PAD);
        let bottom = top + panel_h;
        let px = |v: f64| left + (v - xa.lo) / (xa.hi - xa.lo) * (right - left);
        let py = |v: f64| bottom - (v - ya.lo) / (ya.hi - ya.lo) * panel_h;

        // Markers (zorder 1): face, then an edge in the same colour.
        for (&x, &y) in panel.x.iter().zip(panel.y) {
            let (cx, cy) = (px(x), py(y));
            canvas.fill_ring(cx, cy, 0.0, MARKER_RADIUS, c0(), MARKER_ALPHA);
            canvas.fill_ring(cx, cy, MARKER_RADIUS - MARKER_EDGE / 2.0, MARKER_RADIUS + MARKER_EDGE / 2.0, c0(), MARKER_ALPHA);
        }
        // Grid (zorder 1.5).
        for (t, _) in xa.visible() {
            let x = px(t);
            canvas.fill_rect(x - LINE_WIDTH / 2.0, top, x + LINE_WIDTH / 2.0, bottom, grid_colour(), GRID_ALPHA);
        }
        for (t, _) in ya.visible() {
            let y = py(t);
            canvas.fill_rect(left, y - LINE_WIDTH / 2.0, right, y + LINE_WIDTH / 2.0, grid_colour(), GRID_ALPHA);
        }
        // Dashed zero line (zorder 2): 3.7 on, 1.6 off, scaled by its 1 pt width.
        if panel.zero_line {
            let y = py(0.0);
            let (on, off) = (3.7 * PT, 1.6 * PT);
            let mut x = left;
            while x < right {
                canvas.fill_rect(x, y - PT / 2.0, (x + on).min(right), y + PT / 2.0, BLACK, 1.0);
                x += on + off;
            }
        }
        // Spines, ticks and tick labels.
        let half = LINE_WIDTH / 2.0;
        canvas.fill_rect(left - half, top - half, right + half, top + half, BLACK, 1.0);
        canvas.fill_rect(left - half, bottom - half, right + half, bottom + half, BLACK, 1.0);
        canvas.fill_rect(left - half, top - half, left + half, bottom + half, BLACK, 1.0);
        canvas.fill_rect(right - half, top - half, right + half, bottom + half, BLACK, 1.0);
        for (t, label) in xa.visible() {
            let x = px(t);
            canvas.fill_rect(x - half, bottom, x + half, bottom + TICK_LENGTH, BLACK, 1.0);
            canvas.text(label, x, bottom + TICK_LENGTH + TICK_PAD, LABEL_SIZE, Align::Centre, Align::Start, false, BLACK);
        }
        for (t, label) in ya.visible() {
            let y = py(t);
            canvas.fill_rect(left - TICK_LENGTH, y - half, left, y + half, BLACK, 1.0);
            canvas.text(label, left - TICK_LENGTH - TICK_PAD, y, LABEL_SIZE, Align::End, Align::Centre, false, BLACK);
        }
        if let Some(text) = &xa.offset_text {
            let y = bottom + TICK_LENGTH + TICK_PAD + tick_h;
            canvas.text(text, right, y, LABEL_SIZE, Align::End, Align::Start, false, BLACK);
        }
        if let Some(text) = &ya.offset_text {
            canvas.text(text, left, top - PT, LABEL_SIZE, Align::Start, Align::End, false, BLACK);
        }
        // Axis labels and title.
        let x_label_top = bottom + TICK_LENGTH + TICK_PAD + tick_h + LABEL_PAD;
        canvas.text(&panel.x_label, (left + right) / 2.0, x_label_top, LABEL_SIZE, Align::Centre, Align::Start, false, BLACK);
        let y_label_right = left - TICK_LENGTH - TICK_PAD - ya.widest_label() - LABEL_PAD;
        canvas.text(&panel.y_label, y_label_right, (top + bottom) / 2.0, LABEL_SIZE, Align::Centre, Align::End, true, BLACK);
        canvas.text(&panel.title, (left + right) / 2.0, top - TITLE_PAD, TITLE_SIZE, Align::Centre, Align::End, false, BLACK);
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| crate::io_error(&e, parent))?;
    }
    image::RgbImage::from_raw(w as u32, h as u32, canvas.to_rgb())
        .expect("canvas size")
        .save_with_format(path, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(path.to_path_buf())
}

// ---------------------------------------------------------------------------
// 3D PLY point clouds
// ---------------------------------------------------------------------------

const MARKER_POINTS: usize = 120; // points per data-point sphere
const SPHERE_RADIUS: f64 = 0.04; // sphere radius in normalised units
const AXIS_POINTS: usize = 80; // points per axis line
const AXIS_COLOURS: [[u8; 3]; 3] = [[230, 60, 60], [60, 180, 60], [60, 110, 230]]; // x, y, z
const LABEL_COLOUR: [u8; 3] = [60, 60, 60];
const LABEL_STEP: f64 = 0.03; // spacing of sample points inside a glyph (ems)

type Cloud = Vec<([f64; 3], [u8; 3])>;

pub fn safe_name(name: &str) -> String {
    let mut out = String::new();
    let mut in_run = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
            in_run = false;
        } else if !in_run {
            out.push('_');
            in_run = true;
        }
    }
    out
}

fn normalise(values: &[f64]) -> (Vec<f64>, (f64, f64)) {
    let lo = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = if hi - lo == 0.0 { 1.0 } else { hi - lo };
    (values.iter().map(|v| 2.0 * (v - lo) / span - 1.0).collect(), (lo, hi))
}

/// Roughly even points on a unit sphere (Fibonacci lattice).
fn sphere_offsets(n: usize) -> Vec<[f64; 3]> {
    let golden = std::f64::consts::PI * (1.0 + 5f64.sqrt());
    (0..n)
        .map(|k| {
            let i = k as f64 + 0.5;
            let polar = (1.0 - 2.0 * i / n as f64).acos();
            let azimuth = golden * i;
            [azimuth.cos() * polar.sin(), polar.cos(), azimuth.sin() * polar.sin()]
        })
        .collect()
}

/// `np.linspace(-1, 1, n)`.
fn linspace(n: usize) -> Vec<f64> {
    let step = 2.0 / (n - 1) as f64;
    (0..n).map(|i| if i == n - 1 { 1.0 } else { i as f64 * step + -1.0 }).collect()
}

fn axes() -> Cloud {
    let t = linspace(AXIS_POINTS);
    let mut cloud = vec![];
    for (axis, colour) in AXIS_COLOURS.iter().enumerate() {
        for &v in &t {
            let mut p = [-1.0; 3];
            p[axis] = v;
            cloud.push((p, *colour));
        }
    }
    cloud
}

/// Points that spell `text` as a flat sheet centred on `centre`. `along` is the
/// 3D direction the text reads in, `up` the direction of its top, and `height`
/// the glyph height in normalised units.
fn text_points(text: &str, centre: [f64; 3], along: [f64; 3], up: [f64; 3], height: f64) -> Cloud {
    let (polygons, outline_points) = text_polygons(text);
    // matplotlib's TextPath stores a (0, 0) vertex for each closed contour, so
    // its bounding box always contains the origin.
    let (mut lo, mut hi) = ([0.0f64; 2], [0.0f64; 2]);
    for &(x, y) in &outline_points {
        lo = [lo[0].min(x), lo[1].min(y)];
        hi = [hi[0].max(x), hi[1].max(y)];
    }
    if polygons.is_empty() {
        return vec![];
    }
    // np.arange(a, b, step): NumPy fills a + i * ((a + step) - a).
    let arange = |a: f64, b: f64| -> Vec<f64> {
        let n = ((b - a) / LABEL_STEP).ceil().max(0.0) as usize;
        let delta = (a + LABEL_STEP) - a;
        (0..n).map(|i| if i == 0 { a } else { a + i as f64 * delta }).collect()
    };
    let (xs, ys) = (arange(lo[0], hi[0]), arange(lo[1], hi[1]));
    let mid = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
    let scale = height / (hi[1] - lo[1]).max(1e-9);
    let mut cloud = vec![];
    for &y in &ys {
        for &x in &xs {
            if inside(&polygons, x, y) {
                let (u, v) = ((x - mid[0]) * scale, (y - mid[1]) * scale);
                cloud.push(([0, 1, 2].map(|k| centre[k] + u * along[k] + v * up[k]), LABEL_COLOUR));
            }
        }
    }
    cloud
}

/// Axis names and end values, placed just outside the box.
fn axis_labels(x_name: &str, x_range: (f64, f64), z_name: &str, z_range: (f64, f64), y_range: (f64, f64)) -> Cloud {
    let (x_axis, z_axis, up) = ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
    [
        text_points(x_name, [0.0, -1.35, -1.0], x_axis, up, 0.12),
        text_points(z_name, [-1.25, -1.7, 0.0], z_axis, up, 0.12),
        text_points("y", [-1.75, 0.0, -1.0], x_axis, up, 0.12),
        text_points(&g(x_range.0, 3), [-1.0, -1.25, -1.0], x_axis, up, 0.09),
        text_points(&g(x_range.1, 3), [1.0, -1.25, -1.0], x_axis, up, 0.09),
        text_points(&g(z_range.0, 3), [-1.25, -1.55, -1.0], z_axis, up, 0.09),
        text_points(&g(z_range.1, 3), [-1.25, -1.55, 1.0], z_axis, up, 0.09),
        text_points(&g(y_range.0, 3), [-1.45, -1.0, -1.0], x_axis, up, 0.09),
        text_points(&g(y_range.1, 3), [-1.45, 1.0, -1.0], x_axis, up, 0.09),
    ]
    .concat()
}

/// Write an ASCII PLY point cloud with per-point RGB colours.
fn write_ply(path: &Path, cloud: &Cloud, comments: &[String]) -> Result<PathBuf, String> {
    let mut out = String::from("ply\nformat ascii 1.0\n");
    for line in comments {
        let _ = writeln!(out, "comment {line}");
    }
    let _ = writeln!(out, "element vertex {}", cloud.len());
    for p in ["float x", "float y", "float z", "uchar red", "uchar green", "uchar blue"] {
        let _ = writeln!(out, "property {p}");
    }
    out.push_str("end_header\n");
    for ([x, y, z], [r, g, b]) in cloud {
        let _ = writeln!(out, "{x:.5} {y:.5} {z:.5} {r} {g} {b}");
    }
    std::fs::write(path, out).map_err(|e| crate::io_error(&e, path))?;
    Ok(path.to_path_buf())
}

/// Write one PLY file per pair of input columns. Returns the file paths.
pub fn save_3d_plots(columns: &[(&str, Vec<f64>)], y: &[f64], out_dir: &Path, label: &str) -> Result<Vec<PathBuf>, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| crate::io_error(&e, out_dir))?;
    let (y_norm, y_range) = normalise(y);
    let y_span = if y_range.1 - y_range.0 == 0.0 { 1.0 } else { y_range.1 - y_range.0 };
    let colours: Vec<[u8; 3]> = y
        .iter()
        .map(|v| {
            let index = ((v - y_range.0) / y_span * 256.0).clamp(0.0, 255.0) as usize;
            VIRIDIS[index.min(255)]
        })
        .collect();
    let sphere: Vec<[f64; 3]> = sphere_offsets(MARKER_POINTS).into_iter().map(|p| p.map(|c| c * SPHERE_RADIUS)).collect();

    let mut paths = vec![];
    for (i, (first, x_values)) in columns.iter().enumerate() {
        for (second, z_values) in &columns[i + 1..] {
            let (x_norm, x_range) = normalise(x_values);
            let (z_norm, z_range) = normalise(z_values);
            let mut cloud: Cloud = vec![];
            for k in 0..y.len() {
                let centre = [x_norm[k], y_norm[k], z_norm[k]];
                for offset in &sphere {
                    cloud.push(([0, 1, 2].map(|c| centre[c] + offset[c]), colours[k]));
                }
            }
            cloud.extend(axes());
            cloud.extend(axis_labels(first, x_range, second, z_range, y_range));

            let comments = [
                format!("{label}: y vs {first} and {second}"),
                format!("x axis (red): {first} in [{}, {}]", g(x_range.0, 6), g(x_range.1, 6)),
                format!("y axis (green, vertical): y in [{}, {}]", g(y_range.0, 6), g(y_range.1, 6)),
                format!("z axis (blue): {second} in [{}, {}]", g(z_range.0, 6), g(z_range.1, 6)),
                "each axis is normalised to [-1, 1] for display".to_string(),
            ];
            let path = out_dir.join(format!("{label}_y_vs_{}_{}.ply", safe_name(first), safe_name(second)));
            paths.push(write_ply(&path, &cloud, &comments)?);
        }
    }
    Ok(paths)
}


