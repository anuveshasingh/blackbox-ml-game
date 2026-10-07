//! A small anti-aliased raster canvas with text, enough to draw matplotlib-style plots.
//!
//! Text uses DejaVu Sans (matplotlib's default font), subset to the characters
//! the plots need and built into the binary.

use ab_glyph::{point, Font, FontRef, Glyph, OutlineCurve, PxScale, ScaleFont};
use std::sync::OnceLock;

pub type Colour = [f64; 3];

pub const BLACK: Colour = [0.0, 0.0, 0.0];

pub fn hex(rgb: u32) -> Colour {
    [(rgb >> 16) & 0xff, (rgb >> 8) & 0xff, rgb & 0xff].map(|c| c as f64 / 255.0)
}

pub fn font() -> &'static FontRef<'static> {
    static FONT: OnceLock<FontRef<'static>> = OnceLock::new();
    FONT.get_or_init(|| {
        FontRef::try_from_slice(include_bytes!("../assets/DejaVuSans-subset.ttf")).expect("embedded font")
    })
}

/// Glyphs of `text` laid out from x = 0 on the baseline, in font units of 1 em.
pub fn glyphs(text: &str) -> Vec<(ab_glyph::GlyphId, f64)> {
    let f = font();
    let upem = f.units_per_em().unwrap_or(2048.0) as f64;
    let mut x = 0.0;
    text.chars()
        .map(|ch| {
            let id = f.glyph_id(ch);
            let at = x;
            x += f.h_advance_unscaled(id) as f64 / upem;
            (id, at)
        })
        .collect()
}

/// Width of `text` in ems.
pub fn text_width_em(text: &str) -> f64 {
    let f = font();
    let upem = f.units_per_em().unwrap_or(2048.0) as f64;
    text.chars().map(|ch| f.h_advance_unscaled(f.glyph_id(ch)) as f64 / upem).sum()
}

/// The contours of `text` as polygons, in ems with y up, exactly as matplotlib's
/// `TextPath` builds them: FreeType outlines at 100 px per em in 1/64 px integer
/// units (curve points implied between two control points use FreeType's
/// truncating integer midpoint), advances rounded to 1/64 px, everything scaled
/// by 0.01, and each curve flattened by Agg at its default scale (two segments
/// through the curve's midpoint). Also returns every outline point, curve
/// control points included, for the bounding box.
pub fn text_polygons(text: &str) -> (Vec<Vec<(f64, f64)>>, Vec<(f64, f64)>) {
    let f = font();
    let upem = f.units_per_em().unwrap_or(2048.0) as f64;
    // FT_MulFix: font units → 1/64 px, rounding half away from zero.
    let fixed = |units: f32| (units as f64 * 6400.0 / upem).round() as i64;
    let mid = |a: (f64, f64), b: (f64, f64)| ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
    let mut polygons = Vec::new();
    let mut outline_points = Vec::new();
    let mut x = 0i64; // pen position in 1/64 px
    for ch in text.chars() {
        let id = f.glyph_id(ch);
        let x0 = x as f64 / 64.0;
        x += fixed(f.h_advance_unscaled(id));
        let Some(outline) = f.outline(id) else { continue };

        // Split the curves into contours.
        let mut contours: Vec<Vec<OutlineCurve>> = vec![];
        for curve in &outline.curves {
            let start = curve_start(curve);
            match contours.last_mut() {
                Some(c) if curve_end(c.last().expect("curve")) == start => c.push(curve.clone()),
                _ => contours.push(vec![curve.clone()]),
            }
        }

        let em = |p: (i64, i64)| ((p.0 as f64 / 64.0 + x0) * 0.01, p.1 as f64 / 64.0 * 0.01);
        let snap = |p: ab_glyph::Point| (fixed(p.x), fixed(p.y));
        for contour in contours {
            let n = contour.len();
            let control = |c: &OutlineCurve| match c {
                OutlineCurve::Quad(_, ctrl, _) => Some(*ctrl),
                _ => None,
            };
            // An on-curve point FreeType derives between two control points.
            let implied = |before: &OutlineCurve, after: &OutlineCurve| -> Option<(i64, i64)> {
                let (c1, c2) = (control(before)?, control(after)?);
                let p = curve_start(after);
                (p.x == (c1.x + c2.x) / 2.0 && p.y == (c1.y + c2.y) / 2.0).then(|| {
                    let (a, b) = (snap(c1), snap(c2));
                    ((a.0 + b.0) / 2, (a.1 + b.1) / 2)
                })
            };
            let starts: Vec<(i64, i64)> = (0..n)
                .map(|i| implied(&contour[(i + n - 1) % n], &contour[i]).unwrap_or_else(|| snap(curve_start(&contour[i]))))
                .collect();
            let mut poly = vec![em(starts[0])];
            outline_points.push(em(starts[0]));
            for (i, curve) in contour.iter().enumerate() {
                let (a, b) = (em(starts[i]), em(starts[(i + 1) % n]));
                match curve {
                    OutlineCurve::Line(..) => poly.push(b),
                    OutlineCurve::Quad(_, c, _) => {
                        let c = em(snap(*c));
                        poly.extend([mid(mid(a, c), mid(c, b)), b]);
                        outline_points.push(c);
                    }
                    OutlineCurve::Cubic(_, c1, c2, _) => {
                        let (c1, c2) = (em(snap(*c1)), em(snap(*c2)));
                        let (ab, bc, cd) = (mid(a, c1), mid(c1, c2), mid(c2, b));
                        poly.extend([mid(mid(ab, bc), mid(bc, cd)), b]);
                        outline_points.extend([c1, c2]);
                    }
                }
                outline_points.push(b);
            }
            if poly.len() > 2 {
                polygons.push(poly);
            }
        }
    }
    (polygons, outline_points)
}

fn curve_start(c: &OutlineCurve) -> ab_glyph::Point {
    match *c {
        OutlineCurve::Line(a, _) | OutlineCurve::Quad(a, _, _) | OutlineCurve::Cubic(a, _, _, _) => a,
    }
}

fn curve_end(c: &OutlineCurve) -> ab_glyph::Point {
    match *c {
        OutlineCurve::Line(_, b) | OutlineCurve::Quad(_, _, b) | OutlineCurve::Cubic(_, _, _, b) => b,
    }
}

/// matplotlib's `contains_points` for a text path: the crossings-multiply test
/// on each contour separately, and a point is inside if any contour contains it
/// (so the holes in letters such as "0" are filled).
pub fn inside(polygons: &[Vec<(f64, f64)>], tx: f64, ty: f64) -> bool {
    polygons.iter().any(|poly| {
        let mut odd = false;
        let (mut x0, mut y0) = poly[poly.len() - 1];
        let mut yflag0 = y0 >= ty;
        for &(x1, y1) in poly {
            let yflag1 = y1 >= ty;
            if yflag0 != yflag1 && (((y1 - ty) * (x0 - x1) >= (x1 - tx) * (y0 - y1)) == yflag1) {
                odd = !odd;
            }
            (x0, y0, yflag0) = (x1, y1, yflag1);
        }
        odd
    })
}

#[derive(Clone, Copy)]
pub enum Align {
    Start,
    Centre,
    End,
}

pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pixels: Vec<Colour>,
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Self {
        Canvas { width, height, pixels: vec![[1.0; 3]; width * height] }
    }

    pub fn blend(&mut self, x: i64, y: i64, colour: Colour, alpha: f64) {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 || alpha <= 0.0 {
            return;
        }
        let p = &mut self.pixels[y as usize * self.width + x as usize];
        for c in 0..3 {
            p[c] = p[c] * (1.0 - alpha) + colour[c] * alpha;
        }
    }

    /// Fill the rectangle [x0, x1) × [y0, y1) with anti-aliased edges.
    pub fn fill_rect(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, colour: Colour, alpha: f64) {
        let (x0, x1) = (x0.min(x1), x0.max(x1));
        let (y0, y1) = (y0.min(y1), y0.max(y1));
        for py in y0.floor() as i64..y1.ceil() as i64 {
            let cy = (y1.min(py as f64 + 1.0) - y0.max(py as f64)).max(0.0);
            for px in x0.floor() as i64..x1.ceil() as i64 {
                let cx = (x1.min(px as f64 + 1.0) - x0.max(px as f64)).max(0.0);
                self.blend(px, py, colour, alpha * cx * cy);
            }
        }
    }

    /// Fill the ring between radii `inner` and `outer` (inner 0 = a disc), anti-aliased.
    pub fn fill_ring(&mut self, cx: f64, cy: f64, inner: f64, outer: f64, colour: Colour, alpha: f64) {
        for py in (cy - outer - 1.0).floor() as i64..=(cy + outer + 1.0).ceil() as i64 {
            for px in (cx - outer - 1.0).floor() as i64..=(cx + outer + 1.0).ceil() as i64 {
                let d = ((px as f64 + 0.5 - cx).powi(2) + (py as f64 + 0.5 - cy).powi(2)).sqrt();
                let coverage = (outer - d + 0.5).clamp(0.0, 1.0) - if inner > 0.0 { (inner - d + 0.5).clamp(0.0, 1.0) } else { 0.0 };
                self.blend(px, py, colour, alpha * coverage);
            }
        }
    }

    /// Draw `text` at (x, y) in pixels. `size` is the font size (1 em) in pixels.
    /// Horizontal alignment is along the text; vertical alignment uses the font's
    /// ascent/descent box. `vertical` rotates the text 90° anticlockwise.
    pub fn text(&mut self, text: &str, x: f64, y: f64, size: f64, h: Align, v: Align, vertical: bool, colour: Colour) {
        let f = font();
        let upem = f.units_per_em().unwrap_or(2048.0) as f64;
        let scale = PxScale::from((size * f.height_unscaled() as f64 / upem) as f32);
        let scaled = f.as_scaled(scale);
        let width = text_width_em(text) * size;
        let (ascent, descent) = (scaled.ascent() as f64, scaled.descent() as f64);
        let along = match h {
            Align::Start => 0.0,
            Align::Centre => -width / 2.0,
            Align::End => -width,
        };
        let baseline = match v {
            Align::Start => ascent,
            Align::Centre => (ascent + descent) / 2.0,
            Align::End => descent,
        };

        for (id, at) in glyphs(text) {
            let glyph: Glyph = id.with_scale_and_position(scale, point((along + at * size) as f32, baseline as f32));
            let Some(outlined) = f.outline_glyph(glyph) else { continue };
            let bounds = outlined.px_bounds();
            outlined.draw(|gx, gy, coverage| {
                // (u, w): position in the unrotated text box, origin at the anchor.
                let u = bounds.min.x as f64 + gx as f64;
                let w = bounds.min.y as f64 + gy as f64;
                let (px, py) = if vertical { (x + w, y - u - 1.0) } else { (x + u, y + w) };
                self.blend(px.round() as i64, py.round() as i64, colour, coverage as f64);
            });
        }
    }

    pub fn to_rgb(&self) -> Vec<u8> {
        self.pixels.iter().flat_map(|p| p.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)).collect()
    }
}
