//! Image puzzles: curated inputs, named transforms and pipelines.
//!
//! Every image puzzle takes a curated 256×256 RGB PNG (see `assets/curated/`,
//! built into the binary) and applies either one named transform
//! (`IMAGE_TRANSFORMS`) or a pipeline of them (`PIPELINES`). A puzzle may show
//! one picture or several examples of the same transform. `render_image_puzzle`
//! writes, per picture:
//!
//!     input.png,   output.png     (one picture)
//!     input_N.png, output_N.png   (several pictures, N = 1, 2, ...)
//!
//! `apply` runs any sequence of `IMAGE_TRANSFORMS` on one JPEG/PNG the player
//! supplies (`load_user_image`). All transforms keep the canvas fixed at 256×256
//! and are deterministic. Rounding is round-half-to-even, as NumPy's `np.round`.

use std::path::{Path, PathBuf};

use image::{DynamicImage, ImageFormat, RgbImage};

use crate::puzzles::{Function, Puzzle};

pub const SIZE: usize = 256; // every image the game works on is SIZE × SIZE RGB

const CHUNK_SIZE: usize = 64; // rotate_chunks: square tiles of this many pixels (4 × 4 grid)
const CIRCULAR_SHIFT: (usize, usize) = (0, 32); // circular_shift: (rows, columns), wraps around
const GHOST_SHIFT: usize = 16; // ghost_echo: echo moved this many pixels right (no wrap)
const ECHO_WEIGHT: f64 = 0.5137; // ghost_echo: weight of the image; the echo gets 1 - ECHO_WEIGHT
const OPACITY: f64 = 0.6; // opacity: image weight; the rest is white (a faded picture)
const VIGNETTE_STRENGTH: f64 = 0.6; // vignette: brightness lost at the corners (0 = none, 1 = black)
const STRETCH_FACTOR: f64 = 1.6; // stretch_horizontal: magnification about the centre
const SOLARISE_THRESHOLD: u8 = 128; // solarise: values at or above this are inverted
const POSTERISE_STEP: u8 = 64; // posterise: 256 / 64 = 4 levels per channel: 0, 85, 170, 255

/// A SIZE × SIZE picture, row-major RGB.
pub type Picture = Vec<[u8; 3]>;

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

const CURATED: &[(&str, &[u8])] = &[
    ("checkmate", include_bytes!("../assets/curated/checkmate.png")),
    ("doctor_strange", include_bytes!("../assets/curated/doctor_strange.png")),
    ("lsd", include_bytes!("../assets/curated/lsd.png")),
    ("marbles", include_bytes!("../assets/curated/marbles.png")),
    ("matrix", include_bytes!("../assets/curated/matrix.png")),
    ("molecule", include_bytes!("../assets/curated/molecule.png")),
    ("monet", include_bytes!("../assets/curated/monet.png")),
    ("moon", include_bytes!("../assets/curated/moon.png")),
    ("pexels", include_bytes!("../assets/curated/pexels.png")),
];

fn to_picture(img: &RgbImage) -> Picture {
    img.pixels().map(|p| p.0).collect()
}

/// A curated picture, as a SIZE × SIZE RGB array.
pub fn load_curated(name: &str) -> Picture {
    let (_, bytes) = CURATED.iter().find(|(n, _)| *n == name).expect("curated picture");
    let img = image::load_from_memory_with_format(bytes, ImageFormat::Png).expect("curated PNG");
    to_picture(&img.to_rgb8())
}

/// File types accepted for a player's own picture (`apply --input`).
pub const USER_IMAGE_SUFFIXES: [&str; 3] = [".jpg", ".jpeg", ".png"];

fn format_name(format: ImageFormat) -> String {
    format.extensions_str().first().map_or_else(|| format!("{format:?}"), |e| e.to_uppercase())
}

/// Pillow's `alpha_composite` of one pixel onto opaque white.
fn over_white(r: u8, g: u8, b: u8, a: u8) -> [u8; 3] {
    if a == 0 {
        return [255, 255, 255];
    }
    const PRECISION_BITS: u32 = 7;
    let shift_div_255 = |v: u32| ((v >> 8) + v) >> 8;
    let a = a as u32;
    let outa255 = a * 255 + 255 * (255 - a);
    let coef1 = a * 255 * 255 * (1 << PRECISION_BITS) / outa255;
    let coef2 = 255 * (1 << PRECISION_BITS) - coef1;
    let blend = |src: u8| (shift_div_255(src as u32 * coef1 + 255 * coef2 + (0x80 << PRECISION_BITS)) >> PRECISION_BITS) as u8;
    [blend(r), blend(g), blend(b)]
}

/// Flatten any transparency onto white and return an RGB image.
fn to_rgb(img: DynamicImage) -> RgbImage {
    let high_byte = |v: u16| (v >> 8) as u8;
    match img {
        DynamicImage::ImageRgb8(rgb) => rgb,
        DynamicImage::ImageRgba8(rgba) => {
            let (w, h) = rgba.dimensions();
            RgbImage::from_fn(w, h, |x, y| {
                let [r, g, b, a] = rgba.get_pixel(x, y).0;
                image::Rgb(over_white(r, g, b, a))
            })
        }
        DynamicImage::ImageLumaA8(la) => {
            let (w, h) = la.dimensions();
            RgbImage::from_fn(w, h, |x, y| {
                let [l, a] = la.get_pixel(x, y).0;
                image::Rgb(over_white(l, l, l, a))
            })
        }
        DynamicImage::ImageRgb16(rgb) => {
            let (w, h) = rgb.dimensions();
            RgbImage::from_fn(w, h, |x, y| image::Rgb(rgb.get_pixel(x, y).0.map(high_byte)))
        }
        DynamicImage::ImageRgba16(rgba) => {
            let (w, h) = rgba.dimensions();
            RgbImage::from_fn(w, h, |x, y| {
                let [r, g, b, a] = rgba.get_pixel(x, y).0.map(high_byte);
                image::Rgb(over_white(r, g, b, a))
            })
        }
        DynamicImage::ImageLumaA16(la) => {
            let (w, h) = la.dimensions();
            RgbImage::from_fn(w, h, |x, y| {
                let [l, a] = la.get_pixel(x, y).0.map(high_byte);
                image::Rgb(over_white(l, l, l, a))
            })
        }
        other => other.to_rgb8(),
    }
}

/// Decode a JPEG with libjpeg-turbo's decoder, the one Pillow uses, so the
/// pixels match the Python game exactly. Like Pillow, a truncated file is an error.
fn decode_jpeg(bytes: &[u8]) -> Result<RgbImage, String> {
    use mozjpeg_sys::{jpeg_common_struct, jpeg_error_mgr, jpeg_std_error, JWRN_JPEG_EOF};
    use std::cell::Cell;
    use std::os::raw::c_int;

    thread_local!(static TRUNCATED: Cell<bool> = const { Cell::new(false) });

    extern "C-unwind" fn on_warning(cinfo: &mut jpeg_common_struct, level: c_int) {
        if level < 0 && unsafe { (*cinfo.err).msg_code } == JWRN_JPEG_EOF as c_int {
            TRUNCATED.set(true);
        }
    }
    extern "C-unwind" fn on_error(_cinfo: &mut jpeg_common_struct) {
        std::panic::resume_unwind(Box::new(()));
    }

    let mut err: jpeg_error_mgr = unsafe { std::mem::zeroed() };
    unsafe { jpeg_std_error(&mut err) };
    err.error_exit = Some(on_error);
    err.emit_message = Some(on_warning);
    TRUNCATED.set(false);

    let decoded = std::panic::catch_unwind(move || decode_jpeg_pixels(bytes, err));
    match decoded {
        _ if TRUNCATED.get() => Err("image file is truncated".into()),
        Ok(Ok(img)) => Ok(img),
        _ => Err("broken data stream when reading image file".into()),
    }
}

fn decode_jpeg_pixels(bytes: &[u8], err: mozjpeg_sys::jpeg_error_mgr) -> std::io::Result<RgbImage> {
    use mozjpeg::{ColorSpace, Decompress, Marker};
    let decompress = Decompress::with_err(err).with_markers(&[Marker::APP(14)]).from_mem(bytes)?;
    let (w, h) = (decompress.width() as u32, decompress.height() as u32);
    let pixels: Vec<u8> = match decompress.color_space() {
        ColorSpace::JCS_CMYK | ColorSpace::JCS_YCCK => {
            // Photoshop (Adobe marker) stores CMYK inverted; then Pillow's cmyk2rgb.
            let adobe = decompress.markers().any(|m| m.data.starts_with(b"Adobe"));
            let mut started = decompress.to_colorspace(ColorSpace::JCS_CMYK)?;
            let cmyk: Vec<[u8; 4]> = started.read_scanlines()?;
            started.finish()?;
            let muldiv255 = |a: i32, b: i32| {
                let t = a * b + 128;
                ((t >> 8) + t) >> 8
            };
            cmyk.into_iter()
                .flat_map(|p| {
                    let [c, m, y, k] = if adobe { p.map(|v| 255 - v) } else { p }.map(i32::from);
                    let nk = 255 - k;
                    [c, m, y].map(|v| (nk - muldiv255(v, nk)).clamp(0, 255) as u8)
                })
                .collect()
        }
        _ => {
            let mut started = decompress.rgb()?;
            let rgb: Vec<[u8; 3]> = started.read_scanlines()?;
            started.finish()?;
            rgb.into_iter().flatten().collect()
        }
    };
    RgbImage::from_raw(w, h, pixels).ok_or_else(|| std::io::Error::other("truncated JPEG"))
}

/// Load a player's JPEG or PNG, centre-cropped to a square and resized to SIZE.
pub fn load_user_image(path: &Path) -> Result<Picture, String> {
    let suffix = path.extension().map(|e| format!(".{}", e.to_string_lossy().to_lowercase())).unwrap_or_default();
    if !USER_IMAGE_SUFFIXES.contains(&suffix.as_str()) {
        return Err(format!("only {} files are accepted", USER_IMAGE_SUFFIXES.join(", ")));
    }
    let bytes = std::fs::read(path).map_err(|e| crate::io_error(&e, path))?;
    let format = image::guess_format(&bytes)
        .map_err(|_| format!("cannot identify image file '{}'", path.display()))?;
    if !matches!(format, ImageFormat::Jpeg | ImageFormat::Png) {
        return Err(format!("the file is {}, not a real JPEG or PNG", format_name(format)));
    }
    let img = if format == ImageFormat::Jpeg {
        decode_jpeg(&bytes)?
    } else {
        to_rgb(image::load_from_memory_with_format(&bytes, format).map_err(|e| e.to_string())?)
    };
    let (w, h) = img.dimensions();
    let s = w.min(h);
    let square = image::imageops::crop_imm(&img, (w - s) / 2, (h - s) / 2, s, s).to_image();
    let resized = if s as usize == SIZE { square } else { resize_lanczos(&square, SIZE) };
    Ok(to_picture(&resized))
}

// ---------------------------------------------------------------------------
// Pillow's LANCZOS resize (Resample.c), so player pictures match the Python game
// ---------------------------------------------------------------------------

const PRECISION_BITS: u32 = 32 - 8 - 2;

fn lanczos(x: f64) -> f64 {
    let sinc = |x: f64| {
        if x == 0.0 {
            1.0
        } else {
            let x = x * std::f64::consts::PI;
            x.sin() / x
        }
    };
    if (-3.0..3.0).contains(&x) { sinc(x) * sinc(x / 3.0) } else { 0.0 }
}

/// For each output pixel: (first input pixel, count, fixed-point weights).
fn coefficients(in_size: usize, out_size: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = in_size as f64 / out_size as f64;
    let filter_scale = scale.max(1.0);
    let support = 3.0 * filter_scale;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let ss = 1.0 / filter_scale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;
            let mut k: Vec<f64> = (0..xmax).map(|x| lanczos(((x + xmin) as f64 - center + 0.5) * ss)).collect();
            let ww: f64 = k.iter().sum();
            if ww != 0.0 {
                k.iter_mut().for_each(|w| *w /= ww);
            }
            let fixed = k
                .iter()
                .map(|&w| {
                    let w = w * (1u32 << PRECISION_BITS) as f64;
                    if w < 0.0 { (-0.5 + w) as i32 } else { (0.5 + w) as i32 }
                })
                .collect();
            (xmin, fixed)
        })
        .collect()
}

fn clip8(v: i64) -> u8 {
    if v >= (1i64 << PRECISION_BITS << 8) {
        255
    } else if v <= 0 {
        0
    } else {
        (v >> PRECISION_BITS) as u8
    }
}

fn resize_lanczos(src: &RgbImage, size: usize) -> RgbImage {
    let (w, h) = (src.width() as usize, src.height() as usize);
    let horiz = coefficients(w, size);
    let vert = coefficients(h, size);
    let first_row = vert[0].0;
    let last_row = vert[size - 1].0 + vert[size - 1].1.len();
    let px = |x: usize, y: usize| src.get_pixel(x as u32, y as u32).0;

    // Horizontal pass over only the rows the vertical pass reads.
    let mut temp = vec![[0u8; 3]; size * (last_row - first_row)];
    for y in first_row..last_row {
        for (xx, (xmin, k)) in horiz.iter().enumerate() {
            let mut ss = [1i64 << (PRECISION_BITS - 1); 3];
            for (x, &weight) in k.iter().enumerate() {
                let p = px(xmin + x, y);
                for c in 0..3 {
                    ss[c] += p[c] as i64 * weight as i64;
                }
            }
            temp[(y - first_row) * size + xx] = ss.map(clip8);
        }
    }
    let mut out = RgbImage::new(size as u32, size as u32);
    for (yy, (ymin, k)) in vert.iter().enumerate() {
        for xx in 0..size {
            let mut ss = [1i64 << (PRECISION_BITS - 1); 3];
            for (y, &weight) in k.iter().enumerate() {
                let p = temp[(ymin - first_row + y) * size + xx];
                for c in 0..3 {
                    ss[c] += p[c] as i64 * weight as i64;
                }
            }
            out.put_pixel(xx as u32, yy as u32, image::Rgb(ss.map(clip8)));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------

fn round(v: f64) -> u8 {
    v.round_ties_even().clamp(0.0, 255.0) as u8
}

fn map_values(img: &Picture, f: impl Fn(u8) -> u8) -> Picture {
    img.iter().map(|p| p.map(&f)).collect()
}

/// Cut into CHUNK_SIZE tiles and rotate each tile 90° clockwise in place.
fn rotate_chunks(img: &Picture) -> Picture {
    let mut out = img.clone();
    for top in (0..SIZE).step_by(CHUNK_SIZE) {
        for left in (0..SIZE).step_by(CHUNK_SIZE) {
            for r in 0..CHUNK_SIZE {
                for c in 0..CHUNK_SIZE {
                    out[(top + r) * SIZE + left + c] = img[(top + CHUNK_SIZE - 1 - c) * SIZE + left + r];
                }
            }
        }
    }
    out
}

/// Pixel-wise sum of the image and its flip about the central horizontal line, halved.
fn mirror_sum(img: &Picture) -> Picture {
    (0..SIZE * SIZE)
        .map(|i| {
            let (r, c) = (i / SIZE, i % SIZE);
            let (a, b) = (img[i], img[(SIZE - 1 - r) * SIZE + c]);
            [0, 1, 2].map(|k| round((a[k] as f64 + b[k] as f64) / 2.0))
        })
        .collect()
}

/// Move the image CIRCULAR_SHIFT pixels; what leaves one edge comes back at the other.
fn circular_shift(img: &Picture) -> Picture {
    let (dr, dc) = CIRCULAR_SHIFT;
    (0..SIZE * SIZE)
        .map(|i| {
            let (r, c) = (i / SIZE, i % SIZE);
            img[((r + SIZE - dr) % SIZE) * SIZE + (c + SIZE - dc) % SIZE]
        })
        .collect()
}

/// (R, G, B) → (R, B, G): green and blue exchange.
fn swap_rgb_rbg(img: &Picture) -> Picture {
    img.iter().map(|&[r, g, b]| [r, b, g]).collect()
}

/// (R, G, B) → (B, G, R): red and blue exchange.
fn swap_rgb_bgr(img: &Picture) -> Picture {
    img.iter().map(|&[r, g, b]| [b, g, r]).collect()
}

/// 255 − value on every channel.
fn invert(img: &Picture) -> Picture {
    map_values(img, |v| 255 - v)
}

/// Invert channel values at or above SOLARISE_THRESHOLD; darker values stay.
fn solarise(img: &Picture) -> Picture {
    map_values(img, |v| if v >= SOLARISE_THRESHOLD { 255 - v } else { v })
}

/// Reduce each channel to 4 levels: 0, 85, 170, 255.
fn posterise(img: &Picture) -> Picture {
    let levels = (256 / POSTERISE_STEP as u32) as u8;
    map_values(img, |v| (v / POSTERISE_STEP) * (255 / (levels - 1)))
}

/// Fade towards white: OPACITY × image + (1 − OPACITY) × white.
fn opacity(img: &Picture) -> Picture {
    map_values(img, |v| round(OPACITY * v as f64 + (1.0 - OPACITY) * 255.0))
}

/// Darken towards the corners. The falloff is centred on the image centre.
fn vignette(img: &Picture) -> Picture {
    let half = SIZE as f64 / 2.0;
    let centre = (SIZE as f64 - 1.0) / 2.0;
    (0..SIZE * SIZE)
        .map(|i| {
            let dy = (((i / SIZE) as f64 - centre) / half).powi(2);
            let dx = (((i % SIZE) as f64 - centre) / half).powi(2);
            let falloff = ((dy + dx).sqrt() / 2f64.sqrt()).clamp(0.0, 1.0);
            let factor = 1.0 - VIGNETTE_STRENGTH * (falloff * falloff);
            img[i].map(|v| round(v as f64 * factor))
        })
        .collect()
}

/// Blend the image with a copy moved GHOST_SHIFT px right. No wrap-around:
/// the leftmost GHOST_SHIFT columns keep the original image.
fn ghost_echo(img: &Picture) -> Picture {
    (0..SIZE * SIZE)
        .map(|i| {
            let echo = if i % SIZE >= GHOST_SHIFT { img[i - GHOST_SHIFT] } else { img[i] };
            [0, 1, 2].map(|k| round(ECHO_WEIGHT * img[i][k] as f64 + (1.0 - ECHO_WEIGHT) * echo[k] as f64))
        })
        .collect()
}

/// Horizontal magnification about the centre, nearest-neighbour.
fn stretch_horizontal(img: &Picture) -> Picture {
    let centre = (SIZE as f64 - 1.0) / 2.0;
    let source: Vec<usize> = (0..SIZE)
        .map(|x| ((centre + (x as f64 - centre) / STRETCH_FACTOR).round_ties_even() as i64).clamp(0, SIZE as i64 - 1) as usize)
        .collect();
    (0..SIZE * SIZE).map(|i| img[(i / SIZE) * SIZE + source[i % SIZE]]).collect()
}

/// Every named image transform, usable with `apply` and in pipelines.
/// Each takes and returns a SIZE × SIZE RGB picture, so any sequence is valid.
pub const IMAGE_TRANSFORMS: &[(&str, fn(&Picture) -> Picture)] = &[
    ("rotate_chunks", rotate_chunks),
    ("mirror_sum", mirror_sum),
    ("circular_shift", circular_shift),
    ("swap_rgb_rbg", swap_rgb_rbg),
    ("swap_rgb_bgr", swap_rgb_bgr),
    ("invert", invert),
    ("solarise", solarise),
    ("posterise", posterise),
    ("opacity", opacity),
    ("vignette", vignette),
    ("ghost_echo", ghost_echo),
    ("stretch_horizontal", stretch_horizontal),
];

pub fn is_image_transform(name: &str) -> bool {
    IMAGE_TRANSFORMS.iter().any(|(n, _)| *n == name)
}

/// Apply the named transforms left to right.
pub fn apply_pipeline(img: &Picture, steps: &[&str]) -> Picture {
    steps.iter().fold(img.clone(), |img, step| {
        let (_, f) = IMAGE_TRANSFORMS.iter().find(|(n, _)| n == step).expect("image transform");
        f(&img)
    })
}

// ---------------------------------------------------------------------------
// Pipelines — several transforms applied left to right
// ---------------------------------------------------------------------------

/// (name, curated image, ordered steps)
pub const PIPELINES: &[(&str, &str, &[&str])] = &[
    ("ghost_solarise", "matrix", &["ghost_echo", "solarise"]),
    ("stretch_poster_bgr", "doctor_strange", &["stretch_horizontal", "posterise", "swap_rgb_bgr"]),
    ("invert_shift_chunks", "pexels", &["invert", "circular_shift", "rotate_chunks"]),
];

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

pub fn save_png(img: &Picture, path: &Path) -> Result<PathBuf, String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| crate::io_error(&e, parent))?;
    }
    let raw: Vec<u8> = img.iter().flatten().copied().collect();
    RgbImage::from_raw(SIZE as u32, SIZE as u32, raw)
        .expect("picture size")
        .save_with_format(path, ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(path.to_path_buf())
}

/// Write the input/output pair for each of a puzzle's pictures. Returns the pairs.
pub fn render_image_puzzle(puzzle: &Puzzle, out_dir: &Path) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    let Function::Image { images, transform } = &puzzle.function else { unreachable!("not an image puzzle") };
    let steps: Vec<&str> = match PIPELINES.iter().find(|(n, _, _)| n == transform) {
        Some((_, _, steps)) => steps.to_vec(),
        None => vec![transform],
    };
    images
        .iter()
        .enumerate()
        .map(|(i, name)| {
            // "" for a single picture, "_1", "_2", ... otherwise.
            let suffix = if images.len() == 1 { String::new() } else { format!("_{}", i + 1) };
            let input = load_curated(name);
            Ok((
                save_png(&input, &out_dir.join(format!("input{suffix}.png")))?,
                save_png(&apply_pipeline(&input, &steps), &out_dir.join(format!("output{suffix}.png")))?,
            ))
        })
        .collect()
}
