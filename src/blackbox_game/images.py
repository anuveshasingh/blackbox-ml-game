"""
images.py — Image puzzles: curated inputs, named transforms and pipelines.

Every image puzzle takes a curated 256×256 RGB PNG (see ``curated/``) and
applies either one named transform (``IMAGE_TRANSFORMS``) or a pipeline of
them (``PIPELINES``). ``render_image_puzzle`` writes two files for the player:

    input.png   the curated image (or, for fft_decode, its encoded spectrum)
    output.png  the transformed result

``play.py apply`` runs any sequence of ``IMAGE_TRANSFORMS`` on input.png.

Fourier puzzles use an 8-bit encoding of the spectrum:
    red   = log-magnitude of the fftshifted spectrum, scaled by LOG_CAP
    green = phase, mapped from [-π, π) to [0, 255]
    blue  = 0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

CURATED_DIR = Path(__file__).resolve().parent / "curated"

SOLARISE_THRESHOLD = 128       # pixels at or above this are inverted
POSTERISE_STEP = 64            # 256 // 64 = 4 levels per channel: 0, 85, 170, 255
OVERLAY_COUNT = 3              # times the image is overlaid on itself
OVERLAY_OPACITY = 0.5          # weight of the image already in the stack
OVERLAY_OFFSET = 4             # pixels shifted per overlay step, in x and in y
LOG_CAP = 17.0                 # log1p(|F|) is scaled by LOG_CAP / 255 before encoding


def load_curated(name: str) -> np.ndarray:
    """Load a curated image as a uint8 array of shape (256, 256, 3)."""
    return np.asarray(Image.open(CURATED_DIR / f"{name}.png").convert("RGB"), dtype=np.uint8)


def solarise(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    mask = img >= SOLARISE_THRESHOLD
    out[mask] = 255 - img[mask]
    return out


def posterise(img: np.ndarray) -> np.ndarray:
    levels = 256 // POSTERISE_STEP
    return ((img // POSTERISE_STEP) * (255 // (levels - 1))).astype(np.uint8)


def _paste_shifted(base: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """``base`` moved down by ``rows`` and right by ``cols`` with no wrap-around.

    Pixels the moved copy does not reach keep the base image's values.
    """
    out = base.copy()
    h, w = base.shape[:2]
    out[rows:, cols:] = base[: h - rows, : w - cols]
    return out


def repeated_overlay(img: np.ndarray) -> np.ndarray:
    out = img.astype(float)
    for k in range(1, OVERLAY_COUNT + 1):
        shift = OVERLAY_OFFSET * k
        shifted = _paste_shifted(out, shift, shift)
        out = OVERLAY_OPACITY * out + (1.0 - OVERLAY_OPACITY) * shifted
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def channel_shuffle(img: np.ndarray) -> np.ndarray:
    """(R, G, B) → (B, R, G) for every pixel."""
    return img[..., [2, 0, 1]].copy()


def fft_encode(img: np.ndarray) -> np.ndarray:
    gray = np.asarray(Image.fromarray(img).convert("L"), dtype=float)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.clip(np.log1p(np.abs(spectrum)) / LOG_CAP * 255.0, 0, 255)
    phase = (np.angle(spectrum) + np.pi) / (2.0 * np.pi) * 255.0
    out = np.zeros(gray.shape + (3,), dtype=np.uint8)
    out[..., 0] = np.round(magnitude)
    out[..., 1] = np.round(phase)
    return out


def fft_decode(encoded: np.ndarray) -> np.ndarray:
    magnitude = np.expm1(encoded[..., 0].astype(float) / 255.0 * LOG_CAP)
    phase = encoded[..., 1].astype(float) / 255.0 * 2.0 * np.pi - np.pi
    spectrum = np.fft.ifftshift(magnitude * np.exp(1j * phase))
    gray = np.clip(np.round(np.real(np.fft.ifft2(spectrum))), 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _input_image(image_name: str, transform: str) -> np.ndarray:
    """The picture the player is given: the curated image, or its spectrum for fft_decode."""
    source = load_curated(image_name)
    return fft_encode(source) if transform == "fft_decode" else source


def puzzle_input_image(puzzle) -> np.ndarray:
    """The input.png content for an image puzzle."""
    params = puzzle.function.parameters
    return _input_image(params["image"], params["transform"])


def _to_png(img: np.ndarray, path: Path) -> Path:
    Image.fromarray(img).save(path)
    return path


def render_image_puzzle(image_name: str, transform: str, out_dir: Path) -> list[Path]:
    """Write input.png and output.png for an image puzzle. Returns both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    input_img = _input_image(image_name, transform)
    if transform in PIPELINES:
        verify_pipeline(transform)
        output_img = apply_pipeline(input_img, PIPELINES[transform][1])
    else:
        output_img = IMAGE_TRANSFORMS[transform](input_img)
    return [
        _to_png(input_img, out_dir / "input.png"),
        _to_png(output_img, out_dir / "output.png"),
    ]


# ---------------------------------------------------------------------------
# More transforms, used alone or in pipelines
#
# All keep the canvas fixed and pivot about the image centre. ``rotate`` and
# ``stretch_horizontal`` fill pixels that sample from outside the image with
# black; ``ghost_echo`` and ``repeated_overlay`` leave the base image showing.
# ---------------------------------------------------------------------------

GHOST_SHIFT = 16               # echo offset in pixels (horizontal)
STRETCH_FACTOR = 1.6           # horizontal magnification about the centre
ROTATE_DEGREES = 15.0          # clockwise tilt (as seen on screen) about the centre
ECHO_WEIGHT = 0.5137           # blend weight of the image; the echo gets 1 - ECHO_WEIGHT
# 5-tap Gaussian, sigma 1. Not a binomial kernel: those weights are exact
# halves at the rounding step, which made inverted and blurred orders differ.
_gauss = np.exp(-np.arange(-2, 3) ** 2 / 2.0)
BLUR_KERNEL = _gauss / _gauss.sum()
CIRCULAR_SHIFT = (0, 32)       # (rows, columns) for np.roll
VIGNETTE_STRENGTH = 0.6        # brightness lost at the corners (0 = none, 1 = black)


def _ghost_echo(img: np.ndarray) -> np.ndarray:
    """Blend the image with a copy moved GHOST_SHIFT px right (no wrap-around)."""
    echo = _paste_shifted(img, 0, GHOST_SHIFT)
    mixed = ECHO_WEIGHT * img.astype(float) + (1.0 - ECHO_WEIGHT) * echo
    return np.clip(np.round(mixed), 0, 255).astype(np.uint8)


def _invert(img: np.ndarray) -> np.ndarray:
    return (255 - img).astype(np.uint8)


def _swap_rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return img[..., ::-1].copy()


def _flip_horizontal(img: np.ndarray) -> np.ndarray:
    return img[:, ::-1].copy()


def _sample_map(img: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Nearest-neighbour sampling of img at (rows, cols); out-of-range pixels are black."""
    h, w = img.shape[:2]
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out = np.zeros_like(img)
    out[inside] = img[rows[inside], cols[inside]]
    return out


def _stretch_horizontal(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    src_x = np.round(cx + (xs - cx) / STRETCH_FACTOR).astype(int)
    return _sample_map(img, ys, src_x)


def _rotate(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    theta = np.deg2rad(ROTATE_DEGREES)
    ys, xs = np.mgrid[0:h, 0:w]
    dy, dx = ys - cy, xs - cx
    src_y = np.round(cy + dy * np.cos(theta) - dx * np.sin(theta)).astype(int)
    src_x = np.round(cx + dy * np.sin(theta) + dx * np.cos(theta)).astype(int)
    return _sample_map(img, src_y, src_x)


def _gaussian_blur(img: np.ndarray) -> np.ndarray:
    """Separable 5-tap Gaussian with wrap-around, so it commutes with np.roll."""
    out = img.astype(float)
    k = BLUR_KERNEL
    for axis in (0, 1):
        acc = np.zeros_like(out)
        for i, weight in enumerate(k):
            acc += weight * np.roll(out, shift=i - 2, axis=axis)
        out = acc
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _circular_shift(img: np.ndarray) -> np.ndarray:
    return np.roll(img, shift=CIRCULAR_SHIFT, axis=(0, 1))


def _vignette(img: np.ndarray) -> np.ndarray:
    """Darken towards the corners. The falloff is centred on the image centre."""
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    radius = np.sqrt(((ys - cy) / (h / 2.0)) ** 2 + ((xs - cx) / (w / 2.0)) ** 2)
    factor = 1.0 - VIGNETTE_STRENGTH * np.clip(radius / np.sqrt(2.0), 0.0, 1.0) ** 2
    return np.clip(np.round(img * factor[..., None]), 0, 255).astype(np.uint8)


#: Every named image transform, usable with ``play.py apply`` and in pipelines.
#: Each takes and returns a 256×256 RGB uint8 array, so any sequence is valid.
IMAGE_TRANSFORMS = {
    "solarise": solarise,
    "posterise": posterise,
    "repeated_overlay": repeated_overlay,
    "channel_shuffle": channel_shuffle,
    "swap_rgb_bgr": _swap_rgb_to_bgr,
    "invert": _invert,
    "ghost_echo": _ghost_echo,
    "flip_horizontal": _flip_horizontal,
    "stretch_horizontal": _stretch_horizontal,
    "rotate": _rotate,
    "gaussian_blur": _gaussian_blur,
    "circular_shift": _circular_shift,
    "vignette": _vignette,
    "fft_encode": fft_encode,
    "fft_decode": fft_decode,
}


# ---------------------------------------------------------------------------
# Pipelines — several transforms applied left to right
# ---------------------------------------------------------------------------


def apply_pipeline(img: np.ndarray, steps: list[str]) -> np.ndarray:
    """Apply the named transforms left to right."""
    for step in steps:
        img = IMAGE_TRANSFORMS[step](img)
    return img


#: name → (curated image, ordered steps, commutative?)
PIPELINES = {
    "psychedelic_ghost": ("doctor_strange", ["ghost_echo", "swap_rgb_bgr", "invert"], True),
    "funhouse_pop_art": ("doctor_strange", ["stretch_horizontal", "flip_horizontal", "posterise"], True),
    "tilted_acid_trip": ("matrix", ["rotate", "solarise", "swap_rgb_bgr"], True),
    "dream_negative": ("istockphoto", ["gaussian_blur", "invert", "circular_shift"], True),
    "the_trap": ("matrix", ["ghost_echo", "vignette"], False),
}


def pipeline_orderings(steps: list[str]):
    from itertools import permutations
    return [list(p) for p in permutations(steps)]


def verify_pipeline(name: str) -> None:
    """Raise if a commutative pipeline gives different results in different orders."""
    image_name, steps, commutes = PIPELINES[name]
    if not commutes:
        return
    img = load_curated(image_name)
    reference = apply_pipeline(img, steps)
    for ordering in pipeline_orderings(steps):
        if not np.array_equal(apply_pipeline(img, ordering), reference):
            raise AssertionError(f"{name}: ordering {ordering} changes the result")
