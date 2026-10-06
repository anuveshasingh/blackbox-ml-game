"""
images.py — Round 3 image puzzles: curated inputs and deterministic transforms.

Every puzzle takes a curated 256×256 RGB PNG (see ``curated/``) and applies one
fixed transform. ``render_image_puzzle`` writes two files for the player:

    input.png   the curated image (or, for fft_decode, its encoded spectrum)
    output.png  the transformed result

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


def repeated_overlay(img: np.ndarray) -> np.ndarray:
    out = img.astype(float)
    for k in range(1, OVERLAY_COUNT + 1):
        shift = OVERLAY_OFFSET * k
        shifted = np.roll(out, shift=(shift, shift), axis=(0, 1))
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


def _to_png(img: np.ndarray, path: Path) -> Path:
    Image.fromarray(img).save(path)
    return path


def render_image_puzzle(image_name: str, transform: str, out_dir: Path) -> list[Path]:
    """Write input.png and output.png for an image puzzle. Returns both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    source = load_curated(image_name)
    if transform == "fft_decode":
        input_img = fft_encode(source)
        output_img = fft_decode(input_img)
    elif transform == "fft_encode":
        input_img = source
        output_img = fft_encode(source)
    else:
        input_img = source
        output_img = TRANSFORMS[transform](source)
    return [
        _to_png(input_img, out_dir / "input.png"),
        _to_png(output_img, out_dir / "output.png"),
    ]


TRANSFORMS = {
    "solarise": solarise,
    "posterise": posterise,
    "repeated_overlay": repeated_overlay,
    "channel_shuffle": channel_shuffle,
}
