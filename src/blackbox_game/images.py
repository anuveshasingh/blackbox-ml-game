"""
images.py — Image puzzles: curated inputs, named transforms and pipelines.

Every image puzzle takes a curated 256×256 RGB PNG (see ``curated/``) and
applies either one named transform (``IMAGE_TRANSFORMS``) or a pipeline of
them (``PIPELINES``). A puzzle may show one picture or several examples of the
same transform. ``render_image_puzzle`` writes, per picture:

    input.png,   output.png     (one picture)
    input_N.png, output_N.png   (several pictures, N = 1, 2, ...)

``play.py apply`` runs any sequence of ``IMAGE_TRANSFORMS`` on one JPEG/PNG
the player supplies (``load_user_image``).
All transforms keep the canvas fixed at 256×256 and are deterministic.
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image

CURATED_DIR = Path(__file__).resolve().parent / "curated"
SIZE = 256                     # every image the game works on is SIZE × SIZE RGB

CHUNK_SIZE = 64                # rotate_chunks: square tiles of this many pixels (4 × 4 grid)
CIRCULAR_SHIFT = (0, 32)       # circular_shift: (rows, columns) for np.roll, wraps around
GHOST_SHIFT = 16               # ghost_echo: echo moved this many pixels right (no wrap)
ECHO_WEIGHT = 0.5137           # ghost_echo: weight of the image; the echo gets 1 - ECHO_WEIGHT
OPACITY = 0.6                  # opacity: image weight; the rest is white (a faded picture)
VIGNETTE_STRENGTH = 0.6        # vignette: brightness lost at the corners (0 = none, 1 = black)
STRETCH_FACTOR = 1.6           # stretch_horizontal: magnification about the centre
SOLARISE_THRESHOLD = 128       # solarise: values at or above this are inverted
POSTERISE_STEP = 64            # posterise: 256 // 64 = 4 levels per channel: 0, 85, 170, 255


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _to_rgb(im: Image.Image) -> Image.Image:
    """Flatten any transparency onto white and return an RGB image."""
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        base = Image.new("RGBA", im.size, (255, 255, 255, 255))
        base.alpha_composite(im)
        im = base
    return im.convert("RGB")


def load_curated(name: str) -> np.ndarray:
    """Load a curated image as a uint8 array of shape (SIZE, SIZE, 3)."""
    return np.asarray(Image.open(CURATED_DIR / f"{name}.png").convert("RGB"), dtype=np.uint8)


#: File types accepted for a player's own picture (``play.py apply --input``).
USER_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def load_user_image(path: str | Path) -> np.ndarray:
    """Load a player's JPEG or PNG, centre-cropped to a square and resized to SIZE."""
    if Path(path).suffix.lower() not in USER_IMAGE_SUFFIXES:
        raise ValueError(f"only {', '.join(USER_IMAGE_SUFFIXES)} files are accepted")
    with Image.open(path) as opened:
        if opened.format not in ("JPEG", "PNG"):
            raise ValueError(f"the file is {opened.format}, not a real JPEG or PNG")
        im = _to_rgb(opened)
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(im.resize((SIZE, SIZE), Image.LANCZOS), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _round(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def rotate_chunks(img: np.ndarray) -> np.ndarray:
    """Cut into CHUNK_SIZE tiles and rotate each tile 90° clockwise in place."""
    out = img.copy()
    h, w = img.shape[:2]
    for top in range(0, h, CHUNK_SIZE):
        for left in range(0, w, CHUNK_SIZE):
            tile = img[top:top + CHUNK_SIZE, left:left + CHUNK_SIZE]
            out[top:top + CHUNK_SIZE, left:left + CHUNK_SIZE] = np.rot90(tile, k=-1)
    return out


def mirror_sum(img: np.ndarray) -> np.ndarray:
    """Pixel-wise sum of the image and its flip about the central horizontal line, halved."""
    return _round((img.astype(float) + img[::-1].astype(float)) / 2.0)


def circular_shift(img: np.ndarray) -> np.ndarray:
    """Move the image CIRCULAR_SHIFT pixels; what leaves one edge comes back at the other."""
    return np.roll(img, shift=CIRCULAR_SHIFT, axis=(0, 1))


def swap_rgb_rbg(img: np.ndarray) -> np.ndarray:
    """(R, G, B) → (R, B, G): green and blue exchange."""
    return img[..., [0, 2, 1]].copy()


def swap_rgb_bgr(img: np.ndarray) -> np.ndarray:
    """(R, G, B) → (B, G, R): red and blue exchange."""
    return img[..., ::-1].copy()


def invert(img: np.ndarray) -> np.ndarray:
    """255 − value on every channel."""
    return (255 - img).astype(np.uint8)


def solarise(img: np.ndarray) -> np.ndarray:
    """Invert channel values at or above SOLARISE_THRESHOLD; darker values stay."""
    out = img.copy()
    mask = img >= SOLARISE_THRESHOLD
    out[mask] = 255 - img[mask]
    return out


def posterise(img: np.ndarray) -> np.ndarray:
    """Reduce each channel to 4 levels: 0, 85, 170, 255."""
    levels = 256 // POSTERISE_STEP
    return ((img // POSTERISE_STEP) * (255 // (levels - 1))).astype(np.uint8)


def opacity(img: np.ndarray) -> np.ndarray:
    """Fade towards white: OPACITY × image + (1 − OPACITY) × white."""
    return _round(OPACITY * img.astype(float) + (1.0 - OPACITY) * 255.0)


def vignette(img: np.ndarray) -> np.ndarray:
    """Darken towards the corners. The falloff is centred on the image centre."""
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    radius = np.sqrt(((ys - cy) / (h / 2.0)) ** 2 + ((xs - cx) / (w / 2.0)) ** 2)
    factor = 1.0 - VIGNETTE_STRENGTH * np.clip(radius / np.sqrt(2.0), 0.0, 1.0) ** 2
    return _round(img * factor[..., None])


def ghost_echo(img: np.ndarray) -> np.ndarray:
    """Blend the image with a copy moved GHOST_SHIFT px right. No wrap-around:
    the leftmost GHOST_SHIFT columns keep the original image."""
    echo = img.copy()
    echo[:, GHOST_SHIFT:] = img[:, :-GHOST_SHIFT]
    return _round(ECHO_WEIGHT * img.astype(float) + (1.0 - ECHO_WEIGHT) * echo)


def stretch_horizontal(img: np.ndarray) -> np.ndarray:
    """Horizontal magnification about the centre, nearest-neighbour."""
    w = img.shape[1]
    cx = (w - 1) / 2.0
    xs = np.arange(w)
    src = np.clip(np.round(cx + (xs - cx) / STRETCH_FACTOR).astype(int), 0, w - 1)
    return img[:, src].copy()


#: Every named image transform, usable with ``play.py apply`` and in pipelines.
#: Each takes and returns a SIZE × SIZE RGB uint8 array, so any sequence is valid.
IMAGE_TRANSFORMS = {
    "rotate_chunks": rotate_chunks,
    "mirror_sum": mirror_sum,
    "circular_shift": circular_shift,
    "swap_rgb_rbg": swap_rgb_rbg,
    "swap_rgb_bgr": swap_rgb_bgr,
    "invert": invert,
    "solarise": solarise,
    "posterise": posterise,
    "opacity": opacity,
    "vignette": vignette,
    "ghost_echo": ghost_echo,
    "stretch_horizontal": stretch_horizontal,
}


def apply_pipeline(img: np.ndarray, steps: list[str]) -> np.ndarray:
    """Apply the named transforms left to right."""
    for step in steps:
        img = IMAGE_TRANSFORMS[step](img)
    return img


# ---------------------------------------------------------------------------
# Pipelines — several transforms applied left to right
# ---------------------------------------------------------------------------

#: name → (curated image, ordered steps)
PIPELINES = {
    "ghost_solarise": ("matrix", ["ghost_echo", "solarise"]),
    "stretch_poster_bgr": ("doctor_strange", ["stretch_horizontal", "posterise", "swap_rgb_bgr"]),
    "invert_shift_chunks": ("pexels", ["invert", "circular_shift", "rotate_chunks"]),
}


def pipeline_commutes(name: str) -> bool:
    """True if every ordering of the pipeline's steps gives the same picture."""
    image_name, steps = PIPELINES[name]
    img = load_curated(image_name)
    reference = apply_pipeline(img, steps)
    return all(
        np.array_equal(apply_pipeline(img, list(order)), reference)
        for order in permutations(steps)
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def picture_suffixes(count: int) -> list[str]:
    """File-name suffix per picture: "" for a single picture, "_1", "_2", ... otherwise."""
    return [""] if count == 1 else [f"_{i}" for i in range(1, count + 1)]


def puzzle_pictures(puzzle) -> list[tuple[str, np.ndarray]]:
    """(suffix, input picture) for each example picture of an image puzzle."""
    names = puzzle.function.parameters["images"]
    return list(zip(picture_suffixes(len(names)), (load_curated(n) for n in names)))


def save_png(img: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return path


def render_image_puzzle(puzzle, out_dir: Path) -> list[tuple[Path, Path]]:
    """Write the input/output pair for each of a puzzle's pictures. Returns the pairs."""
    transform = puzzle.function.parameters["transform"]
    steps = PIPELINES[transform][1] if transform in PIPELINES else [transform]
    pairs = []
    for suffix, input_img in puzzle_pictures(puzzle):
        pairs.append((
            save_png(input_img, out_dir / f"input{suffix}.png"),
            save_png(apply_pipeline(input_img, steps), out_dir / f"output{suffix}.png"),
        ))
    return pairs
