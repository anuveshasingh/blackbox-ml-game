"""
puzzles.py — Puzzle catalogue.

Public numbers (puzzle_NN) are catalogue positions, 1 to N in registry order.
To change the game, edit the puzzle definitions below and the registry list.

Beginner — numerical, one obvious feature, no description
Physics  — numerical, two inputs, fixed values stated in the description
Images   — curated picture + one transform or a pipeline of transforms

Physical formulas live in ``physics.py``. Image transforms, pipelines and
curated pictures live in ``images.py`` and ``curated/``.
"""

from __future__ import annotations

from .images import PIPELINES
from .models import FunctionSpec, Puzzle


# ===========================================================================
# BEGINNER (difficulty = 1) — no description: the data is the whole puzzle
# ===========================================================================

def _beginner(id: str, type: str, input_features: list[str], **parameters) -> Puzzle:
    return Puzzle(
        id=id,
        difficulty=1,
        description="",
        input_features=input_features,
        function=FunctionSpec(type=type, parameters=parameters),
    )


_LINE_01 = _beginner("line_01", "linear", ["x"], slope=3.0, intercept=15.0, x_min=-5.0, x_max=5.0)
_LINE_02 = _beginner("line_02", "linear", ["x"], slope=-2.0, intercept=20.0, x_min=-5.0, x_max=5.0)
_SQUARE_01 = _beginner("square_01", "quadratic", ["x"], a=1.0, b=0.0, c=0.0, x_min=-5.0, x_max=5.0)
_SQRT_01 = _beginner("sqrt_01", "sqrt_fn", ["x"], a=2.0, x_min=0.5, x_max=25.0)
_LOG_01 = _beginner("log_01", "log_fn", ["x"], a=3.0, b=1.0, x_min=1.0, x_max=100.0)
_LINE_SIN_01 = _beginner(
    "line_sin_01", "linear_plus_sin", ["x"], slope=1.0, amplitude=2.0, x_min=-10.0, x_max=10.0,
)


# ===========================================================================
# PHYSICS (difficulty = 2)
# Exact outputs. Every quantity that is not an input is fixed, and the fixed
# values are stated in the description the way an exam problem would.
# ===========================================================================

def _physics(id: str, description: str, ranges: dict, fixed: dict, formula: str) -> Puzzle:
    return Puzzle(
        id=id,
        difficulty=2,
        description=description,
        input_features=list(ranges),
        function=FunctionSpec(
            type="physics",
            parameters={
                "formula": formula,
                "ranges": {name: list(bounds) for name, bounds in ranges.items()},
                "fixed": dict(fixed),
            },
        ),
    )


_PHYS_PROJECTILE_Y = _physics(
    "projectile_y",
    "A ball is thrown from the ground at a fixed speed of 10 m/s, at an angle theta "
    "to the horizontal. Record its height above the ground after time t.",
    ranges={"theta": (0.1, 1.4), "t": (0.5, 5.0)},
    fixed={"v0": 10.0},
    formula="projectile_y",
)

_PHYS_SHM_ENERGY = _physics(
    "shm_energy",
    "A 2 kg mass is on a spring with stiffness 4 N/m. The mass is displaced by x "
    "metres and moves with speed v m/s. Record the total mechanical energy.",
    ranges={"x": (0.1, 2.0), "v": (0.5, 5.0)},
    fixed={"k": 4.0, "m": 2.0},
    formula="shm_energy",
)

_PHYS_TRAVELLING_WAVE = _physics(
    "travelling_wave",
    "A wave has amplitude 5 m, wavenumber 1 rad/m, angular frequency 1 rad/s and zero "
    "phase. Record its displacement at position x metres and time t seconds.",
    ranges={"x": (0.0, 5.0), "t": (0.0, 5.0)},
    fixed={"A": 5.0, "k": 1.0, "omega": 1.0, "phi": 0.0},
    formula="travelling_wave",
)

_PHYS_COULOMB_2 = _physics(
    "coulomb_2",
    "Two point charges sit on a line: q1 = 2 μC and q2 = −3 μC. They are r1 and r2 "
    "metres from a measuring point. Record the electric potential at that point.",
    ranges={"r1": (1.0, 5.0), "r2": (1.0, 5.0)},
    fixed={"q1": 2e-6, "q2": -3e-6},
    formula="coulomb_2",
)


# ===========================================================================
# IMAGES (difficulty = 3)
# ===========================================================================

#: The only text shown for any image puzzle. Kept neutral on purpose: it must
#: not hint at the transform, the number of steps, or whether order matters.
_IMAGE_DESCRIPTION = "Work out what was done to each input picture to make its output picture."


def _image(id: str, images: list[str], transform: str) -> Puzzle:
    """``images`` are curated picture names: one picture, or several examples of the
    same transform. ``transform`` is a name in images.IMAGE_TRANSFORMS or images.PIPELINES."""
    return Puzzle(
        id=id,
        difficulty=3,
        description=_IMAGE_DESCRIPTION,
        input_features=[],
        function=FunctionSpec(type="image", parameters={"images": list(images), "transform": transform}),
    )


# Single transforms. Two pictures = two examples of the same puzzle.
_IMG_CHUNKS = _image("rotate_chunks", ["lsd", "checkmate"], "rotate_chunks")
_IMG_MIRROR = _image("mirror_sum", ["moon", "molecule"], "mirror_sum")
_IMG_SHIFT = _image("circular_shift", ["matrix"], "circular_shift")
_IMG_RBG = _image("swap_rgb_rbg", ["marbles", "monet"], "swap_rgb_rbg")

# Pipelines, in images.PIPELINES order.
_IMG_PIPELINES = [_image(name, [image], name) for name, (image, _) in PIPELINES.items()]


# ===========================================================================
# Registry — order here sets the public puzzle numbers
# ===========================================================================

PUZZLE_REGISTRY: dict[str, Puzzle] = {
    p.id: p
    for p in [
        _LINE_01, _LINE_02, _SQUARE_01, _SQRT_01, _LOG_01, _LINE_SIN_01,
        _PHYS_PROJECTILE_Y, _PHYS_SHM_ENERGY, _PHYS_TRAVELLING_WAVE, _PHYS_COULOMB_2,
        _IMG_CHUNKS, _IMG_MIRROR, _IMG_SHIFT, _IMG_RBG,
        *_IMG_PIPELINES,
    ]
}


def puzzle_number(puzzle_id: str) -> int:
    """Return the public number (the NN in puzzle_NN): the 1-based catalogue position."""
    for number, pid in enumerate(PUZZLE_REGISTRY, 1):
        if pid == puzzle_id:
            return number
    raise KeyError(f"Unknown puzzle ID: {puzzle_id}")


def puzzle_id_from_number(number: int) -> str | None:
    """Return the puzzle ID at a public number, or None if out of range."""
    ids = list(PUZZLE_REGISTRY)
    return ids[number - 1] if 1 <= number <= len(ids) else None


def get_puzzle(puzzle_id: str) -> Puzzle:
    """Return a puzzle by ID. Raises KeyError if not found."""
    if puzzle_id not in PUZZLE_REGISTRY:
        raise KeyError(f"Puzzle '{puzzle_id}' not found.")
    return PUZZLE_REGISTRY[puzzle_id]


def list_puzzles(difficulty: int | None = None) -> list[dict]:
    """Return puzzle metadata dicts, optionally for one difficulty (1, 2 or 3)."""
    puzzles = list(PUZZLE_REGISTRY.values())
    if difficulty is not None:
        puzzles = [p for p in puzzles if p.difficulty == difficulty]
    return [
        {
            "id": p.id,
            "difficulty": p.difficulty,
            "description": p.description,
            "input_features": p.input_features,
        }
        for p in puzzles
    ]
