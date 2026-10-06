"""
puzzles.py — Puzzle catalogue.

Public numbers (puzzle_NN) are the catalogue positions: 1 to 16, in registry order.

Beginner — puzzle_01 to puzzle_06
Physics — puzzle_07 to puzzle_10
Images — puzzle_11 to puzzle_16

Physical constants and formulas live in ``physics.py``.
Image transforms live in ``images.py``.
"""

from __future__ import annotations

from .models import Puzzle, FunctionSpec


def _p(**kwargs) -> Puzzle:
    """Thin wrapper — ensures hints defaults to [] when not supplied."""
    kwargs.setdefault("hints", [])
    return Puzzle(**kwargs)


# ===========================================================================
# BEGINNER (difficulty = 1)
# ===========================================================================

_LINE_01 = _p(
    id="line_01",
    title="Obedient Numbers",
    category="linear",
    difficulty=1,
    description=(
        "A machine receives a number and returns another number. "
        "The more you put in, the more you get back. "
        "The relationship seems almost too simple. Can you name it?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="linear",
        parameters={"slope": 3.0, "intercept": 15.0, "x_min": -5.0, "x_max": 5.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "square", "sqrt", "sin", "log"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["identity:x"],
    explanation=(
        "The rule was y = 3x + 15. "
        "Using x directly (identity transform), linear regression finds a "
        "perfect straight line with slope 3. "
        "R² = 1 means the line explains every single data point. "
        "This is the simplest relationship a model can discover."
    ),
    real_world_connection="Distance = speed × time — the original straight line.",
)

_LINE_02 = _p(
    id="line_02",
    title="The Reluctant Ascent",
    category="linear",
    difficulty=1,
    description=(
        "As x grows, y shrinks. The relationship is still perfectly "
        "predictable — just going the wrong way. Can a linear model handle this?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="linear",
        parameters={"slope": -2.0, "intercept": 20.0, "x_min": -5.0, "x_max": 5.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "square", "sqrt", "sin"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["identity:x"],
    explanation=(
        "The rule was y = −2x + 20. "
        "Linear regression handles negative slopes just as well as positive ones. "
        "The learned coefficient will simply be negative. "
        "'Linear' only means straight line — not necessarily going up."
    ),
    real_world_connection=(
        "As a car uses fuel, its tank level drops linearly with distance driven."
    ),
)

_SQUARE_01 = _p(
    id="square_01",
    title="The Bend in the Road",
    category="feature_transform",
    difficulty=1,
    description=(
        "The output y grows, but faster and faster as x increases. "
        "Something is curving. A plain straight line refuses to fit. "
        "What transformation straightens it out?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="quadratic",
        parameters={"a": 1.0, "b": 0.0, "c": 0.0, "x_min": -5.0, "x_max": 5.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "square", "sqrt", "sin", "log"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["square:x"],
    explanation=(
        "The rule was y = x². "
        "On x ∈ [−5, 5], raw x has near-zero correlation with x² "
        "(odd vs even function), so linear regression on x gives R² ≈ 0. "
        "But x² perfectly straightens the parabola — R² = 1. "
        "This is feature engineering: create a better input from what you have."
    ),
    real_world_connection="Kinetic energy = ½mv² — energy grows as the square of speed.",
)

_SQRT_01 = _p(
    id="sqrt_01",
    title="Momentum Decay",
    category="feature_transform",
    difficulty=1,
    description=(
        "Each extra unit of x adds less and less to y. "
        "Growth that slows down — what function does that?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="sqrt_fn",
        parameters={"a": 2.0, "x_min": 0.5, "x_max": 25.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "square", "sqrt", "log", "reciprocal"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["sqrt:x"],
    explanation=(
        "The rule was y = 2√x. "
        "√x grows quickly at first then slows — exactly the 'diminishing returns' shape. "
        "Plotting y vs √x reveals a perfect straight line. "
        "Linear regression on √x finds R² = 1."
    ),
    real_world_connection=(
        "The loudness you perceive from a speaker drops as √(distance)."
    ),
)

_LOG_01 = _p(
    id="log_01",
    title="The Compressed Universe",
    category="feature_transform",
    difficulty=1,
    description=(
        "x ranges from 1 to 100 but y barely budges. "
        "Going from x=1 to x=10 changes y as much as from x=10 to x=100. "
        "What function compresses a huge range into something manageable?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="log_fn",
        parameters={"a": 3.0, "b": 1.0, "x_min": 1.0, "x_max": 100.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "square", "sqrt", "log", "reciprocal"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["log:x"],
    explanation=(
        "The rule was y = 3·log(x) + 1. "
        "Raw x vs y is an almost-flat curve. log(x) compresses large x values "
        "and stretches small ones — turning the curve into a straight line. "
        "After applying log, R² = 1."
    ),
    real_world_connection=(
        "The Richter earthquake scale is logarithmic: "
        "magnitude 7 is 10× stronger than magnitude 6."
    ),
)

_DISTRACTOR_01 = _p(
    id="distractor_01",
    title="Four Suspects",
    category="distractor",
    difficulty=1,
    description=(
        "Four features walk into a room. One of them controls y. "
        "The other three are pure noise — completely unrelated. "
        "Which one is the culprit?"
    ),
    input_features=["x1", "x2", "x3", "x4"],
    function=FunctionSpec(
        type="linear_distractor",
        parameters={"slope": 3.0, "intercept": 0.0, "x_min": -5.0, "x_max": 5.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["identity:x1"],
    explanation=(
        "The rule was y = 3·x1. "
        "x2, x3, x4 are random numbers with no connection to y. "
        "Evaluating each feature separately (R² per feature) quickly reveals "
        "that only x1 gives R² ≈ 1. "
        "Not all data is useful. Learning to ignore distractors is a core ML skill."
    ),
    real_world_connection=(
        "Doctors measure hundreds of biomarkers; only a few "
        "predict a given disease."
    ),
)


# ===========================================================================
# PHYSICS (difficulty = 2)
# Exact, noise-free outputs. Formulas and constants live in physics.py.
# Fixed values are stated in each description, the way exam problems do.
# ===========================================================================

_PHYSICS_TRANSFORMS = ["identity", "square", "sqrt", "reciprocal", "sin", "cos", "exp", "exp_neg"]
_PHYSICS_BINARY = ["multiply", "divide", "add", "subtract"]
_PROJECTILE_RANGES = {"theta": (0.1, 1.4), "t": (0.5, 5.0)}

def _physics(*, id, title, description, ranges, fixed, formula, solution_features,
             explanation, connection) -> Puzzle:
    return _p(
        id=id,
        title=title,
        category="physics",
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
        noise_std=0.0,
        allowed_transforms=_PHYSICS_TRANSFORMS,
        allowed_binary_transforms=_PHYSICS_BINARY,
        intended_model="linear_regression",
        solution_features=solution_features,
        explanation=explanation,
        real_world_connection=connection,
    )


_PHYS_PROJECTILE_Y = _physics(
    id="projectile_y",
    title="Vertical Climb",
    description=(
        "A ball is thrown from the ground at a fixed speed of 10 m/s, at an angle theta "
        "to the horizontal. Record its height above the ground after time t."
    ),
    ranges=_PROJECTILE_RANGES,
    fixed={"v0": 10.0},
    formula="projectile_y",
    solution_features=[{"product": ["t", "sin:theta"]}, "square:t"],
    explanation=(
        "The rule was y = v0·sin(θ)·t − ½·g·t² with v0 = 10. Two features are needed: "
        "t·sin(θ) and t². Linear regression finds the coefficients 10 and −g/2."
    ),
    connection="Vertical height of a projectile under gravity.",
)


_PHYS_SHM_ENERGY = _physics(
    id="shm_energy",
    title="Stored and Moving",
    description=(
        "A 2 kg mass is on a spring with stiffness 4 N/m. The mass is displaced by x "
        "metres and moves with speed v m/s. Record the total mechanical energy."
    ),
    ranges={"x": (0.1, 2.0), "v": (0.5, 5.0)},
    fixed={"k": 4.0, "m": 2.0},
    formula="shm_energy",
    solution_features=["square:x", "square:v"],
    explanation=(
        "The rule was y = ½·k·x² + ½·m·v² with k = 4 and m = 2, so y = 2x² + v². "
        "The squares x² and v² are the features, and the fixed constants are the coefficients."
    ),
    connection="Total mechanical energy of a mass on a spring (simple harmonic motion).",
)


_PHYS_TRAVELLING_WAVE = _physics(
    id="travelling_wave",
    title="Rippling Line",
    description=(
        "A wave has amplitude 5 m, wavenumber 1 rad/m, angular frequency 1 rad/s and zero "
        "phase. Record its displacement at position x metres and time t seconds."
    ),
    ranges={"x": (0.0, 5.0), "t": (0.0, 5.0)},
    fixed={"A": 5.0, "k": 1.0, "omega": 1.0, "phi": 0.0},
    formula="travelling_wave",
    solution_features=[
        {"sum": ["x", {"term": "t", "sign": -1}], "transform": "sin"},
    ],
    explanation=(
        "The rule was y = A·sin(k·x − ω·t + φ) with k = ω = 1 and φ = 0, so y = 5·sin(x − t). "
        "The sine of the phase x − t is the feature. The amplitude 5 is the coefficient."
    ),
    connection="Displacement of a travelling wave on a string at a fixed point.",
)


_PHYS_COULOMB_2 = _physics(
    id="coulomb_2",
    title="Two Sparks",
    description=(
        "Two point charges sit on a line: q1 = 2 μC and q2 = −3 μC. They are r1 and r2 "
        "metres from a measuring point. Record the electric potential at that point."
    ),
    ranges={"r1": (1.0, 5.0), "r2": (1.0, 5.0)},
    fixed={"q1": 2e-6, "q2": -3e-6},
    formula="coulomb_2",
    solution_features=["reciprocal:r1", "reciprocal:r2"],
    explanation=(
        "The rule was y = k_e·(q1/r1 + q2/r2). Potentials add (superposition), so each "
        "distance gives one feature, 1/r1 and 1/r2. The charges are the coefficients."
    ),
    connection="Electric potential from two point charges (superposition).",
)



# ===========================================================================
# IMAGES (difficulty = 3)
# Curated 256×256 inputs and fixed transforms. See images.py.
# ===========================================================================

_IMAGE_TRANSFORMS = []
_IMAGE_BINARY = []


def _image(*, id, title, description, image, transform, explanation) -> Puzzle:
    return _p(
        id=id,
        title=title,
        category="image",
        difficulty=3,
        description=description,
        input_features=[],
        function=FunctionSpec(type="image", parameters={"image": image, "transform": transform}),
        noise_std=0.0,
        allowed_transforms=_IMAGE_TRANSFORMS,
        allowed_binary_transforms=_IMAGE_BINARY,
        intended_model="none",
        solution_features=[],
        explanation=explanation,
        real_world_connection="",
    )


_IMG_SOLARISE = _image(
    id="solarise",
    title="Too Bright",
    description="Two pictures are shown. Work out what was done to the first to make the second.",
    image="doctor_strange",
    transform="solarise",
    explanation="Pixels at or above 128 are inverted (255 − value); darker pixels are unchanged.",
)

_IMG_POSTERISE = _image(
    id="posterise",
    title="Flat Colours",
    description="Two pictures are shown. Work out what was done to the first to make the second.",
    image="istockphoto",
    transform="posterise",
    explanation="Each colour channel is reduced to 4 levels: 0, 85, 170 and 255.",
)

_IMG_OVERLAY = _image(
    id="overlay",
    title="Ghosting",
    description="Two pictures are shown. Work out what was done to the first to make the second.",
    image="matrix",
    transform="repeated_overlay",
    explanation=(
        "The image is blended with copies of itself shifted by 4, 8 and 12 pixels, "
        "each step at 50% opacity."
    ),
)

_IMG_CHANNEL_SHUFFLE = _image(
    id="channel_shuffle",
    title="Swapped Dye",
    description="Two pictures are shown. Work out what was done to the first to make the second.",
    image="pexels",
    transform="channel_shuffle",
    explanation="Each pixel's channels move from (R, G, B) to (B, R, G).",
)

_IMG_FFT_ENCODE = _image(
    id="fft_encode",
    title="Hidden Waves",
    description="Two pictures are shown. Work out what was done to the first to make the second.",
    image="chessboard",
    transform="fft_encode",
    explanation=(
        "The 2D Fourier transform of the grey image, shifted so zero frequency is in "
        "the centre. Red shows log-magnitude and green shows phase."
    ),
)

_IMG_FFT_DECODE = _image(
    id="fft_decode",
    title="Back From the Waves",
    description="Two pictures are shown. Work out what was done to the first to make the second.",
    image="istockphoto",
    transform="fft_decode",
    explanation=(
        "The first picture is the encoded spectrum of an image. The second is the inverse "
        "2D Fourier transform of that spectrum, which recovers the original image."
    ),
)


# ===========================================================================
# Registry
# ===========================================================================

PUZZLE_REGISTRY: dict[str, Puzzle] = {
    p.id: p
    for p in [
        _LINE_01, _LINE_02, _SQUARE_01, _SQRT_01, _LOG_01, _DISTRACTOR_01,
        _PHYS_PROJECTILE_Y, _PHYS_SHM_ENERGY, _PHYS_TRAVELLING_WAVE, _PHYS_COULOMB_2,
        _IMG_SOLARISE, _IMG_POSTERISE, _IMG_OVERLAY, _IMG_CHANNEL_SHUFFLE,
        _IMG_FFT_ENCODE, _IMG_FFT_DECODE,
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
        available = sorted(PUZZLE_REGISTRY.keys())
        raise KeyError(
            f"Puzzle '{puzzle_id}' not found. Available: {available}"
        )
    return PUZZLE_REGISTRY[puzzle_id]


def list_puzzles(difficulty: int | None = None) -> list[dict]:
    """
    Return a list of puzzle metadata dicts.

    Parameters
    ----------
    difficulty : 1 (Beginner), 2 (Intermediate), 3 (Challenge), or None (all).
    """
    puzzles = list(PUZZLE_REGISTRY.values())
    if difficulty is not None:
        puzzles = [p for p in puzzles if p.difficulty == difficulty]
    return [
        {
            "id":                      p.id,
            "title":                   p.title,
            "category":                p.category,
            "difficulty":              p.difficulty,
            "description":             p.description,
            "intended_model":          p.intended_model,
            "input_features":          p.input_features,
            "allowed_transforms":      p.allowed_transforms,
            "allowed_binary_transforms": p.allowed_binary_transforms,
        }
        for p in puzzles
    ]
