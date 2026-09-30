"""
puzzles.py — Library of 25 educational puzzles.

Changes from v1
---------------
- All titles are now intentionally cryptic (non-revealing).
- Hints removed from all puzzles.
- Decision-tree puzzles reduced from 6 → 2 (circle_01, boss_piecewise).
- Four new linear-regression puzzles added:
    abs_01          y = |x|
    cos_01          y = cos(x)
    polynomial_01   y = x² − 3x  (multi-feature)
    phase_01        y = sin(x) + cos(x)

Puzzle catalogue
----------------
Beginner (difficulty=1)  — 6 puzzles:
    line_01          LR   y = 3x + 15
    line_02          LR   y = −2x + 20
    square_01        LR   y = x²
    sqrt_01          LR   y = 2√x
    log_01           LR   y = 3 log(x) + 1
    distractor_01    LR   y = 3x1 + 5 (three red-herring features)

Intermediate (difficulty=2)  — 9 puzzles:
    almost_linear_01 LR   y = x + 0.5 sin(x)
    almost_linear_02 LR   y = 2x + sin(x) + noise
    reciprocal_01    LR   y = 10/x
    abs_01           LR   y = |x|          ← NEW
    cos_01           LR   y = cos(x)       ← NEW
    periodic_01      LR   y = sin(x)
    periodic_02      LR   y = sin(2πx/7)
    product_01       LR   y = x1 × x2
    ratio_01         LR   y = x1 / x2

Challenge (difficulty=3)  — 10 puzzles:
    circle_01        DT   y ∈ {0,1} inside/outside circle
    distance_01      LR   y = √(x1²+x2²)
    cubic_01         LR   y = x³
    boss_piecewise   DT   three-segment piecewise
    boss_multi       LR   y = x1·x2 + 2x3
    boss_sin_sum     LR   y = sin(x) + 0.5 sin(3x)
    boss_multi_feat  LR   y = sin(x1) + x2² + 3x3
    period_boss      LR   y = sin(2πx/7) + noise + distractor
    polynomial_01    LR   y = x² − 3x  (needs two features)  ← NEW
    phase_01         LR   y = sin(x) + cos(x)                ← NEW
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
# INTERMEDIATE (difficulty = 2)
# ===========================================================================

_ALMOST_LINEAR_01 = _p(
    id="almost_linear_01",
    title="The Imposter Line",
    category="almost_linear",
    difficulty=2,
    description=(
        "The scatter plot looks almost linear — but something is off. "
        "A single straight line gives a good but not perfect fit. "
        "What is hiding in the residuals?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="almost_linear",
        parameters={"slope": 1.0, "amplitude": 0.5, "x_min": 0.0, "x_max": 10.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "sin", "cos", "square", "sqrt"],
    allowed_binary_transforms=["add"],
    intended_model="linear_regression",
    solution_features=["identity:x", "sin:x"],
    explanation=(
        "The rule was y = x + 0.5·sin(x). "
        "Raw x gives R² ≈ 0.97 — decent but not perfect. "
        "The residuals have a wave pattern — that's the sin(x) component. "
        "Adding sin(x) as a second feature brings R² to 1.0. "
        "Sometimes a single feature is not enough."
    ),
    real_world_connection=(
        "Daily temperature follows a linear seasonal trend "
        "plus a 24-hour cycle — both matter."
    ),
)

_ALMOST_LINEAR_02 = _p(
    id="almost_linear_02",
    title="Static on the Signal",
    category="almost_linear",
    difficulty=2,
    description=(
        "The data has noise AND a wave hiding inside a trend. "
        "A straight line is not wrong, just incomplete. "
        "Something periodic is mixed in."
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="almost_linear",
        parameters={"slope": 2.0, "amplitude": 1.0, "x_min": 0.0, "x_max": 10.0},
    ),
    noise_std=0.1,
    allowed_transforms=["identity", "sin", "cos", "square"],
    allowed_binary_transforms=["add"],
    intended_model="linear_regression",
    solution_features=["identity:x", "sin:x"],
    explanation=(
        "The rule was y = 2x + sin(x) + noise. "
        "Noise means R² < 1 is expected. "
        "With features [x, sin(x)], R² reaches ~0.99 — as good as it gets. "
        "Real data always has noise; the goal is the best achievable R²."
    ),
    real_world_connection=(
        "Stock prices show a long-term trend plus seasonal cycles — "
        "analysts model both."
    ),
)

_RECIPROCAL_01 = _p(
    id="reciprocal_01",
    title="Vanishing Point",
    category="feature_transform",
    difficulty=2,
    description=(
        "As x grows, y shrinks dramatically — not just a little. "
        "When x is large, y is almost zero. "
        "What function recreates this rapid collapse?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="reciprocal_fn",
        parameters={"a": 10.0, "x_min": 1.0, "x_max": 20.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "reciprocal", "log", "sqrt", "square"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["reciprocal:x"],
    explanation=(
        "The rule was y = 10/x. "
        "Raw x vs y is a steep curve — linear regression fails. "
        "But 1/x perfectly captures this 'inversely proportional' relationship: "
        "y vs (1/x) is a straight line through the origin."
    ),
    real_world_connection=(
        "Gravitational force between two bodies is inversely proportional to distance²."
    ),
)

_ABS_01 = _p(
    id="abs_01",
    title="The Symmetric Grudge",
    category="feature_transform",
    difficulty=2,
    description=(
        "Negative and positive x both produce positive y of the same size. "
        "The function seems to hate signs — it strips them away. "
        "What transformation reveals the straight-line relationship?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="abs_fn",
        parameters={"a": 2.0, "x_min": -5.0, "x_max": 5.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "abs", "square", "sqrt", "sin"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["abs:x"],
    explanation=(
        "The rule was y = 2|x|. "
        "Raw x gives R² ≈ 0 because the positive and negative halves cancel out. "
        "x² also works partially but |x| is the minimal correct transformation. "
        "y vs |x| is a perfect V-shape — a straight line with positive gradient "
        "that linear regression recovers instantly."
    ),
    real_world_connection=(
        "Error or deviation is always measured as an absolute value "
        "— sign doesn't matter, magnitude does."
    ),
)

_COS_01 = _p(
    id="cos_01",
    title="The Quarter-Turn",
    category="periodicity",
    difficulty=2,
    description=(
        "y oscillates like a wave — but it starts at its maximum "
        "and crosses zero at π/2 instead of at 0. "
        "It is not sin. What is it?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="cosine_fn",
        parameters={"amplitude": 1.0, "freq": 1.0, "x_min": 0.0, "x_max": 4 * 3.14159},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "sin", "cos", "square", "sqrt"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["cos:x"],
    explanation=(
        "The rule was y = cos(x). "
        "sin and cos are quarter-turn (π/2 radian) shifts of each other. "
        "sin(x) gives R² ≈ 0 here; cos(x) gives R² = 1. "
        "Both are valid periodic features — but which one you need "
        "depends on where the wave peaks."
    ),
    real_world_connection=(
        "In AC circuits, voltage and current are often offset by a phase angle — "
        "sin vs cos captures that offset."
    ),
)

_PERIODIC_01 = _p(
    id="periodic_01",
    title="The Repeating Rumour",
    category="periodicity",
    difficulty=2,
    description=(
        "y goes up, then down, then up again with perfect regularity. "
        "The pattern is smooth and symmetric. "
        "Name the transformation that captures it exactly."
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="sinusoidal",
        parameters={
            "amplitude": 1.0, "period": 6.28318, "offset": 0.0,
            "x_min": 0.0, "x_max": 4 * 3.14159,
        },
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "sin", "cos", "square", "sqrt"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["sin:x"],
    explanation=(
        "The rule was y = sin(x). "
        "No polynomial captures a smooth repeating wave. "
        "sin(x) matches the pattern exactly — R² = 1. "
        "Linear regression on sin(x) discovers the coefficient is exactly 1."
    ),
    real_world_connection="Ocean tides, sound waves, and AC electricity follow sine curves.",
)

_PERIODIC_02 = _p(
    id="periodic_02",
    title="Seven Days of Nothing",
    category="periodicity",
    difficulty=2,
    description=(
        "y completes one full cycle every 7 units. "
        "Plain sin(x) doesn't fit — it has the wrong period. "
        "How do you engineer a feature with exactly a period of 7?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="sinusoidal",
        parameters={
            "amplitude": 1.0, "period": 7.0, "offset": 0.0,
            "x_min": 0.0, "x_max": 28.0,
        },
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "sin", "cos", "sin_period7", "cos_period7"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["sin_period7:x"],
    explanation=(
        "The rule was y = sin(2π·x/7). "
        "sin(x) has period 2π ≈ 6.28, not 7 — so it fits poorly. "
        "Dividing x by the period before applying sin gives the correct frequency. "
        "Use sin_period7 which computes sin(2π·x/7) directly."
    ),
    real_world_connection="Online traffic peaks every weekend — a 7-day cycle.",
)

_PRODUCT_01 = _p(
    id="product_01",
    title="The Missing Third Variable",
    category="interaction",
    difficulty=2,
    description=(
        "You have x1 and x2. Neither alone explains y. "
        "But something about the two features together does. "
        "What is the hidden combination?"
    ),
    input_features=["x1", "x2"],
    function=FunctionSpec(
        type="product_2d",
        parameters={"x_min": 1.0, "x_max": 10.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "square", "sqrt"],
    allowed_binary_transforms=["multiply", "add", "subtract"],
    intended_model="linear_regression",
    solution_features=[{"binary": "multiply", "a": "x1", "b": "x2"}],
    explanation=(
        "The rule was y = x1 × x2. "
        "If x1 is fixed, y scales with x2 — and vice versa. "
        "Alone each gives R² ≈ 0.4. The product x1·x2 gives R² = 1. "
        "Sometimes the useful feature does not exist yet — you create it."
    ),
    real_world_connection="Revenue = quantity × unit_price — a product interaction.",
)

_RATIO_01 = _p(
    id="ratio_01",
    title="Speed Without Units",
    category="ratio",
    difficulty=2,
    description=(
        "x1 alone is messy. x2 alone is messy. "
        "But maybe the measurements only make sense relative to each other. "
        "What single feature explains everything?"
    ),
    input_features=["x1", "x2"],
    function=FunctionSpec(
        type="ratio_2d",
        parameters={"x_min": 1.0, "x_max": 10.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity"],
    allowed_binary_transforms=["multiply", "divide", "add", "subtract"],
    intended_model="linear_regression",
    solution_features=[{"binary": "divide", "a": "x1", "b": "x2"}],
    explanation=(
        "The rule was y = x1 / x2. "
        "Alone, each feature has weak R² because y depends on their ratio, "
        "not individual values. x1/x2 gives R² = 1. "
        "Ratios appear everywhere: speed, density, efficiency, concentration."
    ),
    real_world_connection="Fuel efficiency is km/litre — a ratio of two measurements.",
)


# ===========================================================================
# CHALLENGE (difficulty = 3)
# ===========================================================================

_CIRCLE_01 = _p(
    id="circle_01",
    title="The Exclusion Zone",
    category="geometry",
    difficulty=3,
    description=(
        "Points are labelled 0 or 1. Neither x1 nor x2 alone separates them. "
        "But there is a clear geometric boundary — "
        "if you could only see it. What feature reveals the boundary?"
    ),
    input_features=["x1", "x2"],
    function=FunctionSpec(
        type="circle_classify",
        parameters={"radius": 4.0, "x_min": -6.0, "x_max": 6.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "square"],
    allowed_binary_transforms=["multiply", "add", "distance"],
    intended_model="decision_tree",
    solution_features=[{"binary": "distance", "a": "x1", "b": "x2"}],
    explanation=(
        "The rule was y = 1 if x1² + x2² < 16 else 0. "
        "The boundary is a circle of radius 4. "
        "A decision tree on √(x1²+x2²) finds the split at radius=4 automatically. "
        "The insight: whenever a boundary looks circular, "
        "distance from the centre is the right feature."
    ),
    real_world_connection=(
        "Whether a radar target is within range depends on its distance "
        "from the antenna — not its x or y coordinate separately."
    ),
)

_DISTANCE_01 = _p(
    id="distance_01",
    title="The Displacement Field",
    category="geometry",
    difficulty=3,
    description=(
        "Two coordinates, one output. y seems to grow with both x1 and x2, "
        "but neither alone tells the whole story. "
        "The output feels geometric."
    ),
    input_features=["x1", "x2"],
    function=FunctionSpec(
        type="distance_2d",
        parameters={"x_min": -5.0, "x_max": 5.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "square", "sqrt"],
    allowed_binary_transforms=["distance", "add", "multiply"],
    intended_model="linear_regression",
    solution_features=[{"binary": "distance", "a": "x1", "b": "x2"}],
    explanation=(
        "The rule was y = √(x1² + x2²) — Euclidean distance from the origin. "
        "x1 and x2 alone give R² ≈ 0.5. "
        "The engineered feature dist = √(x1²+x2²) gives R² = 1. "
        "This is the Pythagorean theorem applied to feature engineering."
    ),
    real_world_connection=(
        "WiFi signal strength depends on your distance from the router, "
        "computed from x and y coordinates."
    ),
)

_CUBIC_01 = _p(
    id="cubic_01",
    title="Tripling the Problem",
    category="feature_transform",
    difficulty=3,
    description=(
        "y is large and positive for large x, large and NEGATIVE for large negative x. "
        "The pattern is not symmetric. x² would be symmetric — this is not. "
        "What power captures the anti-symmetry?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="cubic",
        parameters={"a": 1.0, "x_min": -3.0, "x_max": 3.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "square", "cube", "abs", "sqrt"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["cube:x"],
    explanation=(
        "The rule was y = x³. "
        "x² is even (symmetric): (−3)² = (3)² = 9. "
        "x³ is odd (anti-symmetric): (−3)³ = −27, 3³ = +27. "
        "The anti-symmetry in the data is the fingerprint of an odd power. "
        "Fitting LR on x³ gives R² = 1."
    ),
    real_world_connection="The volume of a sphere grows as r³.",
)

_BOSS_PIECEWISE = _p(
    id="boss_piecewise",
    title="The Three Regimes",
    category="piecewise",
    difficulty=3,
    description=(
        "The slope changes — not once but twice. "
        "Three distinct regions, each with its own behaviour. "
        "Linear regression gives one average slope and captures none of them well. "
        "Which model can find both change-points automatically?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="boss_piecewise",
        parameters={
            "t1": 3.0, "t2": 7.0,
            "slope1": 2.0, "c1": 0.0,
            "slope2": 1.0,
            "slope3": 0.5,
            "x_min": 0.0, "x_max": 15.0,
        },
    ),
    noise_std=0.0,
    allowed_transforms=["identity"],
    allowed_binary_transforms=[],
    intended_model="decision_tree",
    solution_features=["identity:x"],
    explanation=(
        "The rule: y = 2x (x<3), y = x+3 (3≤x<7), y = 0.5x+6.5 (x≥7). "
        "Linear regression averages all three slopes — poor fit. "
        "A decision tree of depth 3 discovers both split points automatically, "
        "achieving near-perfect R² on each segment."
    ),
    real_world_connection=(
        "Electricity tariffs have multiple tiers — "
        "different rates for low, medium, and high usage."
    ),
)

_BOSS_MULTI = _p(
    id="boss_multi",
    title="The Hidden Tax",
    category="interaction",
    difficulty=3,
    description=(
        "Three features: x1, x2, x3. "
        "The output combines something from x1 and x2 together "
        "with something from x3 alone. "
        "Can you build both pieces?"
    ),
    input_features=["x1", "x2", "x3"],
    function=FunctionSpec(
        type="boss_multi",
        parameters={"c": 2.0, "x_min": 1.0, "x_max": 8.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity"],
    allowed_binary_transforms=["multiply", "add"],
    intended_model="linear_regression",
    solution_features=[
        {"binary": "multiply", "a": "x1", "b": "x2"},
        "identity:x3",
    ],
    explanation=(
        "The rule was y = x1·x2 + 2x3. "
        "Using features [x1·x2, x3] together, linear regression learns "
        "coefficients 1 and 2 respectively — R² = 1. "
        "This puzzle combines the product interaction lesson with "
        "the straight-line lesson."
    ),
    real_world_connection=(
        "Revenue = units_sold × price_per_unit + fixed_commission."
    ),
)

_BOSS_SIN_SUM = _p(
    id="boss_sin_sum",
    title="Interfering Signals",
    category="periodicity",
    difficulty=3,
    description=(
        "Two waves are superimposed. sin(x) gives a decent fit but "
        "leaves a residual pattern that is also wave-like, just faster. "
        "The second wave has a frequency three times the first."
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="sinusoidal_sum",
        parameters={
            "a1": 1.0, "f1": 1.0,
            "a2": 0.5, "f2": 3.0,
            "x_min": 0.0, "x_max": 6.28318,
        },
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "sin", "cos"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["sin:x"],
    explanation=(
        "The rule was y = sin(x) + 0.5·sin(3x). "
        "sin(x) alone gives R² ≈ 0.89. "
        "To fully capture y you need a second feature at 3× the frequency. "
        "This hints at Fourier analysis: real signals are sums of multiple frequencies."
    ),
    real_world_connection="Music is a sum of many pure sine waves at different frequencies.",
)

_BOSS_MULTI_FEAT = _p(
    id="boss_multi_feat",
    title="Chaos in Three Channels",
    category="feature_transform",
    difficulty=3,
    description=(
        "Three features arrive. Each needs a different transformation. "
        "x1 lives in an angle-like range. "
        "x2 has a curved, non-symmetric relationship with y. "
        "x3 looks linear. Find the right transformation for each."
    ),
    input_features=["x1", "x2", "x3"],
    function=FunctionSpec(
        type="multi_transform",
        parameters={"a": 1.0, "b": 1.0, "c": 3.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "sin", "square", "sqrt", "log"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["sin:x1", "square:x2", "identity:x3"],
    explanation=(
        "The rule was y = sin(x1) + x2² + 3x3. "
        "Each feature needed its own transform: "
        "sin for the angle-like x1, square for the curved x2, "
        "identity for the already-linear x3. "
        "This is the full feature-engineering pipeline: explore, transform, combine."
    ),
    real_world_connection=(
        "Weather models combine periodic time features, quadratic altitude terms, "
        "and raw sensor readings in one linear model."
    ),
)

_PERIOD_BOSS = _p(
    id="period_boss",
    title="The Noisy Calendar",
    category="periodicity",
    difficulty=3,
    description=(
        "Two inputs: x (a day index) and x2 (unknown). "
        "y follows a 7-day cycle but has noise. "
        "One input is useful. The other is a distractor. "
        "Can you find the weekly pattern through the noise?"
    ),
    input_features=["x", "x2"],
    function=FunctionSpec(
        type="sinusoidal",
        parameters={
            "amplitude": 1.0, "period": 7.0, "offset": 0.0,
            "x_min": 0.0, "x_max": 28.0,
        },
    ),
    noise_std=0.15,
    allowed_transforms=["identity", "sin_period7", "cos_period7", "sin", "cos"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["sin_period7:x"],
    explanation=(
        "The rule was y = sin(2π·x/7) + noise. "
        "x2 gives R² ≈ 0 — it's noise. "
        "sin_period7 gives R² ≈ 0.95 — not 1.0 because of noise, "
        "but clearly the best feature. "
        "Noisy data means perfect R² is impossible; aim for the highest achievable."
    ),
    real_world_connection=(
        "Website traffic has a clear weekly cycle even though "
        "daily fluctuations add noise."
    ),
)

_POLYNOMIAL_01 = _p(
    id="polynomial_01",
    title="The Bent Wire",
    category="feature_transform",
    difficulty=3,
    description=(
        "The curve bends — but it doesn't just go up. "
        "It dips below zero on one side and rises on the other, "
        "as if pulled in opposite directions. "
        "A single feature is not enough. You need two."
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="polynomial2_fn",
        parameters={"a": 1.0, "b": -3.0, "x_min": -2.0, "x_max": 5.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "square", "cube", "sqrt", "abs"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["square:x", "identity:x"],
    explanation=(
        "The rule was y = x² − 3x. "
        "Neither x² alone nor x alone gives R² = 1. "
        "Together as features [x², x], linear regression learns "
        "coefficients 1 and −3, achieving R² = 1. "
        "This shows that multi-term polynomials need multiple engineered features."
    ),
    real_world_connection=(
        "Projectile height = −4.9t² + v·t — a quadratic in time, "
        "requiring both t and t² as features."
    ),
)

_PHASE_01 = _p(
    id="phase_01",
    title="The Hidden Angle",
    category="periodicity",
    difficulty=3,
    description=(
        "It is periodic. It is smooth. "
        "But it is neither sin(x) nor cos(x) alone. "
        "It starts between them and ends between them. "
        "A phase shift hides inside."
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="phase_sum",
        parameters={"x_min": 0.0, "x_max": 4 * 3.14159},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "sin", "cos", "square"],
    allowed_binary_transforms=["add"],
    intended_model="linear_regression",
    solution_features=["sin:x", "cos:x"],
    explanation=(
        "The rule was y = sin(x) + cos(x) = √2 · sin(x + π/4). "
        "sin(x) gives R² ≈ 0.5; cos(x) gives R² ≈ 0.5. "
        "Using BOTH as features, linear regression recovers coefficients "
        "[1, 1] and achieves R² = 1. "
        "Any phase-shifted sinusoid can be written as a·sin(x) + b·cos(x)."
    ),
    real_world_connection=(
        "Signal demodulation in communications uses sin and cos basis functions "
        "to recover amplitude and phase simultaneously."
    ),
)


# ===========================================================================
# Registry
# ===========================================================================

PUZZLE_REGISTRY: dict[str, Puzzle] = {
    p.id: p
    for p in [
        # Beginner
        _LINE_01, _LINE_02, _SQUARE_01, _SQRT_01, _LOG_01, _DISTRACTOR_01,
        # Intermediate
        _ALMOST_LINEAR_01, _ALMOST_LINEAR_02, _RECIPROCAL_01,
        _ABS_01, _COS_01,
        _PERIODIC_01, _PERIODIC_02, _PRODUCT_01, _RATIO_01,
        # Challenge
        _CIRCLE_01, _DISTANCE_01, _CUBIC_01, _BOSS_PIECEWISE,
        _BOSS_MULTI, _BOSS_SIN_SUM, _BOSS_MULTI_FEAT, _PERIOD_BOSS,
        _POLYNOMIAL_01, _PHASE_01,
    ]
}


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
