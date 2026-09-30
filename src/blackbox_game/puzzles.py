"""
puzzles.py — The complete library of 25 educational puzzles.

Puzzle library overview
-----------------------
Beginner (difficulty=1):
    line_01        Straight line (positive slope)
    line_02        Straight line (negative slope)
    square_01      Quadratic — y = x²
    sqrt_01        Square root — y = √x
    log_01         Logarithm — y = log(x)
    distractor_01  Irrelevant features — only one feature matters

Intermediate (difficulty=2):
    almost_linear_01  y = x + 0.5 sin(x) — looks nearly linear
    almost_linear_02  y = 2x + sin(x)
    reciprocal_01     y = 10/x
    piecewise_01      Piecewise flat (decision tree shines)
    piecewise_02      Piecewise linear (two slopes)
    periodic_01       y = sin(x) — pure sine
    periodic_02       y = sin(2πx/7) — weekly cycle
    product_01        y = x1 * x2 — interaction feature
    ratio_01          y = x1 / x2 — ratio feature

Challenge (difficulty=3):
    circle_01         Classification inside/outside a circle
    distance_01       y = sqrt(x1² + x2²) — distance feature engineering
    staircase_01      y = floor(x/10)*10 — discrete staircase
    cubic_01          y = x³ — cubic relationship
    boss_piecewise    Three-segment piecewise (boss puzzle)
    boss_multi        y = x1*x2 + 2x3 — combine interaction + linear
    boss_sin_sum      y = sin(x) + 0.5*sin(3x) — harmonic sum
    boss_multi_feat   y = sin(x1) + x2² + 3x3 — three transforms
    period_boss       y = sin(2πx/7) with noise and distractor
    piecewise_03      Three-region piecewise (boss)

Usage
-----
    from blackbox_game.puzzles import get_puzzle, list_puzzles, PUZZLE_REGISTRY
"""

from __future__ import annotations

from .models import Puzzle, FunctionSpec


# ---------------------------------------------------------------------------
# Helper to build puzzles consistently
# ---------------------------------------------------------------------------

def _p(**kwargs) -> Puzzle:
    """Thin wrapper so each puzzle definition is readable."""
    return Puzzle(**kwargs)


# ===========================================================================
# BEGINNER PUZZLES (difficulty = 1)
# ===========================================================================

_LINE_01 = _p(
    id="line_01",
    title="The Straight Path",
    category="linear",
    difficulty=1,
    description=(
        "A mysterious machine takes a number x and produces output y. "
        "You notice the output goes up steadily as x increases. "
        "Can you find the hidden rule?"
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
    hints=[
        "Try plotting y against x. Does the relationship look like a straight line?",
        "If the relationship is a straight line, which transformation of x keeps it as a straight line?",
        "The simplest transformation is no transformation at all — use x directly!",
    ],
    explanation=(
        "The hidden rule was simply y = 3x + 15. "
        "When you use x directly (no transformation), linear regression finds "
        "a perfect straight-line fit. "
        "R² close to 1 means the line explains almost everything. "
        "This is the simplest possible relationship a model can learn."
    ),
    real_world_connection=(
        "Distance = speed × time is a real straight-line relationship."
    ),
)

_LINE_02 = _p(
    id="line_02",
    title="Going Down",
    category="linear",
    difficulty=1,
    description=(
        "This machine also produces a straight-line relationship, "
        "but the output decreases as x increases. "
        "Can you still find a good fit?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="linear",
        parameters={"slope": -2.0, "intercept": 20.0, "x_min": 0.0, "x_max": 10.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "square", "sqrt", "sin"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["identity:x"],
    hints=[
        "Even if y decreases, can a straight line still fit the data?",
        "Linear regression works for negative slopes too!",
        "Use x directly — linear regression will discover the negative slope automatically.",
    ],
    explanation=(
        "The rule was y = −2x + 20. "
        "Linear regression can find both upward and downward slopes. "
        "The coefficient (slope) the model learns will be negative here. "
        "This teaches that 'linear' just means 'straight line', "
        "regardless of direction."
    ),
    real_world_connection=(
        "As temperature drops, the chance of engine failure can increase — "
        "a negative relationship."
    ),
)

_SQUARE_01 = _p(
    id="square_01",
    title="The Squaring Machine",
    category="feature_transform",
    difficulty=1,
    description=(
        "The output y grows much faster than x. "
        "When x doubles, y more than doubles. "
        "What is the hidden rule? "
        "Hint: try squaring x before fitting."
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
    hints=[
        "Plot y against x. Does the curve bend upward? That suggests something non-linear.",
        "What if you tried x²? Does it turn the curve into a straight line?",
        "Use the 'square' transformation: x² perfectly straightens out this relationship.",
    ],
    explanation=(
        "The rule was y = x². "
        "Plotting raw x against y gives a curved (parabola) shape — "
        "linear regression can't fit a curve with a straight line. "
        "But if you engineer a new feature x², the relationship becomes perfectly linear: "
        "y = 1 × x². This is feature engineering: creating a better input for your model."
    ),
    real_world_connection=(
        "Kinetic energy = ½mv² — energy grows with the square of speed."
    ),
)

_SQRT_01 = _p(
    id="sqrt_01",
    title="The Shrinking Returns",
    category="feature_transform",
    difficulty=1,
    description=(
        "y grows as x increases, but each extra unit of x adds less and less to y. "
        "The growth is 'slowing down'. What is the hidden relationship?"
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
    hints=[
        "The curve bends — but which direction? It grows fast at first, then slows down.",
        "Square root is a function that 'slows down' its growth. Try √x.",
        "Apply sqrt to x. That should straighten the relationship into a line.",
    ],
    explanation=(
        "The rule was y = 2√x. "
        "Raw x vs y looks like a curve that flattens — not a straight line. "
        "But √x grows in the same 'diminishing returns' pattern, "
        "so y vs √x becomes a straight line. "
        "Linear regression on √x finds R² ≈ 1."
    ),
    real_world_connection=(
        "The loudness of sound drops as the square root of distance from the source."
    ),
)

_LOG_01 = _p(
    id="log_01",
    title="The Logarithm Lab",
    category="feature_transform",
    difficulty=1,
    description=(
        "y increases with x, but even more slowly than a square root. "
        "Large values of x barely change y at all. "
        "Which transformation reveals the pattern?"
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
    hints=[
        "The output barely changes even when x is 100 vs 50. Very slow growth.",
        "Which function grows very slowly? Logarithm grows slower than square root.",
        "Apply log(x) to the input. Then plot y against log(x).",
    ],
    explanation=(
        "The rule was y = 3·log(x) + 1. "
        "Raw x vs y is a curve that gets flatter and flatter. "
        "But log(x) captures exactly this 'diminishing returns' behaviour — "
        "it grows without limit but very slowly. "
        "After applying log, the relationship becomes a straight line."
    ),
    real_world_connection=(
        "The Richter scale for earthquakes is logarithmic — "
        "a magnitude-7 quake is 10× more powerful than magnitude-6."
    ),
)

_DISTRACTOR_01 = _p(
    id="distractor_01",
    title="The Red Herrings",
    category="distractor",
    difficulty=1,
    description=(
        "You are given four input features: x1, x2, x3, x4. "
        "Only ONE of them actually determines the output. "
        "Can you figure out which one is useful and which are distractions?"
    ),
    input_features=["x1", "x2", "x3", "x4"],
    function=FunctionSpec(
        type="linear_distractor",
        parameters={"slope": 3.0, "intercept": 5.0, "x_min": 0.0, "x_max": 10.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["identity:x1"],
    hints=[
        "Try each feature one at a time. Which one gives the best R²?",
        "x2, x3, x4 are random — they have no relationship with y.",
        "x1 is the useful feature. Using it alone gives R² ≈ 1.",
    ],
    explanation=(
        "The rule was y = 3·x1 + 5. "
        "x2, x3, x4 are random numbers with no connection to y. "
        "This teaches a crucial lesson: not all available data is useful. "
        "A model trained on irrelevant features will not be reliable. "
        "Always check each feature's relationship with the output individually."
    ),
    real_world_connection=(
        "Doctors have hundreds of lab measurements — only a few "
        "are actually predictive of a given condition."
    ),
)


# ===========================================================================
# INTERMEDIATE PUZZLES (difficulty = 2)
# ===========================================================================

_ALMOST_LINEAR_01 = _p(
    id="almost_linear_01",
    title="Almost a Straight Line",
    category="almost_linear",
    difficulty=2,
    description=(
        "The output y looks almost linear in x — but not quite. "
        "There is a subtle wave-like deviation. "
        "Can you find a feature combination that perfectly explains y?"
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
    hints=[
        "If you use just x, you get a decent R². But look at the residuals — do they form a pattern?",
        "The leftover pattern after removing the linear trend looks wave-like.",
        "Add sin(x) as a second feature alongside x. Together they should explain y perfectly.",
    ],
    explanation=(
        "The rule was y = x + 0.5·sin(x). "
        "Using just x gives R² ≈ 0.97 (good but not perfect). "
        "The residuals (errors) have a wave pattern — that's the sine component. "
        "Adding sin(x) as a second feature brings R² to 1.0. "
        "This shows that sometimes you need multiple features to fully explain an output."
    ),
    real_world_connection=(
        "Daily temperature follows a roughly linear trend across seasons "
        "plus a daily cycle — similar structure."
    ),
)

_ALMOST_LINEAR_02 = _p(
    id="almost_linear_02",
    title="Wiggly Line",
    category="almost_linear",
    difficulty=2,
    description=(
        "The output y is mostly linear but has visible wiggles. "
        "A straight line leaves clear patterns in the residuals. "
        "What additional feature captures the wiggles?"
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
    hints=[
        "Plot y against x. The deviation from a straight line looks periodic.",
        "The wiggles repeat at roughly the same interval. What function repeats?",
        "Use both x and sin(x) as features.",
    ],
    explanation=(
        "The rule was y = 2x + sin(x) + small noise. "
        "The noise makes R² slightly less than 1, but the model with "
        "features [x, sin(x)] gets very close. "
        "Notice how the wiggles in the residuals disappeared once sin(x) was included."
    ),
    real_world_connection=(
        "Stock prices have a long-term trend (linear) plus shorter cycles — "
        "analysts often model both separately."
    ),
)

_RECIPROCAL_01 = _p(
    id="reciprocal_01",
    title="The Shrinking Giant",
    category="feature_transform",
    difficulty=2,
    description=(
        "As x gets larger, y gets smaller — and very quickly. "
        "When x is large, y is almost zero. "
        "Which transformation reveals the hidden straight-line relationship?"
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
    hints=[
        "The output y drops very fast as x increases. More than log or sqrt.",
        "What function shrinks very fast? Consider 1/x.",
        "Apply 1/x. Then plot y against 1/x — it should be a straight line.",
    ],
    explanation=(
        "The rule was y = 10/x. "
        "Raw x vs y is a steep curve — linear regression would fail badly. "
        "But 1/x perfectly captures this 'inversely proportional' relationship. "
        "y vs (1/x) is a perfect straight line through the origin."
    ),
    real_world_connection=(
        "The brightness of a light source falls as 1/distance² — "
        "an inverse relationship."
    ),
)

_PIECEWISE_01 = _p(
    id="piecewise_01",
    title="The Switch",
    category="piecewise",
    difficulty=2,
    description=(
        "Something strange happens near x = 5. "
        "Below 5, y increases steadily with x. "
        "Above 5, y stays almost constant. "
        "A single straight line can't capture this — what model can?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="piecewise_flat",
        parameters={"threshold": 5.0, "slope": 2.0, "x_min": 0.0, "x_max": 10.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity"],
    allowed_binary_transforms=[],
    intended_model="decision_tree",
    solution_features=["identity:x"],
    hints=[
        "Try linear regression on x. What R² do you get? Is there still a pattern in the residuals?",
        "The relationship clearly changes around x = 5. What kind of model splits data into regions?",
        "A decision tree can split at x = 5 and handle each region separately.",
    ],
    explanation=(
        "The rule was: y = 2x if x < 5, else y = 10. "
        "Linear regression gets R² ≈ 0.75 — not bad, but clearly wrong shape. "
        "A decision tree with depth 2 finds the split at x = 5 automatically "
        "and fits both regions perfectly. "
        "Decision trees are great when the relationship changes depending on the region."
    ),
    real_world_connection=(
        "A car's fuel efficiency is roughly linear up to a speed, "
        "then stays roughly constant — a piecewise relationship."
    ),
)

_PIECEWISE_02 = _p(
    id="piecewise_02",
    title="Two Slopes",
    category="piecewise",
    difficulty=2,
    description=(
        "The output increases with x, but the rate of increase changes at some point. "
        "Before the change, y grows steeply. After it, y grows more gently. "
        "Which model handles this best?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="piecewise_linear",
        parameters={
            "slope1": 2.0, "c1": 0.0,
            "slope2": 0.5,
            "threshold": 10.0,
            "x_min": 0.0, "x_max": 20.0,
        },
    ),
    noise_std=0.0,
    allowed_transforms=["identity"],
    allowed_binary_transforms=[],
    intended_model="decision_tree",
    solution_features=["identity:x"],
    hints=[
        "Linear regression will give a single slope. But the slope clearly changes midway.",
        "Look for the 'kink' in the data — the point where the growth rate changes.",
        "A decision tree automatically finds the kink and fits two separate lines.",
    ],
    explanation=(
        "The rule was: y = 2x if x < 10, else y = 0.5x + 15. "
        "Linear regression gets a single average slope — it misses the kink. "
        "A decision tree finds the split at x = 10 and fits two different slopes, "
        "achieving a much better R²."
    ),
    real_world_connection=(
        "Income tax rates change at different brackets — "
        "the relationship between income and tax has multiple slopes."
    ),
)

_PERIODIC_01 = _p(
    id="periodic_01",
    title="The Wave",
    category="periodicity",
    difficulty=2,
    description=(
        "The output y oscillates — it goes up, then down, then up again, "
        "following a repeating pattern. "
        "Can you find the right transformation?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="sinusoidal",
        parameters={"amplitude": 1.0, "period": 6.28, "offset": 0.0,
                    "x_min": 0.0, "x_max": 4 * 3.14159},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "sin", "cos", "square", "sqrt"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["sin:x"],
    hints=[
        "y goes up and down repeatedly. This is called a periodic pattern.",
        "What mathematical function repeats perfectly? Sine and cosine.",
        "Try sin(x) — plot it against y. They should match perfectly.",
    ],
    explanation=(
        "The rule was y = sin(x). "
        "No polynomial or power transformation can capture a wave. "
        "But sin(x) matches the pattern exactly — R² = 1. "
        "This is why trigonometric features are important for periodic data. "
        "Linear regression uses sin(x) as the feature and finds the coefficient is 1."
    ),
    real_world_connection=(
        "Ocean tides, sound waves, and AC electricity all follow sine curves."
    ),
)

_PERIODIC_02 = _p(
    id="periodic_02",
    title="The Hidden Week",
    category="periodicity",
    difficulty=2,
    description=(
        "The output represents something that repeats every 7 days. "
        "On day 0, 7, 14, ... the output starts its cycle again. "
        "Can you find a feature that captures this weekly pattern?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="sinusoidal",
        parameters={"amplitude": 1.0, "period": 7.0, "offset": 0.0,
                    "x_min": 0.0, "x_max": 28.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "sin", "cos", "sin_period7", "cos_period7"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["sin_period7:x"],
    hints=[
        "Count how many x-units it takes for the pattern to repeat. Is it 7?",
        "To capture a period of 7, use sin(2π·x/7) — not just sin(x).",
        "Use the 'sin_period7' transformation: sin(2π·x/7).",
    ],
    explanation=(
        "The rule was y = sin(2π·x/7). "
        "Plain sin(x) would fit a wave with period 2π ≈ 6.28, not 7. "
        "To match a period of exactly 7, you need sin(2π·x/7). "
        "This teaches that you can engineer periodic features for any known period."
    ),
    real_world_connection=(
        "Online store traffic often peaks on weekends — a 7-day cycle "
        "captured by sin(2π·day/7)."
    ),
)

_PRODUCT_01 = _p(
    id="product_01",
    title="The Hidden Combination",
    category="interaction",
    difficulty=2,
    description=(
        "You have two features: x1 and x2. "
        "Neither one alone explains y very well. "
        "But something about their combination does. "
        "Can you figure out what?"
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
    hints=[
        "Try x1 alone. Try x2 alone. Both give poor R². The answer involves both.",
        "What if the two features interact? Think: area = width × height.",
        "Multiply x1 × x2 and use that as your feature.",
    ],
    explanation=(
        "The rule was y = x1 × x2. "
        "Neither x1 nor x2 alone predicts y — if x1 is fixed, y scales with x2, "
        "and vice versa. "
        "The product x1·x2 perfectly captures this interaction. "
        "Feature engineering sometimes means creating entirely new features "
        "by combining existing ones."
    ),
    real_world_connection=(
        "Price = quantity × unit_price — two features whose product is the outcome."
    ),
)

_RATIO_01 = _p(
    id="ratio_01",
    title="The Relative Measure",
    category="ratio",
    difficulty=2,
    description=(
        "You are given x1 and x2. "
        "Plotting y against x1 looks messy. "
        "Plotting y against x2 is also messy. "
        "But maybe the two measurements only make sense relative to each other."
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
    hints=[
        "Try x1 and x2 individually — low R² for both.",
        "Sometimes two measurements only mean something relative to each other. Think speed = distance/time.",
        "Divide x1 by x2 and use that as your single feature.",
    ],
    explanation=(
        "The rule was y = x1 / x2. "
        "Alone, x1 and x2 each have weak R² because y depends on their ratio, "
        "not their individual values. "
        "The feature x1/x2 gives R² = 1. "
        "Ratios are everywhere in real life: speed, density, concentration, efficiency."
    ),
    real_world_connection=(
        "Fuel efficiency is measured as km/litre — "
        "a ratio of two measurements."
    ),
)


# ===========================================================================
# CHALLENGE PUZZLES (difficulty = 3)
# ===========================================================================

_CIRCLE_01 = _p(
    id="circle_01",
    title="Inside the Circle",
    category="geometry",
    difficulty=3,
    description=(
        "Each point has coordinates (x1, x2). "
        "The output y is 1 if the point is inside a hidden circle, "
        "and 0 if it is outside. "
        "Neither x1 nor x2 alone separates the two groups. "
        "What feature reveals the boundary?"
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
    hints=[
        "Plot the points in 2D space with colour = y. Do you see a circle?",
        "The boundary is circular — the points inside are all within some radius of the origin.",
        "Compute distance = sqrt(x1² + x2²). Points with small distance are inside.",
    ],
    explanation=(
        "The rule was y = 1 if x1² + x2² < 16 else 0. "
        "The boundary is a circle of radius 4. "
        "A decision tree on √(x1² + x2²) (the distance from the origin) "
        "finds the correct split at radius = 4. "
        "This shows that geometry can guide feature engineering: "
        "whenever a boundary looks circular, distance from the centre is a useful feature."
    ),
    real_world_connection=(
        "In radar systems, whether an object is within range depends on "
        "its distance from the antenna — not its x or y coordinate alone."
    ),
)

_DISTANCE_01 = _p(
    id="distance_01",
    title="How Far From Zero?",
    category="geometry",
    difficulty=3,
    description=(
        "Each input has two coordinates (x1, x2). "
        "The output y seems to depend on how far the point is from the origin. "
        "Can you engineer the right feature?"
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
    hints=[
        "x1 alone and x2 alone both give poor R². y depends on something about both together.",
        "Think about geometry. If the output is 'how far', what formula gives distance?",
        "Distance from origin = sqrt(x1² + x2²). Use the 'distance' binary transform.",
    ],
    explanation=(
        "The rule was y = √(x1² + x2²) — the Euclidean distance from the origin. "
        "Individually, x1 and x2 don't predict y well. "
        "But the engineered feature dist = √(x1² + x2²) gives R² = 1. "
        "This is the Pythagorean theorem in action: feature engineering using geometry."
    ),
    real_world_connection=(
        "The signal strength from a WiFi router depends on distance, "
        "which is calculated from x and y coordinates."
    ),
)

_STAIRCASE_01 = _p(
    id="staircase_01",
    title="The Staircase",
    category="piecewise",
    difficulty=3,
    description=(
        "The output y increases with x, but in discrete jumps. "
        "Between jumps, y stays flat. "
        "The pattern looks like a staircase. "
        "What model captures this?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="staircase",
        parameters={"step": 10.0, "x_min": 0.0, "x_max": 50.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "floor10"],
    allowed_binary_transforms=[],
    intended_model="decision_tree",
    solution_features=["identity:x"],
    hints=[
        "Linear regression on x gives moderate R². But the actual shape is not a line — it's steps.",
        "The output only changes at x = 10, 20, 30, 40. What model splits at multiple thresholds?",
        "A decision tree with several splits handles a staircase well. Try depth=4.",
    ],
    explanation=(
        "The rule was y = floor(x/10) × 10. "
        "The output jumps by 10 every time x crosses a multiple of 10. "
        "A decision tree finds these exact thresholds and predicts each flat region perfectly. "
        "Linear regression can't fit steps — it only draws a single straight line."
    ),
    real_world_connection=(
        "Postage rates increase in steps based on package weight — a staircase function."
    ),
)

_CUBIC_01 = _p(
    id="cubic_01",
    title="The Cube",
    category="feature_transform",
    difficulty=3,
    description=(
        "The output y grows very fast for positive x and drops very fast for negative x. "
        "It is not symmetric like x². "
        "Can you find the right transformation?"
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
    hints=[
        "The output is negative when x is negative. x² would be symmetric — this isn't.",
        "Try x³. It is negative for negative x and positive for positive x.",
        "Use the 'cube' transformation: x³ gives R² = 1.",
    ],
    explanation=(
        "The rule was y = x³. "
        "x² is symmetric around zero: (-3)² = 9 = 3². "
        "But x³ is anti-symmetric: (-3)³ = -27, while 3³ = +27. "
        "This asymmetry tells you x² is wrong and x³ is right. "
        "Fitting linear regression on x³ gives a perfect straight-line relationship."
    ),
    real_world_connection=(
        "The volume of a sphere grows as the cube of its radius — r³."
    ),
)

_BOSS_PIECEWISE = _p(
    id="boss_piecewise",
    title="Three Slopes",
    category="piecewise",
    difficulty=3,
    description=(
        "The relationship between x and y changes not once but twice. "
        "There are three distinct regions with different rates of change. "
        "Finding both change-points is the challenge. "
        "A decision tree is your friend here."
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
    hints=[
        "Linear regression on x gives poor R². The shape clearly has kinks.",
        "Look for the x values where the slope changes. There are two such points.",
        "A decision tree with depth 3 can find two split points and handle three regions.",
    ],
    explanation=(
        "The rule had three segments: y = 2x for x < 3, y = x + 3 for 3 ≤ x < 7, "
        "and y = 0.5x + 6.5 for x ≥ 7. "
        "Linear regression averages over all three slopes and fits none of them well. "
        "A decision tree of depth 3 discovers both split points automatically, "
        "achieving near-perfect fit on each segment."
    ),
    real_world_connection=(
        "Electricity pricing has multiple tiers — different rates for low, medium, "
        "and high usage."
    ),
)

_BOSS_MULTI = _p(
    id="boss_multi",
    title="Product Plus Linear",
    category="interaction",
    difficulty=3,
    description=(
        "Three features: x1, x2, x3. "
        "The output combines an interaction between x1 and x2 "
        "with a linear term from x3. "
        "Can you build all three pieces?"
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
    hints=[
        "Try x1 alone, x2 alone, x3 alone — moderate R² for each.",
        "Try x1*x2. Does combining them improve R²? What's still missing?",
        "Use features [x1*x2, x3] together. Linear regression on these two features explains y.",
    ],
    explanation=(
        "The rule was y = x1·x2 + 2x3. "
        "You need two features: the product x1·x2 (captures the interaction) "
        "and x3 directly (captures the linear term). "
        "Linear regression learns coefficients 1 for x1·x2 and 2 for x3. "
        "This combines what you learned in the product puzzle and the straight-line puzzle."
    ),
    real_world_connection=(
        "Revenue = units_sold × price_per_unit + fixed_bonus — "
        "a product plus a linear term."
    ),
)

_BOSS_SIN_SUM = _p(
    id="boss_sin_sum",
    title="Two Waves",
    category="periodicity",
    difficulty=3,
    description=(
        "The output y is a combination of two waves with different frequencies. "
        "A single sin(x) is not enough. "
        "You will need to find two trigonometric features."
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
    solution_features=["sin:x"],   # approximate — the sine with 3x requires manual engineering
    hints=[
        "sin(x) gives a decent fit but not R² ≈ 1. There is still a pattern in the residuals.",
        "The residual pattern is also wave-like, but faster (higher frequency).",
        "You need sin(x) AND a feature at 3× the frequency. But sin(3x) is not in the palette — "
        "try combining sin with other transforms.",
    ],
    explanation=(
        "The rule was y = sin(x) + 0.5·sin(3x). "
        "sin(x) alone gets R² ≈ 0.89. "
        "To fully explain y you need sin(x) and also a feature tracking the faster oscillation. "
        "This teaches that real signals can be sums of multiple frequencies — "
        "a concept called Fourier analysis (not required knowledge, but now you've seen it!)"
    ),
    real_world_connection=(
        "Music is a sum of many pure tones (sine waves) at different frequencies."
    ),
)

_BOSS_MULTI_FEAT = _p(
    id="boss_multi_feat",
    title="Three Transformations",
    category="feature_transform",
    difficulty=3,
    description=(
        "Three features: x1 (angle-like), x2 (possibly quadratic), x3 (linear). "
        "The hidden rule uses a different transformation for each. "
        "Your job: find which transformation applies to each feature."
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
    hints=[
        "Try each feature individually with several transforms. "
        "Which transform gives the best R² for each feature in isolation?",
        "x1 is in [0, 2π] — that suggests a trigonometric transformation. "
        "x2 has a curved relationship — maybe square?",
        "Use features: [sin(x1), x2², x3]. Together they should give R² ≈ 1.",
    ],
    explanation=(
        "The rule was y = sin(x1) + x2² + 3x3. "
        "Each feature needed its own transformation: "
        "sin for the angle-like feature, square for the curved one, "
        "and identity for the linear one. "
        "This is the full pipeline of feature engineering: "
        "explore, transform, combine, evaluate."
    ),
    real_world_connection=(
        "Weather prediction combines periodic variables (time of year → sin/cos), "
        "quadratic terms (altitude²), and linear variables (sensor readings) in one model."
    ),
)

_PERIOD_BOSS = _p(
    id="period_boss",
    title="Noisy Week with a Distractor",
    category="periodicity",
    difficulty=3,
    description=(
        "The output follows a 7-day cycle, but there is noise AND a distractor feature x2. "
        "x2 is random and should be ignored. "
        "Can you still find the weekly pattern?"
    ),
    input_features=["x", "x2"],
    function=FunctionSpec(
        type="sinusoidal",
        parameters={"amplitude": 1.0, "period": 7.0, "offset": 0.0,
                    "x_min": 0.0, "x_max": 28.0},
    ),
    noise_std=0.15,
    allowed_transforms=["identity", "sin_period7", "cos_period7", "sin", "cos"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["sin_period7:x"],
    hints=[
        "x2 is a distractor. Try it against y — does it give good R²? It shouldn't.",
        "Focus on x. The pattern repeats every 7 units.",
        "Use sin(2π·x/7). Noise makes R² slightly below 1, but still very high.",
    ],
    explanation=(
        "The rule was y = sin(2π·x/7) + noise. "
        "With noise, perfect R² = 1 is impossible, but sin_period7 gives R² ≈ 0.95+, "
        "while x2 gives R² ≈ 0. "
        "This teaches two lessons: (1) noise is real — don't expect perfect fits, "
        "and (2) irrelevant features should be discarded."
    ),
    real_world_connection=(
        "Website traffic has a weekly cycle. Other random events create noise — "
        "but the weekly pattern is still very clearly detectable."
    ),
)

_PIECEWISE_03 = _p(
    id="piecewise_03",
    title="The Three Zones",
    category="piecewise",
    difficulty=3,
    description=(
        "Imagine three different zones of x. "
        "In each zone, the relationship between x and y is completely different. "
        "No single straight line can capture all three zones. "
        "Which model handles this best?"
    ),
    input_features=["x"],
    function=FunctionSpec(
        type="boss_piecewise",
        parameters={
            "t1": 5.0, "t2": 10.0,
            "slope1": 3.0, "c1": 0.0,
            "slope2": 0.0,
            "slope3": -1.5,
            "x_min": 0.0, "x_max": 15.0,
        },
    ),
    noise_std=0.0,
    allowed_transforms=["identity"],
    allowed_binary_transforms=[],
    intended_model="decision_tree",
    solution_features=["identity:x"],
    hints=[
        "The data clearly has three visually distinct regions.",
        "In the middle region, y barely changes. In the right region, y actually decreases.",
        "A decision tree with depth 3 can split the data at x=5 and x=10.",
    ],
    explanation=(
        "The rule: y = 3x for x < 5, y = 15 (flat) for 5 ≤ x < 10, y = −1.5x + c for x ≥ 10. "
        "Three completely different behaviours! "
        "Linear regression produces a near-flat average — terrible fit. "
        "A decision tree discovers both split points and fits each region. "
        "Decision trees shine when data comes in distinct regions."
    ),
    real_world_connection=(
        "A spring behaves linearly, then plateaus (elastic limit), "
        "then breaks — three distinct regimes."
    ),
)


# ===========================================================================
# Puzzle registry
# ===========================================================================

PUZZLE_REGISTRY: dict[str, Puzzle] = {
    p.id: p
    for p in [
        # Beginner
        _LINE_01,
        _LINE_02,
        _SQUARE_01,
        _SQRT_01,
        _LOG_01,
        _DISTRACTOR_01,
        # Intermediate
        _ALMOST_LINEAR_01,
        _ALMOST_LINEAR_02,
        _RECIPROCAL_01,
        _PIECEWISE_01,
        _PIECEWISE_02,
        _PERIODIC_01,
        _PERIODIC_02,
        _PRODUCT_01,
        _RATIO_01,
        # Challenge
        _CIRCLE_01,
        _DISTANCE_01,
        _STAIRCASE_01,
        _CUBIC_01,
        _BOSS_PIECEWISE,
        _BOSS_MULTI,
        _BOSS_SIN_SUM,
        _BOSS_MULTI_FEAT,
        _PERIOD_BOSS,
        _PIECEWISE_03,
    ]
}


def get_puzzle(puzzle_id: str) -> Puzzle:
    """
    Retrieve a puzzle by its unique ID.

    Parameters
    ----------
    puzzle_id : str
        The puzzle identifier (e.g. ``"square_01"``).

    Returns
    -------
    Puzzle

    Raises
    ------
    KeyError
        If no puzzle with that ID exists.
    """
    if puzzle_id not in PUZZLE_REGISTRY:
        available = sorted(PUZZLE_REGISTRY.keys())
        raise KeyError(
            f"Puzzle '{puzzle_id}' not found. "
            f"Available puzzles: {available}"
        )
    return PUZZLE_REGISTRY[puzzle_id]


def list_puzzles(difficulty: int | None = None) -> list[dict]:
    """
    Return a list of puzzle metadata dictionaries.

    Parameters
    ----------
    difficulty : int | None
        Filter by difficulty (1, 2, or 3). Pass None to return all.

    Returns
    -------
    list[dict]
        Each dict has keys: id, title, category, difficulty, description,
        intended_model, input_features, allowed_transforms.
    """
    puzzles = PUZZLE_REGISTRY.values()
    if difficulty is not None:
        puzzles = [p for p in puzzles if p.difficulty == difficulty]

    return [
        {
            "id": p.id,
            "title": p.title,
            "category": p.category,
            "difficulty": p.difficulty,
            "description": p.description,
            "intended_model": p.intended_model,
            "input_features": p.input_features,
            "allowed_transforms": p.allowed_transforms,
            "allowed_binary_transforms": p.allowed_binary_transforms,
            "num_hints": len(p.hints),
        }
        for p in puzzles
    ]
