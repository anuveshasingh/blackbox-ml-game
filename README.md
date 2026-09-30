# 🎮 Blackbox ML Game

> An educational puzzle game engine that teaches Linear Regression and Decision Trees to first-year students — no prior ML knowledge required.

---

## What Is This?

Students at IIT Delhi play a puzzle game where a **hidden black-box function** maps inputs to outputs. Their job is to:

1. Explore the relationship between inputs and output
2. Try different **feature transformations** (x², √x, log(x), sin(x), …)
3. Choose the right model (**linear regression** or **decision tree**)
4. Discover the hidden rule and earn points

The game teaches **feature engineering** and model selection through discovery — not lectures.

---

## How the Game Works

```
Input feature(s) → ??? BLACK BOX ??? → output y
```

Each puzzle hides a function like:
- `y = x²` — can you figure out to square x?
- `y = 2x if x < 5, else 10` — decision tree!
- `y = x1 × x2` — combine two features!
- `y = sin(2πx/7)` — weekly cycle!

The player experiments by applying transformations and measuring how well a model fits:

> "If I use x², how well does linear regression explain y?"

The engine returns:

```json
{
    "is_correct": true,
    "r2": 0.999,
    "message": "🎉 Excellent! You found the hidden relationship!",
    "score_result": {"total_score": 192, "quality_bonus": 42}
}
```

### Understanding R²

| R² value | Meaning |
|----------|---------|
| ≈ 1.0    | This feature almost perfectly explains y |
| ≈ 0.7    | Some structure is captured but there's more |
| ≈ 0.0    | This feature barely helps |
| Negative | Worse than predicting the mean |

---

## Installation

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# Clone or download the repository
git clone <repo-url>
cd blackbox-ml-game

# Install (with development tools)
pip install -e ".[dev]"

# Or just install the runtime
pip install -e .
```

**Dependencies:** `numpy`, `pandas`, `scikit-learn`, `matplotlib` — nothing else.

---

## Quick Start

```python
from blackbox_game import (
    get_puzzle,
    generate_dataset,
    evaluate_submission,
    compare_models,
    get_hint,
)

# 1. Look at a puzzle
puzzle = get_puzzle("square_01")
print(puzzle.description)
# "The output y grows much faster than x..."

# 2. Generate data
dataset = generate_dataset(puzzle, n_samples=100, seed=42)
X = dataset["X"]   # pandas DataFrame
y = dataset["y"]   # numpy array

print(X.head())
print(y[:5])

# 3. Request a hint
hint = get_hint(puzzle, hint_index=0)
print(hint["text"])
# "Plot y against x. Does the curve bend upward?"

# 4. Try the wrong feature first
result = evaluate_submission(
    puzzle_id="square_01",
    model="linear_regression",
    features=["identity:x"],   # raw x — wrong!
)
print(result["r2"])         # 0.0  (x on [-5,5] is uncorrelated with x²)
print(result["is_correct"]) # False

# 5. Try the correct feature
result = evaluate_submission(
    puzzle_id="square_01",
    model="linear_regression",
    features=["square:x"],    # x² — correct!
)
print(result["r2"])          # 1.0
print(result["is_correct"])  # True
print(result["score_result"]["total_score"])  # 200
```

---

## Running the Full Example

```bash
python examples/basic_usage.py
```

This simulates a student solving four puzzles, including wrong guesses, hint usage, and final correct submissions.

---

## Running the Tests

```bash
pytest tests/ -v
```

All **110 tests** should pass in about 1 second.

```
110 passed in 1.01s
```

---

## Repository Structure

```
blackbox-ml-game/
│
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── src/
│   └── blackbox_game/
│       ├── __init__.py        ← Public API (start here)
│       ├── models.py          ← Puzzle & FunctionSpec dataclasses
│       ├── transforms.py      ← Feature transformation library
│       ├── puzzles.py         ← All 25 puzzle definitions
│       ├── generator.py       ← Reproducible dataset generation
│       ├── evaluator.py       ← sklearn model evaluation
│       ├── scoring.py         ← Deterministic scoring engine
│       ├── hints.py           ← Hint & explanation retrieval
│       └── viz.py             ← Optional matplotlib helpers
│
├── tests/
│   ├── conftest.py
│   ├── test_transforms.py     ← 32 transform tests
│   ├── test_generator.py      ← 14 generator tests
│   ├── test_evaluator.py      ← 17 evaluator tests
│   ├── test_puzzles.py        ← 23 puzzle/hint tests
│   └── test_scoring.py        ← 24 scoring & API tests
│
└── examples/
    └── basic_usage.py         ← Walkthrough demo
```

---

## Full Puzzle List

### Beginner (difficulty = 1)

| ID | Title | Concept Taught |
|----|-------|----------------|
| `line_01` | The Straight Path | Linear relationship (positive slope) |
| `line_02` | Going Down | Linear relationship (negative slope) |
| `square_01` | The Squaring Machine | Feature transform: x² |
| `sqrt_01` | The Shrinking Returns | Feature transform: √x |
| `log_01` | The Logarithm Lab | Feature transform: log(x) |
| `distractor_01` | The Red Herrings | Irrelevant features — only x1 matters |

### Intermediate (difficulty = 2)

| ID | Title | Concept Taught |
|----|-------|----------------|
| `almost_linear_01` | Almost a Straight Line | y = x + 0.5sin(x) — multi-feature |
| `almost_linear_02` | Wiggly Line | y = 2x + sin(x) with small noise |
| `reciprocal_01` | The Shrinking Giant | Feature transform: 1/x |
| `piecewise_01` | The Switch | Piecewise flat — decision tree wins |
| `piecewise_02` | Two Slopes | Piecewise linear — decision tree wins |
| `periodic_01` | The Wave | y = sin(x) — periodicity |
| `periodic_02` | The Hidden Week | y = sin(2πx/7) — 7-day cycle |
| `product_01` | The Hidden Combination | y = x1 × x2 — interaction feature |
| `ratio_01` | The Relative Measure | y = x1 / x2 — ratio feature |

### Challenge (difficulty = 3)

| ID | Title | Concept Taught |
|----|-------|----------------|
| `circle_01` | Inside the Circle | 2D classification with distance feature |
| `distance_01` | How Far From Zero? | y = √(x1² + x2²) — geometry |
| `staircase_01` | The Staircase | Discrete steps — decision tree |
| `cubic_01` | The Cube | Feature transform: x³ |
| `boss_piecewise` | Three Slopes | Three-segment piecewise — boss puzzle |
| `boss_multi` | Product Plus Linear | y = x1·x2 + 2x3 — combine lessons |
| `boss_sin_sum` | Two Waves | y = sin(x) + 0.5·sin(3x) — harmonics |
| `boss_multi_feat` | Three Transformations | sin(x1) + x2² + 3x3 — three transforms |
| `period_boss` | Noisy Week with Distractor | Weekly cycle with noise + irrelevant feature |
| `piecewise_03` | The Three Zones | Three distinct regions — boss tree puzzle |

---

## How to Add a New Puzzle

### Step 1: Define the function in `generator.py`

```python
def _fn_my_function(rng, n, p):
    """y = sin(x) * x"""
    lo, hi = p.get("x_min", 0.0), p.get("x_max", 10.0)
    x = rng.uniform(lo, hi, n)
    y = np.sin(x) * x
    return {"x": x}, y

# Add to the registry
_FUNCTION_REGISTRY["my_function"] = _fn_my_function
```

### Step 2: Define the puzzle in `puzzles.py`

```python
_MY_PUZZLE = _p(
    id="my_puzzle_01",
    title="The Swinging Ramp",
    category="feature_transform",
    difficulty=2,
    description="Something waves AND grows at the same time...",
    input_features=["x"],
    function=FunctionSpec(
        type="my_function",
        parameters={"x_min": 0.0, "x_max": 10.0},
    ),
    noise_std=0.0,
    allowed_transforms=["identity", "sin", "cos", "square"],
    allowed_binary_transforms=[],
    intended_model="linear_regression",
    solution_features=["sin:x", "identity:x"],  # or whatever
    hints=[
        "The output grows but also oscillates.",
        "Try combining two features: one for the growth, one for the waves.",
        "sin(x) × x — but our library doesn't have that directly. Explore!",
    ],
    explanation="...",
    real_world_connection="...",
)
```

### Step 3: Register it

Add your puzzle to the list at the bottom of `puzzles.py`:

```python
PUZZLE_REGISTRY: dict[str, Puzzle] = {
    p.id: p
    for p in [
        # ... existing puzzles ...
        _MY_PUZZLE,   # ← add here
    ]
}
```

### Step 4: Write a test

Add a test to `tests/test_generator.py` or `tests/test_evaluator.py` verifying the puzzle generates correct data and the intended solution achieves high R².

---

## Puzzle Schema

Each puzzle is a `Puzzle` dataclass that can be serialised to JSON:

```python
puzzle = get_puzzle("square_01")
print(puzzle.to_json())
```

```json
{
  "id": "square_01",
  "title": "The Squaring Machine",
  "category": "feature_transform",
  "difficulty": 1,
  "description": "The output y grows much faster than x...",
  "input_features": ["x"],
  "function": {
    "type": "quadratic",
    "parameters": {"a": 1.0, "b": 0.0, "c": 0.0, "x_min": -5.0, "x_max": 5.0}
  },
  "noise_std": 0.0,
  "allowed_transforms": ["identity", "square", "sqrt", "sin", "log"],
  "allowed_binary_transforms": [],
  "intended_model": "linear_regression",
  "solution_features": ["square:x"],
  "hints": [
    "Plot y against x. Does the curve bend upward?",
    "What if you tried x²?",
    "Use the 'square' transformation: x² perfectly straightens this."
  ],
  "explanation": "The rule was y = x²...",
  "real_world_connection": "Kinetic energy = ½mv²..."
}
```

> **Note:** `solution_features` is **not** sent to the player; it is used server-side to validate submissions.

---

## Feature Specification Format

When calling `evaluate_submission`, the `features` parameter accepts three formats:

### 1. Plain column name
```python
features=["x"]          # use raw feature x
features=["x1", "x2"]  # use two raw features
```

### 2. Transform shorthand `"transform:column"`
```python
features=["square:x"]       # x²
features=["sin:x"]          # sin(x)
features=["log:x"]          # log(x)
features=["sin:x1", "identity:x2"]  # multiple features
```

### 3. Binary transform dict
```python
features=[{"binary": "multiply", "a": "x1", "b": "x2"}]  # x1 × x2
features=[{"binary": "divide",   "a": "x1", "b": "x2"}]  # x1 / x2
features=[{"binary": "distance", "a": "x1", "b": "x2"}]  # √(x1²+x2²)
```

### Available transforms

| Key | Description |
|-----|-------------|
| `identity` | x (no change) |
| `square` | x² |
| `cube` | x³ |
| `sqrt` | √x |
| `abs` | \|x\| |
| `log` | log(x) |
| `log2` | log₂(x) |
| `reciprocal` | 1/x |
| `sin` | sin(x) |
| `cos` | cos(x) |
| `sin_2pi` | sin(2π·x) |
| `cos_2pi` | cos(2π·x) |
| `sin_period7` | sin(2π·x/7) |
| `cos_period7` | cos(2π·x/7) |
| `exp` | eˣ |
| `floor10` | floor(x/10)·10 |
| `step` | 1 if x≥0 else 0 |

| Binary key | Description |
|------------|-------------|
| `multiply` | x1 × x2 |
| `divide` | x1 / x2 |
| `add` | x1 + x2 |
| `subtract` | x1 − x2 |
| `distance` | √(x1² + x2²) |

---

## Scoring System

| Action | Points |
|--------|--------|
| Correct model type | +50 |
| Correct feature(s) — R² ≥ 0.92 | +100 |
| Quality bonus (scales R² 0.92→1.0) | up to +50 |
| Wrong submission | −5 |
| Each hint used | −20 |

Score never goes below 0.

```python
from blackbox_game import PlayerSession, compute_score

session = PlayerSession(puzzle_id="square_01")

# Student requests a hint
session.use_hint()

# Wrong attempt
result = compute_score(session, model_correct=True, fit_quality=0.3)
print(result["total_score"])   # 0 (penalty, clamped)

# Correct attempt
result = compute_score(session, model_correct=True, fit_quality=1.0)
print(result["total_score"])   # 200 − 20 (hint) = 180
```

---

## How the Frontend Can Consume This API

This is a **pure Python backend** with no web framework. Your frontend should wrap it however makes sense (Flask, FastAPI, Django, etc.).

### Example FastAPI wrapper (not included, for reference)

```python
from fastapi import FastAPI
from blackbox_game import (
    list_puzzles, get_puzzle, generate_dataset,
    evaluate_submission, compare_models, get_hint
)

app = FastAPI()

@app.get("/puzzles")
def api_list_puzzles(difficulty: int | None = None):
    return list_puzzles(difficulty=difficulty)

@app.get("/puzzles/{puzzle_id}")
def api_get_puzzle(puzzle_id: str):
    puzzle = get_puzzle(puzzle_id)
    return puzzle.to_dict()

@app.get("/puzzles/{puzzle_id}/data")
def api_get_data(puzzle_id: str, n_samples: int = 100, seed: int = 42):
    puzzle = get_puzzle(puzzle_id)
    ds = generate_dataset(puzzle, n_samples=n_samples, seed=seed)
    return {"X": ds["X"].to_dict(orient="list"), "y": ds["y"].tolist()}

@app.post("/puzzles/{puzzle_id}/submit")
def api_submit(puzzle_id: str, body: dict):
    return evaluate_submission(
        puzzle_id=puzzle_id,
        model=body["model"],
        features=body["features"],
    )

@app.get("/puzzles/{puzzle_id}/hints/{hint_index}")
def api_get_hint(puzzle_id: str, hint_index: int):
    puzzle = get_puzzle(puzzle_id)
    return get_hint(puzzle, hint_index)
```

### Key design decisions for the frontend

- **Session management**: `PlayerSession` is an in-memory object. A web backend should serialize it to a database or pass it back to the client as JSON between requests.
- **Dataset**: Generated deterministically from `seed=42` by default. The frontend does not need to store the data.
- **Solution features**: Never returned to the client; only used server-side during validation.
- **Explanations**: Only returned inside `evaluate_submission` when `is_correct=True`.
- **Hint cost**: The −20 penalty is tracked server-side in `PlayerSession.hints_used`.

---

## Visualization (Optional)

```python
import matplotlib.pyplot as plt
from blackbox_game import get_puzzle, generate_dataset
from blackbox_game.viz import (
    plot_xy,
    plot_residuals,
    plot_tree_predictions,
    plot_transform_comparison,
)

puzzle = get_puzzle("square_01")
ds = generate_dataset(puzzle, n_samples=100)
x = ds["X"]["x"].values
y = ds["y"]

# 1. Raw scatter
fig = plot_xy(x, y, xlabel="x", ylabel="y", title="square_01: x vs y")
fig.savefig("scatter.png")

# 2. Compare transforms side-by-side
fig = plot_transform_comparison(x, y, ["identity", "square", "sqrt", "log", "sin"])
fig.savefig("transforms.png")

# 3. Residuals after fitting square:x
import numpy as np
x_sq = x ** 2
fig = plot_residuals(x_sq, y, title="Residuals after using x²")
fig.savefig("residuals.png")

# 4. Decision tree for a piecewise puzzle
puzzle2 = get_puzzle("piecewise_01")
ds2 = generate_dataset(puzzle2, n_samples=200)
fig = plot_tree_predictions(ds2["X"]["x"].values, ds2["y"])
fig.savefig("tree.png")
```

---

## Assumptions About the Frontend

1. **Session state**: The frontend is responsible for persisting `PlayerSession` between requests (e.g., store as JSON in a database or cookie).
2. **Data format**: `generate_dataset` returns a `pandas.DataFrame`. When serialising to JSON for the frontend, call `.to_dict(orient="list")` on the DataFrame.
3. **Authentication/leaderboard**: Not included. The backend assumes a single session per call.
4. **Seed**: Using the same seed (default 42) for all players ensures everyone gets the same data for a given puzzle.
5. **Time tracking**: `PlayerSession.start_time` is set at creation. Pass the session object across requests to track elapsed time.
6. **Binary features**: The `features` list supports dict entries like `{"binary": "multiply", "a": "x1", "b": "x2"}`. The frontend must send these as JSON objects.

---

## License

MIT — free to use and modify for educational purposes.
