# Blackbox ML Game

> Clone the repo. Look at the data. Experiment with transformations. Discover the hidden rule. Submit your answer.

No lectures. No maths homework. Just puzzles.

---

## What Is This?

A puzzle game where a **hidden black-box function** maps inputs to outputs:

```
x  →  ??? BLACK BOX ???  →  y
```

You can see the data. You cannot see the function. Your job is to:

1. Explore which features (raw or transformed) explain `y`
2. Choose the right model (`linear_regression` or `decision_tree`)
3. Write your answer in a JSON file
4. Run the scorer and see your score

The game teaches **feature engineering** and model selection through discovery.  
Target audience: first-year IIT Delhi students 

The exact answer key is documented separately in [GROUND_TRUTH.md](GROUND_TRUTH.md).

---

## Quick Start (after cloning)

```bash
# 1. Install
pip install -e .

# 2. List all puzzles
python play.py list

# 3. Look at a puzzle and its data
python play.py show puzzle_03

# 3a. Also save the generated data plot under outputs/puzzle_03/puzzle_03_base.png
python play.py show puzzle_03 --plot

# 4. See all available transformations
python play.py transforms

# 5. Generate a blank answers template
python play.py template

# 6. Edit my_answers.json with your guesses, then submit
python play.py submit my_answers.json
```

### Evaluate custom points

You can evaluate your own input points instead of using randomly generated data. Put one point per line in a text file, with values in the puzzle's input-feature order. Spaces or commas may separate values; a header and lines beginning with `#` are also allowed.

For a puzzle with `x1` and `x2` inputs, `points.txt` could contain:

```text
x1,x2
0,1
2,3
-1,4
```

Evaluate the hidden function and write a CSV containing the input columns and `y`:

```bash
python play.py points puzzle_17 --input points.txt
```

By default this creates an output under `outputs/puzzle_17/` for `distance_01`. Add `--plot` to also create the neighboring PNG:

```bash
python play.py points puzzle_17 --input points.txt --plot
```

To fit a model to the supplied points and export `y`, `prediction`, and `residual` (`y - prediction`):

```bash
python play.py residuals puzzle_17 --input points.txt \
    --features square:x1 square:x2 --plot
```

This creates a residual CSV and, with `--plot`, the neighboring PNG under the matching `outputs/puzzle_XX/` folder. The generic puzzle number prevents the puzzle name from being revealed by output filenames. Use `--output FILE` with either command to choose a different CSV path. Use `--no-noise` when evaluating a puzzle whose configured output includes noise and you want the underlying noiseless function.

Generated formula outputs and plots use reproducible uniform noise bounded by `epsilon = 1.0` with NumPy seed `42`. CSV values include formula noise; plot jitter is visual noise applied on top of those values.

---

## How the Game Works

### Step 1 — Read the puzzle

```bash
python play.py show puzzle_03
```

You'll see:
- A cryptic description (no spoilers)
- 12 rows of sample data
- The list of transforms you're allowed to use
- With `--plot`, a base scatter plot saved under `outputs/puzzle_XX/puzzle_XX_base.png`

### Step 2 — Fill in your answer

Open `my_answers.json`. Find the puzzle entry and fill in the `"model"` and `"features"` fields:

```json
{
    "puzzle_id": "puzzle_03",
  "model": "linear_regression",
  "features": ["square:x"]
}
```

`square:x1` means: take the input column `x1`, square every value, and give
those transformed values to the model. For example, `x1 = -3, 2` becomes
`square:x1 = 9, 4`. It is useful when the relationship depends on `x1²`.

### Step 3 — Submit

```bash
python play.py submit my_answers.json
```

You'll see:

```
✓  The Bend in the Road          [Beginner]
    ID: puzzle_03                  model: linear_regression
   features: ["square:x"]
   R² = 1.0000   score = 200
   ✓ Correct! R² = 1.0.
   Explanation: The rule was y = x²...
```

---

## Input File Format

The answers file is a **JSON file** — either a single object or an array of objects.

### Single submission
```json
{
    "puzzle_id": "puzzle_03",
    "model":     "linear_regression",
    "features":  ["square:x"]
}
```

### Multiple submissions (recommended)
```json
[
    {
        "puzzle_id": "puzzle_01",
        "model":     "linear_regression",
        "features":  ["identity:x"]
    },
    {
        "puzzle_id": "puzzle_03",
        "model":     "linear_regression",
        "features":  ["square:x"]
    },
    {
        "puzzle_id": "puzzle_14",
        "model":     "linear_regression",
        "features":  [{"binary": "multiply", "a": "x1", "b": "x2"}]
    }
]
```

### Required fields

| Field | Type | Values |
|-------|------|--------|
| `puzzle_id` | string | any ID from `python play.py list` |
| `model` | string | `"linear_regression"` or `"decision_tree"` |
| `features` | array | see feature format below |

### Feature format

Each element in `"features"` can be one of three forms:

#### 1. Raw column name
```json
"x"
"x1"
```
Uses the column as-is (equivalent to `"identity:x"`).

#### 2. Transform shorthand: `"transform_key:column_name"`
```json
"identity:x"      // x  (no change)
"square:x"        // x²
"cube:x"          // x³
"sqrt:x"          // √x
"abs:x"           // |x|
"log:x"           // log(x)
"reciprocal:x"    // 1/x
"sin:x"           // sin(x)
"cos:x"           // cos(x)
"sin_period7:x"   // sin(2π·x/7)
"cos_period7:x"   // cos(2π·x/7)
```

#### 3. Binary transform dict (for combining two columns)
```json
{"binary": "multiply",  "a": "x1", "b": "x2"}   // x1 × x2
{"binary": "divide",    "a": "x1", "b": "x2"}   // x1 / x2
{"binary": "add",       "a": "x1", "b": "x2"}   // x1 + x2
{"binary": "distance",  "a": "x1", "b": "x2"}   // √(x1² + x2²)
```

#### Multiple features (e.g. for multi-term relationships)
```json
["square:x", "identity:x"]           // x² and x together
["sin:x",    "cos:x"]                 // sin(x) and cos(x) together
["sin:x1",   "square:x2", "identity:x3"]
```

> **Scale invariance:** If the correct feature is `x`, submitting `3x` also works — linear
> regression absorbs the constant coefficient automatically.

---

## Available Transforms

Run `python play.py transforms` for the full list with descriptions.

| Key | Formula | Power family? |
|-----|---------|:---:|
| `identity` | x | ✓ (x¹) |
| `square` | x² | ✓ (x²) |
| `cube` | x³ | ✓ (x³) |
| `sqrt` | √x | ✓ (x^0.5) |
| `reciprocal` | 1/x | ✓ (x^−1) |
| `abs` | \|x\| | |
| `log` | ln(x) | |
| `log2` | log₂(x) | |
| `sin` | sin(x) | |
| `cos` | cos(x) | |
| `sin_2pi` | sin(2π·x) | |
| `cos_2pi` | cos(2π·x) | |
| `sin_period7` | sin(2π·x/7) | |
| `cos_period7` | cos(2π·x/7) | |
| `exp` | eˣ | |
| `floor10` | floor(x/10)·10 | |
| `step` | 1 if x≥0 else 0 | |

---

## Scoring System

| Outcome | Points |
|---------|--------|
| Correct model type | +50 |
| Correct features (R² ≥ 0.92) | +100 |
| Quality bonus (R² from 0.92 → 1.0) | up to +50 |
| Wrong submission | −5 |

**Total score for a perfect first attempt: 200 points.**

### Fuzzy matching (partial credit)

If you use a power-family transform with the **wrong exponent**, you still earn partial credit:

```
partial feature score = int(100 / |submitted_power − correct_power|)
```

Examples (correct answer is `square:x`, i.e. x²):

| You submit | Power diff | Feature bonus |
|------------|-----------|---------------|
| `cube:x` (x³) | \|3−2\| = 1 | 99 pts |
| `identity:x` (x¹) | \|1−2\| = 1 | 99 pts |
| `sqrt:x` (x^0.5) | \|0.5−2\| = 1.5 | 66 pts |
| `reciprocal:x` (x^−1) | \|−1−2\| = 3 | 33 pts |
| `sin:x` | not power family | 0 pts |

> Score is **never negative**.

### What is R²?

| R² | Meaning |
|----|---------|
| 1.0 | Perfect — your feature explains y completely |
| 0.92+ | Correct (passes the threshold) |
| 0.7–0.92 | Some structure captured; try a different transform |
| 0–0.7 | Weak fit |
| Negative | Your feature is worse than predicting the average |

---

## All 25 Puzzles

### Beginner (6 puzzles)

| ID | Title | Concept |
|----|-------|---------|
| `puzzle_01` | Obedient Numbers | Positive linear relationship |
| `puzzle_02` | The Reluctant Ascent | Negative linear relationship |
| `puzzle_03` | The Bend in the Road | Feature transform: x² |
| `puzzle_04` | Momentum Decay | Feature transform: √x |
| `puzzle_05` | The Compressed Universe | Feature transform: log(x) |
| `puzzle_06` | Four Suspects | Identify the one useful feature |

### Intermediate (9 puzzles)

| ID | Title | Concept |
|----|-------|---------|
| `puzzle_07` | The Imposter Line | y = x + 0.5·sin(x) — use both |
| `puzzle_08` | Static on the Signal | Linear + periodic + noise |
| `puzzle_09` | Vanishing Point | Feature transform: 1/x |
| `puzzle_10` | The Symmetric Grudge | Feature transform: \|x\| |
| `puzzle_11` | The Quarter-Turn | Feature transform: cos(x) |
| `puzzle_12` | The Repeating Rumour | y = sin(x) — periodicity |
| `puzzle_13` | Seven Days of Nothing | y = sin(2πx/7) — weekly cycle |
| `puzzle_14` | The Missing Third Variable | y = x1 × x2 — interaction |
| `puzzle_15` | Speed Without Units | y = x1 / x2 — ratio feature |

### Challenge (10 puzzles)

| ID | Title | Concept |
|----|-------|---------|
| `puzzle_16` | The Exclusion Zone | **Decision tree** — circular boundary |
| `puzzle_17` | The Displacement Field | y = √(x1²+x2²) — geometry |
| `puzzle_18` | Tripling the Problem | Feature transform: x³ |
| `puzzle_19` | The Three Regimes | **Decision tree** — 3 regions |
| `puzzle_20` | The Hidden Tax | y = x1·x2 + 2x3 — combine lessons |
| `puzzle_21` | Interfering Signals | Two superimposed frequencies |
| `puzzle_22` | Chaos in Three Channels | Three different transforms |
| `puzzle_23` | The Noisy Calendar | Weekly cycle + noise + distractor |
| `puzzle_24` | The Bent Wire | y = x²−3x — needs two features |
| `puzzle_25` | The Hidden Angle | y = sin(x)+cos(x) — phase shift |

> **Decision trees:** Only `puzzle_16` and `puzzle_19` require `"model": "decision_tree"`.
> All others use `"model": "linear_regression"`.

---

## Running the Tests

```bash
pytest tests/ -v
```

Expected: **130 tests pass in ~2 seconds.**

---

## Repository Structure

```
blackbox-ml-game/
├── play.py                        ← START HERE — game CLI
├── my_answers.json                ← your answers go here (after `python play.py template`)
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── src/blackbox_game/
│   ├── __init__.py                ← public Python API
│   ├── models.py                  ← Puzzle + FunctionSpec dataclasses
│   ├── transforms.py              ← 17 unary + 5 binary transforms
│   ├── puzzles.py                 ← all 25 puzzle definitions
│   ├── generator.py               ← reproducible dataset generation
│   ├── evaluator.py               ← sklearn model fitting
│   ├── scoring.py                 ← scoring + fuzzy matching engine
│   ├── hints.py                   ← explanation retrieval
│   └── viz.py                     ← optional matplotlib helpers
│
└── tests/
    ├── test_transforms.py
    ├── test_generator.py
    ├── test_evaluator.py
    ├── test_puzzles.py
    └── test_scoring.py
```

---

## For Developers (Python API)

The game can also be used programmatically:

```python
from blackbox_game import (
    get_puzzle, generate_dataset, evaluate_submission, compare_models,
    list_puzzles, list_transforms,
)

# See all puzzles
puzzles = list_puzzles(difficulty=1)   # beginner only

# Get data
puzzle  = get_puzzle("square_01")
dataset = generate_dataset(puzzle, n_samples=100, seed=42)
X = dataset["X"]   # pandas DataFrame
y = dataset["y"]   # numpy array

# Submit a guess
result = evaluate_submission(
    puzzle_id="square_01",
    model="linear_regression",
    features=["square:x"],
)
print(result["is_correct"])   # True
print(result["r2"])           # 1.0
print(result["score_result"]["total_score"])  # 200

# Compare both models
comparison = compare_models("boss_piecewise", features=["identity:x"])
print(comparison["recommended_model"])   # "decision_tree"
```

### Adding a new puzzle

1. **Add a generator function** in `generator.py`:
   ```python
   def _fn_my_puzzle(rng, n, p):
       x = rng.uniform(p.get("x_min", 0), p.get("x_max", 10), n)
       return {"x": x}, np.sin(x) * x
   _FUNCTION_REGISTRY["my_puzzle"] = _fn_my_puzzle
   ```

2. **Define the puzzle** in `puzzles.py`:
   ```python
   _MY_PUZZLE = _p(
       id="my_puzzle_01",
       title="The Swinging Ramp",
       ...
       function=FunctionSpec(type="my_puzzle", parameters={"x_min": 0, "x_max": 10}),
       intended_model="linear_regression",
       solution_features=["sin:x", "identity:x"],
   )
   ```

3. **Add it to `PUZZLE_REGISTRY`** at the bottom of `puzzles.py`.

---

## FAQ

**Q: What does a "fuzzy match" mean?**  
A: You used the right *family* of transform (e.g. a power of x) but the wrong exponent. You get partial credit based on `1/|your_power − correct_power|`.

**Q: Why does `cx` give the same score as `x`?**  
A: Linear regression learns the coefficient automatically. If the true rule is `y = 3x`, submitting `identity:x` is correct — the model learns the 3.

**Q: Can I use multiple features?**  
A: Yes. `"features": ["square:x", "identity:x"]` submits both at once. Linear regression will use them together.

**Q: What seed / sample size is used for scoring?**  
A: Always `n_samples=200, seed=42`. Everyone gets the same data for every puzzle.

**Q: Are there hints?**  
A: No hints in this version. Use `python play.py show <puzzle_id>` to see the data and transforms, then experiment.
