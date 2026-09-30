"""
basic_usage.py — A walkthrough of the blackbox_game API.

Run this file to see the game engine in action:

    python examples/basic_usage.py

This example simulates a student solving three puzzles step-by-step,
including wrong attempts, hint usage, and final correct submission.
"""

from blackbox_game import (
    list_puzzles,
    get_puzzle,
    generate_dataset,
    apply_transform,
    evaluate_submission,
    compare_models,
    get_hint,
    get_explanation,
    PlayerSession,
    list_transforms,
)


def separator(title: str = ""):
    width = 60
    if title:
        print(f"\n{'─' * 10} {title} {'─' * (width - len(title) - 12)}")
    else:
        print("─" * width)


# ===========================================================================
# 0. Discover available puzzles
# ===========================================================================

separator("0. Puzzle Catalogue")
all_puzzles = list_puzzles()
print(f"Total puzzles: {len(all_puzzles)}\n")
print(f"{'ID':<25} {'Difficulty':<12} {'Category':<20} Title")
print("-" * 80)
for p in all_puzzles:
    diff_label = {1: "Beginner", 2: "Intermediate", 3: "Challenge"}.get(
        p["difficulty"], str(p["difficulty"])
    )
    print(f"{p['id']:<25} {diff_label:<12} {p['category']:<20} {p['title']}")


# ===========================================================================
# 1. PUZZLE: line_01 — Straight line (Beginner)
# ===========================================================================

separator("1. Beginner — The Straight Path (line_01)")

puzzle = get_puzzle("line_01")
print(f"Title      : {puzzle.title}")
print(f"Description: {puzzle.description}")
print(f"Features   : {puzzle.input_features}")
print(f"Transforms : {puzzle.allowed_transforms}")

dataset = generate_dataset(puzzle, n_samples=100, seed=42)
X = dataset["X"]
y = dataset["y"]
print(f"\nDataset shape: X={X.shape}, y={y.shape}")
print("First 5 rows:")
print(X.head())
print("First 5 y:", y[:5].round(2))

# Wrong attempt: try x² first
print("\n[Attempt 1] Using x² — wrong transform")
result = evaluate_submission(
    puzzle_id="line_01",
    model="linear_regression",
    features=["square:x"],
)
print(f"  is_correct: {result['is_correct']}, R²: {result.get('r2', 'N/A')}")
print(f"  message: {result['message']}")

# Correct: use x directly
print("\n[Attempt 2] Using x directly")
result = evaluate_submission(
    puzzle_id="line_01",
    model="linear_regression",
    features=["identity:x"],
)
print(f"  is_correct: {result['is_correct']}, R²: {result.get('r2', 'N/A')}")
print(f"  message: {result['message']}")
if result["is_correct"]:
    exp = result["explanation"]
    print(f"\n  Explanation: {exp['explanation']}")
    print(f"  Real-world : {exp['real_world_connection']}")


# ===========================================================================
# 2. PUZZLE: square_01 — Quadratic (Beginner)
# ===========================================================================

separator("2. Beginner — The Squaring Machine (square_01)")

puzzle = get_puzzle("square_01")
session = PlayerSession(puzzle_id="square_01")

print("\n[Hint 1 requested]")
hint = get_hint(puzzle, session.use_hint())
print(f"  Hint: {hint['text']}  (cost: {hint['cost']} pts)")

print("\n[Attempt 1] Using raw x — student's first guess")
result = evaluate_submission(
    puzzle_id="square_01",
    model="linear_regression",
    features=["identity:x"],
    session=session,
)
print(f"  is_correct: {result['is_correct']}, R²: {result.get('r2', 'N/A')}")
print(f"  message: {result['message']}")

print("\n[Hint 2 requested]")
hint2 = get_hint(puzzle, session.use_hint())
print(f"  Hint: {hint2['text']}  (cost: {hint2['cost']} pts)")

print("\n[Attempt 2] Using x² — correct!")
result = evaluate_submission(
    puzzle_id="square_01",
    model="linear_regression",
    features=["square:x"],
    session=session,
)
print(f"  is_correct: {result['is_correct']}, R²: {result.get('r2', 'N/A')}")
print(f"  Score breakdown: {result['score_result']}")


# ===========================================================================
# 3. PUZZLE: piecewise_01 — Decision tree (Intermediate)
# ===========================================================================

separator("3. Intermediate — The Switch (piecewise_01)")

puzzle = get_puzzle("piecewise_01")
print(f"Intended model: {puzzle.intended_model}")

print("\n[Attempt 1] Student tries linear regression")
result_lr = evaluate_submission(
    puzzle_id="piecewise_01",
    model="linear_regression",
    features=["identity:x"],
)
print(f"  is_correct: {result_lr['is_correct']}, R²: {result_lr.get('r2', 'N/A')}")
print(f"  message: {result_lr['message']}")

print("\n[Model Comparison] Which model fits better?")
comparison = compare_models("piecewise_01", features=["identity:x"])
print(f"  Linear R²     : {comparison['linear_regression'].get('r2', 'N/A')}")
print(f"  Tree R²       : {comparison['decision_tree'].get('r2', 'N/A')}")
print(f"  Recommended   : {comparison['recommended_model']}")
print(f"  Summary       : {comparison['summary']}")

print("\n[Attempt 2] Student tries decision tree — correct!")
result_dt = evaluate_submission(
    puzzle_id="piecewise_01",
    model="decision_tree",
    features=["identity:x"],
)
print(f"  is_correct: {result_dt['is_correct']}, R²: {result_dt.get('r2', 'N/A')}")
print(f"  message: {result_dt['message']}")


# ===========================================================================
# 4. PUZZLE: product_01 — Interaction feature (Intermediate)
# ===========================================================================

separator("4. Intermediate — Hidden Combination (product_01)")

print("\n[Attempt 1] Try x1 alone")
r1 = evaluate_submission("product_01", "linear_regression", ["identity:x1"])
print(f"  R² with x1 only: {r1.get('r2', 'N/A')}")

print("\n[Attempt 2] Try x1 × x2")
r2 = evaluate_submission(
    "product_01",
    "linear_regression",
    [{"binary": "multiply", "a": "x1", "b": "x2"}],
)
print(f"  R² with x1*x2: {r2.get('r2', 'N/A')}, correct: {r2['is_correct']}")


# ===========================================================================
# 5. Available transforms (for frontend palette)
# ===========================================================================

separator("5. Transform Palette (for frontend display)")
transforms = list_transforms()
print(f"{'Key':<18} Description")
print("-" * 40)
for t in transforms:
    print(f"{t['key']:<18} {t['description']}")


separator("Done!")
print("\nAll examples complete. Run `pytest tests/` to execute the full test suite.")
