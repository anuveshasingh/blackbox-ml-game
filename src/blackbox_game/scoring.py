"""
scoring.py — Deterministic scoring engine with fuzzy feature matching.

Scoring overview
----------------
Exact correct answer (R² ≥ threshold AND correct model):
    +50  model bonus
    +100 feature bonus
    +0–50 quality bonus (scales from R²=0.92 → R²=1.0)

Fuzzy power-family match (wrong power, but same column, correct model):
    +25  partial model bonus
    +int(100 / |n - m|)  partial feature bonus  (capped at 99)

Wrong submission:
    −5 penalty

Scale invariance:
    cx and x are treated identically — linear regression absorbs the constant,
    so R² is invariant to feature scaling.  No special code needed.

Session tracking
----------------
PlayerSession is a lightweight in-memory object the frontend can serialise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCORE_CORRECT_MODEL   = 50
SCORE_CORRECT_FEATURE = 100
SCORE_QUALITY_BONUS   = 50
PENALTY_WRONG_SUBMIT  = 5

CORRECT_R2_THRESHOLD       = 0.92
CORRECT_ACCURACY_THRESHOLD = 0.90

# ---------------------------------------------------------------------------
# Fuzzy matching — power family
# ---------------------------------------------------------------------------

#: Maps transform key → real-number exponent.
#: Only transforms in this dict participate in power-family fuzzy matching.
POWER_MAP: dict[str, float] = {
    "reciprocal": -1.0,
    "sqrt":        0.5,
    "identity":    1.0,
    "square":      2.0,
    "cube":        3.0,
}


def _parse_power_feature(feat: Union[str, dict]) -> tuple[float, str] | None:
    """
    Extract (power, column_name) from a feature spec if it is a power transform.

    Returns None for binary dicts or non-power transforms.
    Plain column names (e.g. "x") are treated as identity (power=1).
    """
    if isinstance(feat, dict):
        return None
    if not isinstance(feat, str):
        return None
    if ":" in feat:
        transform, col = feat.split(":", 1)
        transform = transform.strip()
        col = col.strip()
        if transform in POWER_MAP:
            return (POWER_MAP[transform], col)
        return None
    else:
        # Plain column name → identity transform
        return (1.0, feat)


def check_fuzzy_match(
    submitted_features: list,
    solution_features: list,
) -> tuple[float, str]:
    """
    Determine whether submitted features are fuzzy power-family matches
    for the solution features.

    The fuzzy fraction encodes how close the submitted power is to the
    correct power.  For a single-feature puzzle:

        fuzzy_fraction = 1 / |submitted_power − solution_power|

    This fraction is then multiplied by SCORE_CORRECT_FEATURE and capped
    at 99 (always less than a full-correct answer).

    For multi-feature puzzles the fractions are averaged across matched
    features (normalised by the number of solution features).

    Parameters
    ----------
    submitted_features : list of feature specs submitted by the player
    solution_features  : list of canonical correct feature specs

    Returns
    -------
    (fuzzy_fraction, description)
        fuzzy_fraction in [0, 1)  — multiply by SCORE_CORRECT_FEATURE for points
        description               — human-readable explanation
    """
    if not submitted_features or not solution_features:
        return 0.0, ""

    used_sol_indices: set[int] = set()
    matched_fracs: list[float] = []
    matched_descs: list[str] = []

    for sub_feat in submitted_features:
        sub_info = _parse_power_feature(sub_feat)
        if sub_info is None:
            continue
        sub_p, sub_col = sub_info

        best_frac = 0.0
        best_desc = ""
        best_idx  = -1

        for j, sol_feat in enumerate(solution_features):
            if j in used_sol_indices:
                continue
            sol_info = _parse_power_feature(sol_feat)
            if sol_info is None:
                continue
            sol_p, sol_col = sol_info

            if sub_col != sol_col:
                continue  # different column — not a match

            diff = abs(sub_p - sol_p)
            if diff == 0:
                frac = 0.99  # same power but R² still low → multi-feature puzzle / wrong model
                desc = f"Same power family for '{sub_col}' (check model type or add missing features)"
            else:
                frac = min(0.99, 1.0 / diff)
                desc = (
                    f"Close on '{sub_col}': you used x^{sub_p:.2g}, "
                    f"answer uses x^{sol_p:.2g}. "
                    f"Fuzzy score = 1/|{sub_p:.2g}−{sol_p:.2g}| = {frac:.2f}"
                )

            if frac > best_frac:
                best_frac = frac
                best_desc = desc
                best_idx  = j

        if best_frac > 0 and best_idx >= 0:
            matched_fracs.append(best_frac)
            matched_descs.append(best_desc)
            used_sol_indices.add(best_idx)

    if not matched_fracs:
        return 0.0, ""

    # Average across solution features (penalises missing features)
    avg_frac = sum(matched_fracs) / max(len(solution_features), 1)
    return min(0.99, avg_frac), " | ".join(matched_descs)


# ---------------------------------------------------------------------------
# Player session
# ---------------------------------------------------------------------------

@dataclass
class PlayerSession:
    """
    Tracks one player's progress through a single puzzle.

    Attributes
    ----------
    puzzle_id  : puzzle being played
    attempts   : total submissions made
    solved     : True once a correct submission is accepted
    start_time : Unix timestamp when the session began
    end_time   : Unix timestamp when first solved (or None)
    score      : accumulated score (never drops below 0)
    """
    puzzle_id: str
    attempts: int = 0
    solved: bool = False
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    score: int = 0

    @property
    def time_taken(self) -> float | None:
        if self.end_time is None:
            return None
        return round(self.end_time - self.start_time, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["time_taken"] = self.time_taken
        return d


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def compute_score(
    session: PlayerSession,
    model_correct: bool,
    fit_quality: float,
    task: str = "regression",
    fuzzy_info: tuple[float, str] | None = None,
) -> dict:
    """
    Compute the score for a single submission attempt and update ``session``.

    Parameters
    ----------
    session       : PlayerSession to update in-place
    model_correct : whether the submitted model matches the intended model
    fit_quality   : R² (regression) or accuracy (classification)
    task          : "regression" or "classification"
    fuzzy_info    : (fuzzy_fraction, description) from check_fuzzy_match, or None

    Returns
    -------
    dict
        is_correct, is_fuzzy, points_earned, total_score,
        model_bonus, feature_bonus, quality_bonus, penalty, message
    """
    session.attempts += 1

    threshold = (
        CORRECT_R2_THRESHOLD if task == "regression"
        else CORRECT_ACCURACY_THRESHOLD
    )
    feature_correct = fit_quality >= threshold
    is_correct = model_correct and feature_correct
    is_fuzzy   = False

    model_bonus   = 0
    feature_bonus = 0
    quality_bonus = 0
    penalty       = 0

    if is_correct:
        model_bonus   = SCORE_CORRECT_MODEL
        feature_bonus = SCORE_CORRECT_FEATURE
        q = max(0.0, (fit_quality - threshold) / (1.0 - threshold))
        quality_bonus = round(q * SCORE_QUALITY_BONUS)
        if not session.solved:
            session.solved  = True
            session.end_time = time.time()

    elif fuzzy_info and fuzzy_info[0] > 0 and model_correct:
        # Partial credit: close but not quite
        is_fuzzy       = True
        frac           = fuzzy_info[0]
        model_bonus    = SCORE_CORRECT_MODEL // 2           # 25 pts
        feature_bonus  = min(
            SCORE_CORRECT_FEATURE - 1,
            int(SCORE_CORRECT_FEATURE * frac)               # 1–99 pts
        )
        quality_bonus  = 0

    else:
        penalty = PENALTY_WRONG_SUBMIT

    points_earned = model_bonus + feature_bonus + quality_bonus - penalty
    session.score = max(0, session.score + points_earned)

    message = _compose_message(
        is_correct=is_correct,
        is_fuzzy=is_fuzzy,
        model_correct=model_correct,
        feature_correct=feature_correct,
        fit_quality=fit_quality,
        task=task,
        fuzzy_desc=fuzzy_info[1] if fuzzy_info else "",
    )

    return {
        "is_correct":    is_correct,
        "is_fuzzy":      is_fuzzy,
        "points_earned": points_earned,
        "total_score":   session.score,
        "model_bonus":   model_bonus,
        "feature_bonus": feature_bonus,
        "quality_bonus": quality_bonus,
        "penalty":       penalty,
        "message":       message,
    }


# ---------------------------------------------------------------------------
# Message composer
# ---------------------------------------------------------------------------

def _compose_message(
    is_correct: bool,
    is_fuzzy: bool,
    model_correct: bool,
    feature_correct: bool,
    fit_quality: float,
    task: str,
    fuzzy_desc: str,
) -> str:
    metric = "R²" if task == "regression" else "accuracy"
    val    = round(fit_quality, 3)

    if is_correct:
        return (
            f"✓ Correct! {metric} = {val}. "
            "Check the explanation to understand why this works."
        )

    if is_fuzzy:
        return f"≈ Fuzzy match — partial credit. {fuzzy_desc}"

    parts = []
    if not model_correct:
        parts.append(
            "Wrong model type. Think about whether the relationship "
            "is a straight line or has kinks/regions."
        )
    if not feature_correct:
        if task == "regression":
            if val < 0:
                parts.append(
                    f"R² = {val} — worse than predicting the mean. "
                    "Try a completely different transformation."
                )
            elif val < 0.40:
                parts.append(f"R² = {val} — very weak fit.")
            elif val < 0.70:
                parts.append(f"R² = {val} — some structure explained, but much remains.")
            else:
                parts.append(f"R² = {val} — getting closer, but not tight enough yet.")
        else:
            parts.append(f"Accuracy = {val} — not quite there.")

    return " ".join(parts) if parts else "Not quite. Keep experimenting."
