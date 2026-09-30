"""
scoring.py — Deterministic scoring engine for player submissions.

Scoring philosophy
------------------
The game rewards:
  - Picking the right model type                 +50 pts
  - Providing features that produce a good fit   +100 pts (scaled by R²/accuracy)
  - Quality bonus for near-perfect fit           up to +50 pts extra
  - Penalises wrong submissions                  -5 pts per failed attempt
  - Penalises hint usage                         -20 pts per hint requested

Score never goes below 0.

Session tracking
----------------
``PlayerSession`` is a lightweight in-memory object that tracks attempts,
hints used, and time.  The frontend can serialise it to JSON if needed.

Usage
-----
    from blackbox_game.scoring import PlayerSession, compute_score

    session = PlayerSession(puzzle_id="square_01")
    result = compute_score(session, model_correct=True, fit_quality=0.999,
                           task="regression")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCORE_CORRECT_MODEL   = 50
SCORE_CORRECT_FEATURE = 100
SCORE_QUALITY_BONUS   = 50      # max bonus for near-perfect fit
PENALTY_WRONG_SUBMIT  = 5
PENALTY_HINT          = 20

# Thresholds for "correct"
CORRECT_R2_THRESHOLD       = 0.92
CORRECT_ACCURACY_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Player session
# ---------------------------------------------------------------------------

@dataclass
class PlayerSession:
    """
    Tracks one student's progress through a single puzzle.

    Attributes
    ----------
    puzzle_id : str
    attempts : int        total submissions made
    hints_used : int      hints the player has requested
    solved : bool         True once a correct submission is accepted
    start_time : float    Unix timestamp when the session began
    end_time : float | None
    score : int           accumulated score (can only grow after first solve)
    """
    puzzle_id: str
    attempts: int = 0
    hints_used: int = 0
    solved: bool = False
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    score: int = 0

    @property
    def time_taken(self) -> float | None:
        """Elapsed seconds, or None if not yet solved."""
        if self.end_time is None:
            return None
        return round(self.end_time - self.start_time, 2)

    def use_hint(self) -> int:
        """
        Record that the player requested a hint.

        Returns
        -------
        int
            The index (0-based) of the hint they should receive.
        """
        idx = self.hints_used
        self.hints_used += 1
        return idx

    def to_dict(self) -> dict:
        d = asdict(self)
        d["time_taken"] = self.time_taken
        return d


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def compute_score(
    session: PlayerSession,
    model_correct: bool,
    fit_quality: float,
    task: str = "regression",
) -> dict:
    """
    Compute the score for a single submission attempt.

    This function is pure / deterministic: the same inputs always
    produce the same output.  It also updates ``session`` in-place.

    Parameters
    ----------
    session : PlayerSession
        The player's current session object.
    model_correct : bool
        Whether the submitted model type matches ``puzzle.intended_model``.
    fit_quality : float
        R² for regression tasks, accuracy for classification tasks.
        Should be in [-∞, 1] for regression or [0, 1] for classification.
    task : str
        "regression" or "classification".

    Returns
    -------
    dict with keys:
        is_correct, points_earned, total_score, model_bonus,
        feature_bonus, quality_bonus, penalty, message
    """
    session.attempts += 1
    penalty = 0
    model_bonus = 0
    feature_bonus = 0
    quality_bonus = 0

    # Determine whether the feature is "correct"
    threshold = (
        CORRECT_R2_THRESHOLD
        if task == "regression"
        else CORRECT_ACCURACY_THRESHOLD
    )
    feature_correct = fit_quality >= threshold

    is_correct = model_correct and feature_correct

    if is_correct:
        model_bonus = SCORE_CORRECT_MODEL
        feature_bonus = SCORE_CORRECT_FEATURE

        # Quality bonus: scales from 0 to SCORE_QUALITY_BONUS
        # R² of 0.92 → 0 bonus, R² of 1.0 → full bonus
        if task == "regression":
            q = max(0.0, (fit_quality - threshold) / (1.0 - threshold))
        else:
            q = max(0.0, (fit_quality - threshold) / (1.0 - threshold))
        quality_bonus = round(q * SCORE_QUALITY_BONUS)

        if not session.solved:
            session.solved = True
            session.end_time = time.time()
    else:
        # Wrong submission penalty (applied even if model is right but fit is poor)
        penalty = PENALTY_WRONG_SUBMIT

    # Hint penalty (cumulative across session)
    hint_deduction = session.hints_used * PENALTY_HINT

    points_earned = model_bonus + feature_bonus + quality_bonus - penalty
    new_total = max(0, session.score + points_earned - hint_deduction)

    # Clamp so the score can only increase meaningfully after first solve
    if not is_correct:
        new_total = max(0, session.score - penalty)

    session.score = new_total

    message = _compose_message(
        is_correct=is_correct,
        model_correct=model_correct,
        feature_correct=feature_correct,
        fit_quality=fit_quality,
        task=task,
    )

    return {
        "is_correct": is_correct,
        "points_earned": points_earned if is_correct else -penalty,
        "total_score": session.score,
        "model_bonus": model_bonus,
        "feature_bonus": feature_bonus,
        "quality_bonus": quality_bonus,
        "penalty": penalty,
        "hint_deduction": hint_deduction,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Message composer
# ---------------------------------------------------------------------------

def _compose_message(
    is_correct: bool,
    model_correct: bool,
    feature_correct: bool,
    fit_quality: float,
    task: str,
) -> str:
    metric_name = "R²" if task == "regression" else "accuracy"
    metric_val = round(fit_quality, 3)

    if is_correct:
        return (
            f"🎉 Excellent! You found the hidden relationship! "
            f"{metric_name} = {metric_val}. "
            "Check the explanation to understand why this works."
        )

    parts = []
    if not model_correct:
        parts.append(
            "The model type doesn't seem right. "
            "Think about whether the relationship looks like a straight line "
            "or has segments / kinks."
        )
    if not feature_correct:
        q = metric_val
        if task == "regression":
            if q < 0:
                parts.append(
                    f"R² = {q} — this feature is doing worse than predicting the average. "
                    "Try a completely different transformation."
                )
            elif q < 0.40:
                parts.append(
                    f"R² = {q} — very weak fit. "
                    "This feature doesn't capture the pattern well."
                )
            elif q < 0.70:
                parts.append(
                    f"R² = {q} — some structure is explained, but there is still a lot missing."
                )
            else:
                parts.append(
                    f"R² = {q} — you are getting closer! "
                    "But the fit is not quite tight enough yet."
                )
        else:
            parts.append(
                f"Accuracy = {q} — not quite there. "
                "Try different features or adjust the model."
            )

    return " ".join(parts) if parts else "Not quite. Keep experimenting!"
