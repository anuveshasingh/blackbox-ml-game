"""
test_scoring.py — Tests for the scoring engine.
"""

import pytest
from blackbox_game.scoring import (
    PlayerSession,
    compute_score,
    CORRECT_R2_THRESHOLD,
    PENALTY_HINT,
    PENALTY_WRONG_SUBMIT,
    SCORE_CORRECT_MODEL,
    SCORE_CORRECT_FEATURE,
    SCORE_QUALITY_BONUS,
)
from blackbox_game import evaluate_submission


class TestPlayerSession:

    def test_initial_state(self):
        s = PlayerSession(puzzle_id="test_01")
        assert s.attempts == 0
        assert s.hints_used == 0
        assert s.solved is False
        assert s.score == 0
        assert s.end_time is None

    def test_use_hint_increments_counter(self):
        s = PlayerSession(puzzle_id="test_01")
        idx = s.use_hint()
        assert idx == 0
        assert s.hints_used == 1
        idx2 = s.use_hint()
        assert idx2 == 1
        assert s.hints_used == 2

    def test_time_taken_none_before_solve(self):
        s = PlayerSession(puzzle_id="test_01")
        assert s.time_taken is None

    def test_to_dict_contains_required_keys(self):
        s = PlayerSession(puzzle_id="test_01")
        d = s.to_dict()
        for key in ["puzzle_id", "attempts", "hints_used", "solved", "score"]:
            assert key in d


class TestComputeScore:

    def _session(self):
        return PlayerSession(puzzle_id="test_01")

    def test_correct_answer_earns_positive_score(self):
        s = self._session()
        result = compute_score(s, model_correct=True, fit_quality=0.99)
        assert result["is_correct"] is True
        assert result["points_earned"] > 0
        assert s.score > 0

    def test_incorrect_answer_earns_penalty(self):
        s = self._session()
        result = compute_score(s, model_correct=True, fit_quality=0.10)
        assert result["is_correct"] is False
        assert result["points_earned"] < 0

    def test_correct_beats_incorrect(self):
        s1 = self._session()
        compute_score(s1, model_correct=True, fit_quality=0.999)
        score_correct = s1.score

        s2 = self._session()
        compute_score(s2, model_correct=True, fit_quality=0.10)
        score_incorrect = s2.score

        assert score_correct > score_incorrect

    def test_hint_reduces_final_score(self):
        """Using a hint should result in a lower total score after a correct solve."""
        s_no_hint = self._session()
        compute_score(s_no_hint, model_correct=True, fit_quality=0.999)
        score_no_hint = s_no_hint.score

        s_hint = self._session()
        s_hint.use_hint()  # use one hint
        compute_score(s_hint, model_correct=True, fit_quality=0.999)
        score_with_hint = s_hint.score

        assert score_no_hint > score_with_hint

    def test_score_never_negative(self):
        """Score should never drop below zero regardless of penalties."""
        s = self._session()
        # Make lots of wrong attempts
        for _ in range(20):
            compute_score(s, model_correct=False, fit_quality=-5.0)
        assert s.score >= 0

    def test_wrong_model_is_not_correct(self):
        s = self._session()
        # Good fit but wrong model type
        result = compute_score(s, model_correct=False, fit_quality=0.99)
        assert result["is_correct"] is False

    def test_quality_bonus_for_perfect_fit(self):
        s = self._session()
        result = compute_score(s, model_correct=True, fit_quality=1.0)
        assert result["quality_bonus"] == SCORE_QUALITY_BONUS

    def test_no_quality_bonus_at_threshold(self):
        s = self._session()
        result = compute_score(
            s, model_correct=True, fit_quality=CORRECT_R2_THRESHOLD
        )
        assert result["quality_bonus"] == 0

    def test_attempts_increment(self):
        s = self._session()
        compute_score(s, model_correct=True, fit_quality=0.10)
        compute_score(s, model_correct=True, fit_quality=0.10)
        assert s.attempts == 2

    def test_session_marked_solved_on_correct(self):
        s = self._session()
        compute_score(s, model_correct=True, fit_quality=0.999)
        assert s.solved is True
        assert s.end_time is not None

    def test_message_is_string(self):
        s = self._session()
        result = compute_score(s, model_correct=True, fit_quality=0.999)
        assert isinstance(result["message"], str)
        assert len(result["message"]) > 0


class TestEvaluateSubmissionAPI:
    """Integration tests for the high-level evaluate_submission function."""

    def test_correct_submission_returns_true(self):
        result = evaluate_submission(
            puzzle_id="square_01",
            model="linear_regression",
            features=["square:x"],
        )
        assert result["is_correct"] is True
        assert result["r2"] > 0.99

    def test_wrong_transform_returns_false(self):
        result = evaluate_submission(
            puzzle_id="square_01",
            model="linear_regression",
            features=["identity:x"],
        )
        assert result["is_correct"] is False

    def test_wrong_model_returns_false(self):
        result = evaluate_submission(
            puzzle_id="line_01",
            model="decision_tree",
            features=["identity:x"],
        )
        assert result["is_correct"] is False

    def test_unknown_puzzle_raises(self):
        with pytest.raises(KeyError):
            evaluate_submission(
                puzzle_id="does_not_exist",
                model="linear_regression",
                features=["identity:x"],
            )

    def test_unknown_model_returns_error(self):
        result = evaluate_submission(
            puzzle_id="line_01",
            model="xgboost",
            features=["identity:x"],
        )
        assert "error" in result

    def test_explanation_included_on_success(self):
        result = evaluate_submission(
            puzzle_id="line_01",
            model="linear_regression",
            features=["identity:x"],
        )
        assert result["is_correct"] is True
        assert "explanation" in result

    def test_explanation_not_leaked_on_failure(self):
        result = evaluate_submission(
            puzzle_id="square_01",
            model="linear_regression",
            features=["identity:x"],   # wrong transform
        )
        assert result["is_correct"] is False
        assert "explanation" not in result

    def test_piecewise_puzzle_tree_correct(self):
        result = evaluate_submission(
            puzzle_id="piecewise_01",
            model="decision_tree",
            features=["identity:x"],
        )
        assert result["is_correct"] is True

    def test_product_puzzle_with_binary_feature(self):
        result = evaluate_submission(
            puzzle_id="product_01",
            model="linear_regression",
            features=[{"binary": "multiply", "a": "x1", "b": "x2"}],
        )
        assert result["is_correct"] is True

    def test_session_object_returned(self):
        result = evaluate_submission(
            puzzle_id="line_01",
            model="linear_regression",
            features=["identity:x"],
        )
        assert "session" in result
        assert "score" in result["session"]
