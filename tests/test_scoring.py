"""
test_scoring.py — Tests for scoring engine, fuzzy matching, and evaluate_submission.
"""

import pytest
from blackbox_game.scoring import (
    PlayerSession, compute_score, check_fuzzy_match, _parse_power_feature,
    CORRECT_R2_THRESHOLD, PENALTY_WRONG_SUBMIT,
    SCORE_CORRECT_MODEL, SCORE_CORRECT_FEATURE, SCORE_QUALITY_BONUS,
    POWER_MAP,
)
from blackbox_game import evaluate_submission


# ---------------------------------------------------------------------------
# PlayerSession
# ---------------------------------------------------------------------------

class TestPlayerSession:

    def test_initial_state(self):
        s = PlayerSession(puzzle_id="test_01")
        assert s.attempts == 0
        assert s.solved is False
        assert s.score == 0
        assert s.end_time is None

    def test_to_dict_contains_required_keys(self):
        s = PlayerSession(puzzle_id="test_01")
        d = s.to_dict()
        for key in ["puzzle_id", "attempts", "solved", "score"]:
            assert key in d

    def test_time_taken_none_before_solve(self):
        assert PlayerSession(puzzle_id="x").time_taken is None


# ---------------------------------------------------------------------------
# Power family parsing
# ---------------------------------------------------------------------------

class TestParsePowerFeature:

    def test_transform_shorthand(self):
        info = _parse_power_feature("square:x")
        assert info == (2.0, "x")

    def test_identity_shorthand(self):
        info = _parse_power_feature("identity:x")
        assert info == (1.0, "x")

    def test_reciprocal(self):
        info = _parse_power_feature("reciprocal:x")
        assert info == (-1.0, "x")

    def test_plain_column_is_identity(self):
        info = _parse_power_feature("x")
        assert info == (1.0, "x")

    def test_non_power_transform_returns_none(self):
        assert _parse_power_feature("sin:x") is None
        assert _parse_power_feature("log:x") is None
        assert _parse_power_feature("cos:x") is None

    def test_binary_dict_returns_none(self):
        assert _parse_power_feature({"binary": "multiply", "a": "x1", "b": "x2"}) is None


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

class TestFuzzyMatching:

    def test_exact_power_gives_high_fraction(self):
        frac, desc = check_fuzzy_match(["square:x"], ["square:x"])
        assert frac > 0.9

    def test_off_by_one_power(self):
        """square (2) vs cube (3): diff=1, fraction=1/1=1.0 capped at 0.99"""
        frac, desc = check_fuzzy_match(["square:x"], ["cube:x"])
        assert abs(frac - 0.99) < 0.01
        assert "x^" in desc  # description mentions powers

    def test_off_by_two_powers(self):
        """identity (1) vs cube (3): diff=2, fraction=1/2=0.5"""
        frac, desc = check_fuzzy_match(["identity:x"], ["cube:x"])
        assert abs(frac - 0.5) < 0.01

    def test_sqrt_vs_square(self):
        """sqrt (0.5) vs square (2): diff=1.5, fraction=1/1.5≈0.667"""
        frac, desc = check_fuzzy_match(["sqrt:x"], ["square:x"])
        assert abs(frac - (1 / 1.5)) < 0.01

    def test_wrong_column_no_match(self):
        """Different column names → no fuzzy match."""
        frac, _ = check_fuzzy_match(["square:x1"], ["square:x2"])
        assert frac == 0.0

    def test_non_power_transform_no_match(self):
        """sin is not in power family → no fuzzy match."""
        frac, _ = check_fuzzy_match(["sin:x"], ["square:x"])
        assert frac == 0.0

    def test_empty_features_no_match(self):
        frac, _ = check_fuzzy_match([], ["square:x"])
        assert frac == 0.0

    def test_multi_feature_partial(self):
        """Submit [square:x] for solution [square:x, identity:x] → partial (0.5)."""
        frac, _ = check_fuzzy_match(["square:x"], ["square:x", "identity:x"])
        assert frac < 0.99  # partial, not full


# ---------------------------------------------------------------------------
# compute_score
# ---------------------------------------------------------------------------

class TestComputeScore:

    def _session(self):
        return PlayerSession(puzzle_id="test")

    def test_correct_answer_earns_full_score(self):
        s = self._session()
        r = compute_score(s, model_correct=True, fit_quality=0.999)
        assert r["is_correct"] is True
        assert r["model_bonus"] == SCORE_CORRECT_MODEL
        assert r["feature_bonus"] == SCORE_CORRECT_FEATURE
        assert s.score > 0

    def test_wrong_model_incorrect(self):
        s = self._session()
        r = compute_score(s, model_correct=False, fit_quality=0.999)
        assert r["is_correct"] is False

    def test_wrong_answer_penalty(self):
        s = self._session()
        r = compute_score(s, model_correct=True, fit_quality=0.1)
        assert r["is_correct"] is False
        assert r["penalty"] == PENALTY_WRONG_SUBMIT

    def test_score_never_negative(self):
        s = self._session()
        for _ in range(20):
            compute_score(s, model_correct=False, fit_quality=-5.0)
        assert s.score >= 0

    def test_correct_beats_wrong(self):
        s1 = self._session()
        compute_score(s1, model_correct=True, fit_quality=0.999)

        s2 = self._session()
        compute_score(s2, model_correct=True, fit_quality=0.1)

        assert s1.score > s2.score

    def test_fuzzy_gives_partial_credit(self):
        """Fuzzy match → partial credit, no penalty, less than full correct."""
        s_correct = self._session()
        compute_score(s_correct, model_correct=True, fit_quality=0.999)
        full_score = s_correct.score

        s_fuzzy = self._session()
        fuzzy_info = (0.5, "off by 2 powers")  # 50% fraction
        compute_score(s_fuzzy, model_correct=True, fit_quality=0.1,
                      fuzzy_info=fuzzy_info)
        fuzzy_score = s_fuzzy.score

        assert fuzzy_score > 0          # partial credit given
        assert fuzzy_score < full_score  # but less than full

    def test_fuzzy_wrong_model_no_credit(self):
        """Even a good fuzzy match gives nothing if the model is wrong."""
        s = self._session()
        fuzzy_info = (0.99, "almost exact")
        r = compute_score(s, model_correct=False, fit_quality=0.1,
                          fuzzy_info=fuzzy_info)
        assert r["is_fuzzy"] is False
        assert r["feature_bonus"] == 0

    def test_quality_bonus_for_perfect_fit(self):
        s = self._session()
        r = compute_score(s, model_correct=True, fit_quality=1.0)
        assert r["quality_bonus"] == SCORE_QUALITY_BONUS

    def test_no_quality_bonus_at_threshold(self):
        s = self._session()
        r = compute_score(s, model_correct=True, fit_quality=CORRECT_R2_THRESHOLD)
        assert r["quality_bonus"] == 0

    def test_session_solved_flag(self):
        s = self._session()
        compute_score(s, model_correct=True, fit_quality=0.999)
        assert s.solved is True
        assert s.end_time is not None

    def test_attempts_increment(self):
        s = self._session()
        compute_score(s, model_correct=True, fit_quality=0.1)
        compute_score(s, model_correct=True, fit_quality=0.1)
        assert s.attempts == 2

    def test_message_is_string(self):
        s = self._session()
        r = compute_score(s, model_correct=True, fit_quality=0.999)
        assert isinstance(r["message"], str) and len(r["message"]) > 0


# ---------------------------------------------------------------------------
# evaluate_submission integration
# ---------------------------------------------------------------------------

class TestEvaluateSubmissionAPI:

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
            features=["sin:x"],
        )
        assert result["is_correct"] is False

    def test_fuzzy_match_detected(self):
        """cube:x on square_01 (y=x²) → fuzzy match (power off by 1)."""
        result = evaluate_submission(
            puzzle_id="square_01",
            model="linear_regression",
            features=["cube:x"],
        )
        assert result["is_correct"] is False
        assert result["is_fuzzy"] is True
        assert result["score_result"]["feature_bonus"] > 0

    def test_wrong_model_not_correct(self):
        result = evaluate_submission(
            puzzle_id="line_01",
            model="decision_tree",
            features=["identity:x"],
        )
        assert result["is_correct"] is False

    def test_unknown_puzzle_raises(self):
        with pytest.raises(KeyError):
            evaluate_submission(
                puzzle_id="nonexistent",
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

    def test_explanation_on_success_only(self):
        # Correct → explanation included
        ok = evaluate_submission("line_01", "linear_regression", ["identity:x"])
        assert ok["is_correct"] is True
        assert "explanation" in ok

        # Wrong → no explanation
        bad = evaluate_submission("square_01", "linear_regression", ["sin:x"])
        assert bad["is_correct"] is False
        assert "explanation" not in bad

    def test_projectile_y_correct(self):
        result = evaluate_submission(
            puzzle_id="projectile_y",
            model="linear_regression",
            features=[{"product": ["t", "sin:theta"]}, "square:t"],
        )
        assert result["is_correct"] is True

    def test_shm_energy_correct(self):
        result = evaluate_submission(
            puzzle_id="shm_energy",
            model="linear_regression",
            features=["square:x", "square:v"],
        )
        assert result["is_correct"] is True

    def test_coulomb_2_correct(self):
        result = evaluate_submission(
            puzzle_id="coulomb_2",
            model="linear_regression",
            features=["reciprocal:r1", "reciprocal:r2"],
        )
        assert result["is_correct"] is True

    def test_travelling_wave_correct(self):
        result = evaluate_submission(
            puzzle_id="travelling_wave",
            model="linear_regression",
            features=[{"sum": ["x", {"term": "t", "sign": -1}], "transform": "sin"}],
        )
        assert result["is_correct"] is True

    def test_session_in_response(self):
        result = evaluate_submission("line_01", "linear_regression", ["identity:x"])
        assert "session" in result
        assert "score" in result["session"]

    def test_no_hints_field_in_session(self):
        """hints_used removed from session since hints are disabled."""
        result = evaluate_submission("line_01", "linear_regression", ["identity:x"])
        assert "hints_used" not in result["session"]
