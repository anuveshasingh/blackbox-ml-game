"""
test_transforms.py — Tests for the transformation library.
"""

import numpy as np
import pytest

from blackbox_game.transforms import (
    apply_transform,
    apply_binary_transform,
    list_transforms,
    list_binary_transforms,
    TRANSFORM_REGISTRY,
    BINARY_TRANSFORM_REGISTRY,
)


class TestUnaryTransforms:
    x_pos = np.array([1.0, 2.0, 4.0, 9.0, 16.0])
    x_mixed = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    x_any = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_identity(self):
        result = apply_transform("identity", self.x_any)
        np.testing.assert_array_almost_equal(result, self.x_any)

    def test_square(self):
        result = apply_transform("square", self.x_any)
        expected = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_square_negative(self):
        """Squaring negative numbers should still work (symmetric)."""
        result = apply_transform("square", self.x_mixed)
        assert np.all(result >= 0)

    def test_cube(self):
        result = apply_transform("cube", np.array([2.0, -2.0]))
        np.testing.assert_array_almost_equal(result, [8.0, -8.0])

    def test_sqrt_positive(self):
        result = apply_transform("sqrt", self.x_pos)
        expected = np.sqrt(self.x_pos)
        np.testing.assert_array_almost_equal(result, expected)

    def test_sqrt_negative_returns_nan(self):
        """sqrt of negative value should return NaN, not raise."""
        result = apply_transform("sqrt", np.array([-1.0, 4.0]))
        assert np.isnan(result[0])
        assert np.isfinite(result[1])

    def test_abs(self):
        result = apply_transform("abs", self.x_mixed)
        assert np.all(result >= 0)

    def test_log_positive(self):
        result = apply_transform("log", self.x_pos)
        expected = np.log(self.x_pos)
        np.testing.assert_array_almost_equal(result, expected)

    def test_log_zero_or_negative_returns_nan(self):
        result = apply_transform("log", np.array([-1.0, 0.0, 1.0]))
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isfinite(result[2])

    def test_log2(self):
        result = apply_transform("log2", np.array([1.0, 2.0, 4.0]))
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 2.0])

    def test_reciprocal(self):
        result = apply_transform("reciprocal", np.array([1.0, 2.0, 4.0]))
        np.testing.assert_array_almost_equal(result, [1.0, 0.5, 0.25])

    def test_reciprocal_near_zero_returns_nan(self):
        result = apply_transform("reciprocal", np.array([0.0, 1.0]))
        assert np.isnan(result[0])
        assert np.isfinite(result[1])

    def test_sin(self):
        x = np.array([0.0, np.pi / 2, np.pi])
        result = apply_transform("sin", x)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 0.0], decimal=5)

    def test_cos(self):
        x = np.array([0.0, np.pi / 2, np.pi])
        result = apply_transform("cos", x)
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, -1.0], decimal=5)

    def test_sin_period7(self):
        """sin(2π·x/7) should have period 7."""
        x = np.array([0.0, 7.0, 14.0])
        result = apply_transform("sin_period7", x)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0], decimal=5)

    def test_floor10(self):
        result = apply_transform("floor10", np.array([5.0, 15.0, 25.0]))
        np.testing.assert_array_almost_equal(result, [0.0, 10.0, 20.0])

    def test_unknown_transform_raises(self):
        with pytest.raises(ValueError, match="Unknown transform"):
            apply_transform("nonsense_transform", np.array([1.0]))

    def test_all_registered_transforms_callable(self):
        """Every registered transform should run without crashing on valid input."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        for key in TRANSFORM_REGISTRY:
            result = apply_transform(key, x)
            assert isinstance(result, np.ndarray), f"Transform '{key}' did not return ndarray"

    def test_list_transforms_returns_list_of_dicts(self):
        transforms = list_transforms()
        assert isinstance(transforms, list)
        assert len(transforms) > 0
        for t in transforms:
            assert "key" in t
            assert "description" in t


class TestBinaryTransforms:
    a = np.array([2.0, 4.0, 6.0])
    b = np.array([1.0, 2.0, 3.0])

    def test_multiply(self):
        result = apply_binary_transform("multiply", self.a, self.b)
        np.testing.assert_array_almost_equal(result, [2.0, 8.0, 18.0])

    def test_divide(self):
        result = apply_binary_transform("divide", self.a, self.b)
        np.testing.assert_array_almost_equal(result, [2.0, 2.0, 2.0])

    def test_divide_by_zero_returns_nan(self):
        result = apply_binary_transform("divide", np.array([1.0]), np.array([0.0]))
        assert np.isnan(result[0])

    def test_add(self):
        result = apply_binary_transform("add", self.a, self.b)
        np.testing.assert_array_almost_equal(result, [3.0, 6.0, 9.0])

    def test_subtract(self):
        result = apply_binary_transform("subtract", self.a, self.b)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_distance(self):
        """sqrt(3² + 4²) = 5"""
        result = apply_binary_transform(
            "distance", np.array([3.0]), np.array([4.0])
        )
        np.testing.assert_array_almost_equal(result, [5.0])

    def test_unknown_binary_transform_raises(self):
        with pytest.raises(ValueError, match="Unknown binary transform"):
            apply_binary_transform("nonsense", self.a, self.b)

    def test_list_binary_transforms(self):
        transforms = list_binary_transforms()
        assert isinstance(transforms, list)
        assert len(transforms) > 0
