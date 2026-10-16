"""Tests for postgkyl.numerics.diff_metrics -- the pure max-abs/max-rel
element-wise difference metric shared by postgkyl.diagnostics.compare.
"""

from __future__ import annotations

import numpy as np
import pytest

from postgkyl.numerics import max_abs_rel_diff


class TestMaxAbsRelDiff:

  def test_identical_arrays_are_zero(self):
    values = np.array([1.0, -2.0, 3.5])
    assert max_abs_rel_diff(values, values) == (0.0, 0.0)

  def test_all_zero_arrays_are_zero_not_nan(self):
    zeros = np.zeros(4)
    assert max_abs_rel_diff(zeros, zeros) == (0.0, 0.0)

  def test_scalar_difference(self):
    values0 = np.array([2.0])
    values = np.array([3.0])
    max_abs, max_rel = max_abs_rel_diff(values0, values)
    assert max_abs == pytest.approx(1.0)
    assert max_rel == pytest.approx(0.5)

  def test_abs_and_rel_maxima_can_come_from_different_elements(self):
    values0 = np.array([1.0, 10.0, 100.0])
    values = np.array([1.1, 10.0, 100.5])
    max_abs, max_rel = max_abs_rel_diff(values0, values)
    assert max_abs == pytest.approx(0.5)  # |100.5-100|, the largest abs diff
    assert max_rel == pytest.approx(0.1)  # |1.1-1|/1, the largest rel diff

  def test_zero_reference_element_gives_infinite_relative_diff(self):
    values0 = np.array([0.0, 1.0])
    values = np.array([1.0, 1.0])
    max_abs, max_rel = max_abs_rel_diff(values0, values)
    assert max_abs == pytest.approx(1.0)
    assert max_rel == np.inf

  def test_multidimensional_arrays(self):
    values0 = np.full((3, 4), 2.0)
    values = values0.copy()
    values[1, 2] = 2.5
    max_abs, max_rel = max_abs_rel_diff(values0, values)
    assert max_abs == pytest.approx(0.5)
    assert max_rel == pytest.approx(0.25)

  def test_shape_mismatch_raises(self):
    with pytest.raises(ValueError, match="shape mismatch"):
      max_abs_rel_diff(np.zeros(3), np.zeros(4))

  def test_empty_arrays_are_zero(self):
    empty = np.array([])
    assert max_abs_rel_diff(empty, empty) == (0.0, 0.0)
