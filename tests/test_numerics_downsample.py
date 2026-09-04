"""Tests for postgkyl.numerics.downsample."""

from __future__ import annotations

import numpy as np

from postgkyl.numerics.downsample import downsample


class TestDownsample:

  def test_no_arrays_returns_empty_tuple(self):
    assert downsample() == ()

  def test_zero_max_points_returns_unchanged(self):
    x = np.linspace(0, 1, 100)
    out, = downsample(x, maximum_points_per_axis=0)
    np.testing.assert_array_equal(out, x)

  def test_negative_max_points_returns_unchanged(self):
    x = np.linspace(0, 1, 100)
    out, = downsample(x, maximum_points_per_axis=-5)
    np.testing.assert_array_equal(out, x)

  def test_none_max_points_returns_unchanged(self):
    x = np.linspace(0, 1, 100)
    out, = downsample(x, maximum_points_per_axis=None)
    np.testing.assert_array_equal(out, x)

  def test_scalar_array_returns_unchanged(self):
    x = np.array(5.0)
    out, = downsample(x, maximum_points_per_axis=10)
    np.testing.assert_array_equal(out, x)

  def test_mismatched_shapes_returns_unchanged(self):
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 50)
    out_x, out_y = downsample(x, y, maximum_points_per_axis=10)
    assert out_x.shape == (100, )
    assert out_y.shape == (50, )

  def test_already_within_limit_returns_unchanged(self):
    x = np.linspace(0, 1, 5)
    out, = downsample(x, maximum_points_per_axis=20)
    np.testing.assert_array_equal(out, x)

  def test_1d_downsampling_caps_axis_length(self):
    x = np.linspace(0, 10, 100)
    out, = downsample(x, maximum_points_per_axis=20)
    assert out.shape[0] <= 21

  def test_1d_downsampling_keeps_endpoints(self):
    x = np.linspace(0, 10, 100)
    out, = downsample(x, maximum_points_per_axis=20)
    assert out[0] == x[0]
    assert out[-1] == x[-1]

  def test_multiple_arrays_downsampled_consistently(self):
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    x_ds, y_ds = downsample(x, y, maximum_points_per_axis=10)
    assert x_ds.shape == y_ds.shape
    np.testing.assert_allclose(y_ds, np.sin(x_ds))

  def test_2d_downsampling(self):
    value = np.random.default_rng(0).random((100, 100))
    out, = downsample(value, maximum_points_per_axis=10)
    assert out.shape[0] <= 11
    assert out.shape[1] <= 11

  def test_3d_downsampling(self):
    value = np.random.default_rng(0).random((30, 30, 30))
    out, = downsample(value, maximum_points_per_axis=10)
    assert all(s <= 11 for s in out.shape)
