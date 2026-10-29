"""Gkeyll surface faces retain their positions and lower-dimensional DG law."""

from pathlib import Path

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython, operations
from postgkyl.gdatastate import materialize_point_values
from postgkyl.io.gkyl_reader import GkylReader
from postgkyl.io.surface import unpack_surface
from generate_test_data import write_gkyl_field

GEN = Path(__file__).parent / "test_data" / "generated"
ROUND_OFF = dict(rtol=2e-13, atol=2e-13)
needs_gkeyll = pytest.mark.skipif(not gpython.available(),
                                  reason="no compiled Gkeyll basis found")


def _filename(ndim, direction):
  return GEN / f"surface_{ndim}d_dir{direction}.gkyl"


@pytest.mark.parametrize("ndim,direction", [(1, 0), (2, 0), (2, 1), (3, 0),
                                            (3, 1), (3, 2)])
@pytest.mark.parametrize("python_reader", [False, True])
def test_surface_load_reconstructs_faces_and_field_blocks(
    ndim, direction, python_reader, monkeypatch):
  filename = _filename(ndim, direction)
  basis_available = gpython.available()
  if python_reader:
    monkeypatch.setattr(gpython, "available", lambda: False)
  data = pg.load(filename)
  assert data.backend == ("numpy"
                          if python_reader or not basis_available else "gkyl")
  raw = GkylReader(str(filename))
  raw.preload()
  raw_grid, raw_values = raw.load()
  block = (1, 2, 4)[ndim - 1]
  expected_shape = list(raw_values.shape[:-1])
  expected_shape[direction] += 1
  assert data.values.shape == (*expected_shape, 2 * block)
  assert data.ctx["value_form"] == f"modal_surf{direction}"
  assert data.ctx["num_comps"] == 2 * block
  assert tuple(data.ctx["cells"]) == tuple(expected_shape)
  np.testing.assert_array_equal(data.grid[direction], raw_grid[direction])
  head = tuple(
      slice(None, -1) if d == direction else slice(None) for d in range(ndim))
  tail = tuple(-1 if d == direction else slice(None) for d in range(ndim))
  np.testing.assert_array_equal(data.values[head], raw_values[..., :2 * block])
  np.testing.assert_array_equal(data.values[tail],
                                raw_values[tail + (slice(2 * block, None), )])
  if basis_available:
    np.testing.assert_array_equal(
        data.select(comp=1).values, data.values[..., block:2 * block])


@needs_gkeyll
@pytest.mark.parametrize("ndim,direction", [(1, 0), (2, 0), (2, 1), (3, 0),
                                            (3, 1), (3, 2)])
def test_surface_interpolation_matches_affine_field(ndim, direction):
  data = pg.load(_filename(ndim, direction))
  result = data.interpolate(num_interp=3)
  assert result.backend == "numpy"
  assert result.is_interpolated
  assert result.values.shape[direction] == len(data.grid[direction])
  np.testing.assert_array_equal(result.grid[direction], data.grid[direction])
  coordinates = []
  for d, grid in enumerate(result.grid):
    coordinates.append(grid if d == direction else 0.5 * (grid[:-1] + grid[1:]))
  mesh = np.meshgrid(*coordinates, indexing="ij")
  tangents = [d for d in range(ndim) if d != direction]
  expected = 5.0 + 2.0 * mesh[direction]
  for slope, d in zip((3.0, -4.0), tangents):
    expected = expected + slope * mesh[d]
  np.testing.assert_allclose(result.values[..., 0], expected, **ROUND_OFF)
  np.testing.assert_allclose(result.values[..., 1], 7.0 - expected, **ROUND_OFF)


@needs_gkeyll
@pytest.mark.parametrize("ndim,direction", [(1, 0), (2, 0), (2, 1), (3, 0),
                                            (3, 1), (3, 2)])
def test_surface_local_poly_evaluates_tangential_polynomial(ndim, direction):
  data = pg.load(_filename(ndim, direction))
  result = data.local_poly(npoints=3)
  assert result.backend == "numpy"
  assert result.is_interpolated
  np.testing.assert_array_equal(result.grid[direction], data.grid[direction])
  assert result.values.shape[direction] == data.values.shape[direction]
  tangents = [d for d in range(ndim) if d != direction]
  for d in tangents:
    assert result.values.shape[
        d] == 3 * data.values.shape[d] + data.values.shape[d] - 1
  mesh = np.meshgrid(*result.grid, indexing="ij")
  expected = 5.0 + 2.0 * mesh[direction]
  for slope, d in zip((3.0, -4.0), tangents):
    expected = expected + slope * mesh[d]
  finite = np.isfinite(result.values[..., 0])
  assert np.any(finite)
  np.testing.assert_allclose(result.values[..., 0][finite], expected[finite],
                             **ROUND_OFF)
  np.testing.assert_allclose(result.values[..., 1][finite],
                             (7.0 - expected)[finite], **ROUND_OFF)
  if tangents:
    assert np.any(~finite)


@needs_gkeyll
def test_selected_surface_local_poly_can_plot():
  data = pg.load(_filename(2, 1)).select(z1=0)
  result = data.local_poly(npoints=3)
  assert result.values.shape[1] == 1
  np.testing.assert_array_equal(result.grid[1], data.grid[1])
  result.plot(no_show=True)


def test_1d_surface_plots_physical_values_at_faces():
  data = pg.load(_filename(1, 0))
  points = materialize_point_values(data)
  expected = 5.0 + 2.0 * data.grid[0]
  assert points.is_interpolated
  np.testing.assert_array_equal(points.grid[0], data.grid[0])
  np.testing.assert_allclose(data.values[:, 0], expected, **ROUND_OFF)
  np.testing.assert_allclose(points.values[:, 0], expected, **ROUND_OFF)
  np.testing.assert_allclose(points.values[:, 1], 7.0 - expected, **ROUND_OFF)
  np.testing.assert_allclose(data.interpolate().values, data.values,
                             **ROUND_OFF)
  np.testing.assert_allclose(data.local_poly().values, data.values, **ROUND_OFF)
  np.testing.assert_allclose((data + 2.0).interpolate().values,
                             data.values + 2.0, **ROUND_OFF)
  figure = data.select(comp=0).plot(no_show=True)
  line = figure.axes[0].lines[0]
  np.testing.assert_array_equal(line.get_xdata(), data.grid[0])
  np.testing.assert_allclose(line.get_ydata(), expected, **ROUND_OFF)
  assert data.ctx["value_form"] == "modal_surf0"


def test_1d_surface_scalar_arithmetic_preserves_face_values():
  data = pg.load(_filename(1, 0))
  values = data.values.copy()
  np.testing.assert_allclose((data / 2.0).interpolate().values, values / 2.0,
                             **ROUND_OFF)
  np.testing.assert_allclose((data - 2.0).interpolate().values, values - 2.0,
                             **ROUND_OFF)
  np.testing.assert_allclose((2.0 - data).interpolate().values, 2.0 - values,
                             **ROUND_OFF)
  np.testing.assert_allclose((data - data).interpolate().values,
                             np.zeros_like(values), **ROUND_OFF)
  np.testing.assert_array_equal(data.values, values)
  with pytest.raises(ValueError, match="scalar or another surface field"):
    data + np.ones_like(values)
  with pytest.raises(NotImplementedError, match="products and quotients"):
    data / data
  with pytest.raises(NotImplementedError, match="surface nonlinear"):
    2.0 / data


@needs_gkeyll
def test_surface_arithmetic_and_point_operations():
  data = pg.load(_filename(3, 1))
  shifted = (7.0 - 2.0 * data).select(comp=0).interpolate(num_interp=2)
  base = data.select(comp=0).interpolate(num_interp=2)
  np.testing.assert_allclose(shifted.values, 7.0 - 2.0 * base.values,
                             **ROUND_OFF)
  np.testing.assert_allclose((data + data).interpolate(num_interp=2).values,
                             2.0 * data.interpolate(num_interp=2).values,
                             **ROUND_OFF)
  np.testing.assert_allclose(
      operations.evaluate("f 2 *", data).interpolate(num_interp=2).values,
      2.0 * data.interpolate(num_interp=2).values, **ROUND_OFF)
  point = data.interpolate(num_interp=2)
  np.testing.assert_allclose((point * point).values, point.values**2,
                             **ROUND_OFF)
  np.testing.assert_allclose(
      operations.evaluate("f sq", point).values, point.values**2, **ROUND_OFF)
  assert point.select(z1=0).values.shape[1] == 1
  integral = point.select(comp=0).integrate()
  centers = [0.5 * (axis[0] + axis[-1]) for axis in point.grid]
  volume = np.prod([axis[-1] - axis[0] for axis in point.grid])
  expected = (5.0 + 2.0 * centers[1] + 3.0 * centers[0] -
              4.0 * centers[2]) * volume
  np.testing.assert_allclose(integral, expected, **ROUND_OFF)
  point.select(z2=0).select(comp=0).plot(no_show=True)


@needs_gkeyll
def test_surface_rejects_unsupported_coefficient_operations():
  data = pg.load(_filename(3, 2))
  with pytest.raises(NotImplementedError, match="surface coefficient"):
    data * data
  with pytest.raises(NotImplementedError, match="surface nonlinear"):
    data**2
  with pytest.raises(ValueError, match="surface nonlinear"):
    operations.evaluate("f sq", data)
  with pytest.raises(NotImplementedError, match="surface coefficients"):
    data.represent(to="nodal")


def test_surface_rejects_partial_read_and_bad_direction(tmp_path):
  filename = _filename(3, 2)
  with pytest.raises(ValueError, match="full read"):
    pg.load(filename, z2="1:3")
  with pytest.raises(ValueError, match="full read"):
    pg.load(filename, component="0:4")
  invalid = tmp_path / "bad_surface.gkyl"
  write_gkyl_field(invalid, [2], [0.], [1.],
                   np.zeros((2, 2)),
                   1,
                   "serendipity",
                   metadata={"value_form": "modal_surf1"})
  with pytest.raises(ValueError, match="out of range"):
    pg.load(invalid)


@pytest.mark.parametrize("value_form",
                         ["modal_surf", "modal_surf-1", "modal_surf1x"])
def test_surface_rejects_malformed_direction(tmp_path, value_form):
  invalid = tmp_path / "bad_surface.gkyl"
  write_gkyl_field(invalid, [2], [0.], [1.],
                   np.zeros((2, 2)),
                   1,
                   "serendipity",
                   metadata={"value_form": value_form})
  with pytest.raises(ValueError, match="invalid surface value_form"):
    pg.load(invalid)


@pytest.mark.parametrize("ncomp", [1, 3])
def test_surface_rejects_invalid_component_count(tmp_path, ncomp):
  invalid = tmp_path / "bad_surface.gkyl"
  write_gkyl_field(invalid, [2], [0.], [1.],
                   np.zeros((2, ncomp)),
                   1,
                   "serendipity",
                   metadata={"value_form": "modal_surf0"})
  with pytest.raises(ValueError, match="two equal halves"):
    pg.load(invalid)


def test_surface_rejects_payload_shape_inconsistent_with_cells():
  grid = [np.array([0., 0.5, 1.])]
  values = np.zeros((3, 2))
  ctx = {"value_form": "modal_surf0", "cells": [2]}
  with pytest.raises(ValueError, match="stored cell counts"):
    unpack_surface(grid, values, ctx)
