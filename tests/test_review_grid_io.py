"""Analytic regressions for integration and coordinate-consuming workflows."""

from pathlib import Path

import msgpack
import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython, io
from postgkyl.io.metadata import unpack_gkyl_metadata
from postgkyl.io.writer import _build_meta

GEN = Path(__file__).parent / "test_data" / "generated"
needs_gkeyll = pytest.mark.skipif(not gpython.available(),
                                  reason="compiled Gkeyll unavailable")


def _points():
  with np.load(GEN / "point_polynomials.npz") as fixture:
    return pg.GData().push([fixture["x"], fixture["y"]], fixture["values"])


@needs_gkeyll
@pytest.mark.parametrize("form", ["modal", "nodal", "quad"])
@pytest.mark.parametrize("backend", ["gkyl", "numpy"])
@pytest.mark.parametrize("stem, expected", [
    ("piecewise_linear_1d", [12., 4.]),
    ("polynomial_1d_ms_p1", [7.5, 11.625]),
])
def test_dg_integral_preserves_domain_fields_and_discontinuous_traces(
    form, backend, stem, expected):
  # The piecewise field jumps from 3 to 6 at x=1. Integrating its two
  # polynomial cells gives 2+10=12; the other field integrates 1 over [-1,3].
  data = pg.load(str(GEN / f"{stem}.gkyl")).represent(to=form)
  if backend == "numpy":
    data = data._result(data.grid, data.values.copy())
  before = data.values.copy()
  np.testing.assert_allclose(data.integrate(), expected, rtol=0, atol=2e-14)
  np.testing.assert_array_equal(data.values, before)
  assert data.ctx["value_form"] == form


@needs_gkeyll
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("inplace", [False, True])
def test_partial_nodal_integral_retains_exact_polynomial(axis, inplace):
  # f=(2+x)(2+2y), g=(4-x/4)(4-y/2), on [-1,2] x [1/4,9/4].
  data = pg.load(str(GEN / "polynomial_2d_ms_p1.gkyl")).represent(to="nodal")
  result = data.integrate(axis=axis, inplace=inplace)
  assert (result is data) == inplace
  assert result.ctx["value_form"] == "modal"
  points = result.interpolate()
  coord = (points.grid[0][:-1] + points.grid[0][1:]) / 2
  expected = (np.stack([7.5 * (2 + 2 * coord), 11.625 * (4 - coord / 2)],
                       axis=-1) if axis == 0 else np.stack(
                           [9 * (2 + coord), 6.75 * (4 - coord / 4)], axis=-1))
  np.testing.assert_allclose(points.values, expected, rtol=0, atol=8e-14)


@pytest.mark.parametrize("coordinates",
                         [[0., 1.], [0., 1., 2.], [-3., -2.8, 4.]])
def test_point_constant_integral_is_the_supplied_domain_length(coordinates):
  x = np.asarray(coordinates)
  data = pg.GData().push([x], np.stack([np.ones_like(x), 3 * x + 2], axis=-1))
  exact = [x[-1] - x[0], 1.5 * (x[-1]**2 - x[0]**2) + 2 * (x[-1] - x[0])]
  np.testing.assert_allclose(data.integrate(), exact, rtol=0, atol=2e-14)


def test_nonuniform_point_integral_has_exact_trapezoidal_error():
  data = _points()
  x, y = data.grid
  # For x^2, trapezoid error is +sum(h^3)/6. Apply the product-domain
  # lengths to x^2+2*y^2, whose exact integral is 28.75.
  error = np.sum(np.diff(x)**3) / 3 + np.sum(np.diff(y)**3)
  np.testing.assert_allclose(data.integrate(), [9.75, 28.75 + error],
                             rtol=0,
                             atol=3e-14)
  partial = data.integrate(axis=0)
  expected = np.stack(
      [10.5 - 4.5 * y, 3 + np.sum(np.diff(x)**3) / 6 + 6 * y**2], axis=-1)
  np.testing.assert_allclose(partial.values, expected, rtol=0, atol=3e-14)


def test_nonuniform_point_gradient_resolves_all_axes_and_components():
  data = _points()
  x, y = np.meshgrid(*data.grid, indexing="ij")
  expected = np.stack([3 + 5 * y, 2 * x, -4 + 5 * x, 4 * y], axis=-1)
  np.testing.assert_allclose(data.differentiate().values,
                             expected,
                             rtol=0,
                             atol=5e-13)


@needs_gkeyll
@pytest.mark.parametrize("form", ["nodal", "quad"])
@pytest.mark.parametrize("backend", ["gkyl", "numpy"])
def test_gradient_refuses_a_stencil_across_packed_dg_nodes(form, backend):
  data = pg.load(str(GEN / "piecewise_linear_1d.gkyl")).represent(to=form)
  if backend == "numpy":
    data = data._result(data.grid, data.values.copy())
  with pytest.raises(ValueError, match="modal coefficients"):
    data.differentiate()


@pytest.mark.parametrize("layout", ["points", "edges"])
def test_text_export_preserves_coordinate_locations_and_complete_values(
    tmp_path, layout):
  if layout == "points":
    data = _points()
    coords = data.grid
  else:
    with np.load(GEN / "nonuniform_polynomials.npz") as fixture:
      data = pg.GData().push([fixture["x"], fixture["y"]], fixture["values"])
    coords = [(edges[:-1] + edges[1:]) / 2 for edges in data.grid]
  path = data.save(str(tmp_path / "samples"), extension="txt")
  written = np.loadtxt(path, delimiter=",")
  x, y = np.meshgrid(*coords, indexing="ij")
  expected = np.column_stack([x.ravel(), y.ravel(), data.values.reshape(-1, 2)])
  np.testing.assert_allclose(written, expected, rtol=2e-15, atol=2e-15)


def test_invalid_text_grid_fails_before_replacing_output(tmp_path):
  data = pg.GData().push([np.arange(5.)], np.ones((3, 1)))
  path = tmp_path / "existing.txt"
  path.write_text("previous result\n")
  with pytest.raises(ValueError, match="coordinate count"):
    data.save(str(path), extension="txt")
  assert path.read_text() == "previous result\n"


@pytest.mark.parametrize("layout", ["nonuniform", "points", "mapped"])
@pytest.mark.parametrize("existing", [False, True])
def test_gkyl_rejects_unrepresentable_geometry_without_touching_output(
    tmp_path, layout, existing):
  if layout == "nonuniform":
    with np.load(GEN / "nonuniform_polynomials.npz") as fixture:
      data = pg.GData().push([fixture["x"], fixture["y"]], fixture["values"])
    np.testing.assert_allclose(data.integrate(), [9.75, 9.], rtol=0, atol=3e-14)
  elif layout == "points":
    data = _points()
  else:
    x, y = np.meshgrid(np.linspace(0., 1., 4),
                       np.linspace(0., 2., 5),
                       indexing="ij")
    data = pg.GData(ctx={"grid_type": "mapped", "mapped_axes": {0: 0, 1: 0}})
    data.push([x + y / 3, y], np.ones((3, 4, 1)))
  path = tmp_path / "geometry.gkyl"
  if existing:
    path.write_bytes(b"previous result")
  with pytest.raises(ValueError, match="cannot preserve"):
    data.save(str(path))
  assert path.exists() == existing
  if existing:
    assert path.read_bytes() == b"previous result"


@needs_gkeyll
def test_real_mapping_cannot_be_saved_with_missing_geometry(tmp_path):
  data = pg.load(str(GEN / "2d_ms_p1.gkyl")).interpolate()
  data = data.map(str(GEN / "2d_c2p_stretch_ms_p1.gkyl"))
  path = tmp_path / "mapped.gkyl"
  with pytest.raises(ValueError, match="cannot preserve"):
    data.save(str(path))
  assert not path.exists()


@needs_gkeyll
@pytest.mark.parametrize("reader", [io.GkylReader, io.GkylCReader])
def test_supported_roundtrip_preserves_coordinates_and_analytic_calculus(
    tmp_path, monkeypatch, reader):
  data = pg.load(str(GEN / "polynomial_1d_ms_p1.gkyl"))
  path = data.save(str(tmp_path / "polynomial.gkyl"))
  monkeypatch.setattr(io, "_READERS", {"test": reader})
  back = pg.load(path)
  np.testing.assert_array_equal(back.grid[0], data.grid[0])
  np.testing.assert_allclose(back.integrate(), [7.5, 11.625],
                             rtol=0,
                             atol=2e-14)
  # The fallback reader yields coefficients too. Interpolation makes their
  # physical field available to the same finite-difference derivative.
  derivative = back.interpolate().differentiate()
  expected = np.broadcast_to([1., -0.25], derivative.values.shape)
  np.testing.assert_allclose(derivative.values, expected, rtol=0, atol=4e-14)


def test_axis_identity_schema_uses_strict_msgpack_keys_and_reads_legacy():
  axes = {0: 0, 1: 0, 3: 3}
  packed = msgpack.packb(_build_meta({"mapped_axes": axes}), use_bin_type=True)
  assert msgpack.unpackb(packed)["mapped_axes"] == {"0": 0, "1": 0, "3": 3}
  assert unpack_gkyl_metadata(packed)["mapped_axes"] == axes
  legacy = msgpack.packb({"mapped_axes": axes}, use_bin_type=True)
  assert unpack_gkyl_metadata(legacy)["mapped_axes"] == axes
