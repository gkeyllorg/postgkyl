"""Independent polynomial checks for storage, field blocks and basis identity."""

from pathlib import Path

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import dg, gpython
from postgkyl.gdatastate import materialize_point_values
from postgkyl.gdatastate.layout import dg_layout

GEN = Path(__file__).parent / "test_data" / "generated"
needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


def test_numpy_modal_point_consumers_reject_coefficients():
  """Partial loading changes storage, never the mathematical meaning."""
  data = pg.load(GEN / "representation_affine_1d.gkyl", z0="0:1")
  assert data.backend == "numpy" and data.ctx["value_form"] == "modal"
  for consume in (np.asarray, materialize_point_values, lambda d: d.magsq(),
                  lambda d: d.fft(), lambda d: d.mask(lower=0),
                  lambda d: d.fit("linear"), lambda d: pg.grid(d)):
    with pytest.raises(ValueError, match="modal|coefficients"):
      consume(data)


@needs_gkeyll
@pytest.mark.parametrize("form", ["nodal", "quad"])
@pytest.mark.parametrize("backend", ["numpy", "gkyl"])
def test_physical_selection_and_magsq_on_packed_affine_fields(form, backend):
  """f=1+2*x, g=3-x: field selection and |(f,g)|² at actual nodes."""
  source = pg.load(GEN / "representation_affine_1d.gkyl")
  represented = source.represent(to=form)
  if backend == "numpy":
    represented.values = np.array(represented.values)
  selected = represented.select(comp=1)
  assert dg_layout(selected).num_fields == 1
  actual = materialize_point_values(selected)
  np.testing.assert_allclose(actual.values[:, 0],
                             3 - actual.grid[0],
                             rtol=2e-14,
                             atol=2e-14)
  if backend == "gkyl":
    modal = selected.represent(to="modal")
    np.testing.assert_allclose(modal.values,
                               source.select(comp=1).values,
                               rtol=2e-14,
                               atol=2e-14)
  squared = represented.magsq(coords="0:2")
  x = squared.grid[0]
  np.testing.assert_allclose(squared.values[:, 0], (1 + 2 * x)**2 + (3 - x)**2,
                             rtol=2e-14,
                             atol=2e-14)
  assert squared.values.shape[-1] == 1
  assert materialize_point_values(squared) is squared


@needs_gkeyll
@pytest.mark.parametrize("verb", ["interpolate", "local_poly"])
@pytest.mark.parametrize("backend", ["numpy", "gkyl"])
@pytest.mark.parametrize("interface", [0., -0.5])
def test_native_and_numpy_nodal_interpolation_exact_affine(
    verb, backend, interface):
  source = pg.load(GEN / "representation_affine_1d.gkyl").represent(to="nodal")
  # Moving the common face changes physical cell widths while preserving
  # each reference-cell polynomial and its values at local evaluation points.
  source.grid[0][1] = interface
  if backend == "numpy":
    source.values = np.array(source.values)
  result = getattr(source, verb)()
  if verb == "interpolate":
    np.testing.assert_array_equal(
        result.grid[0],
        [-1., (interface - 1) / 2, interface, (interface + 1) / 2, 1.])
    x = np.array([-0.75, -0.25, 0.25, 0.75])
  else:
    # Two cell endpoints each, plus one separator at their common face.
    x = np.array([-1., 0., 0., 0., 1.])
    np.testing.assert_array_equal(result.grid[0],
                                  [-1., interface, interface, interface, 1.])
  expected = np.column_stack([1 + 2 * x, 3 - x])
  if verb == "local_poly":
    expected[2] = np.nan
  np.testing.assert_allclose(result.values, expected, rtol=2e-14, atol=2e-14)
  for again in ("interpolate", "local_poly"):
    with pytest.raises(ValueError, match="point values|repeated"):
      getattr(result, again)()


@needs_gkeyll
@pytest.mark.parametrize("verb", [dg.interpolate, dg.local_poly])
@pytest.mark.parametrize("ncomp", [0, 1, 3])
def test_dg_boundary_rejects_incomplete_fields(verb, ncomp):
  with pytest.raises(ValueError, match="complete field blocks"):
    verb(np.zeros((1, ncomp)), [np.array([0., 1.])],
         poly_order=1,
         basis_type="serendipity")


@needs_gkeyll
@pytest.mark.parametrize("form", ["modal", "nodal", "quad"])
def test_vlasov_hybrid_highest_mode_preserved(form):
  """Coefficient 15 is a product of P1(x), P1(v1), and P2(v2)."""
  source = pg.load(GEN / "hybrid_1x2v_highest.gkyl")
  assert (source.ctx["num_cdim"], source.ctx["num_vdim"]) == (1, 2)
  represented = source.represent(to=form)
  if form == "quad":
    represented = represented.represent(to="modal")
  result = represented.interpolate(num_interp=3)
  axes = [(axis[:-1] + axis[1:]) / 2 for axis in result.grid]
  x, v1, v2 = np.meshgrid(*axes, indexing="ij")
  expected = 1.5 * np.sqrt(5 / 8) * x * v1 * (3 * v2**2 - 1)
  np.testing.assert_allclose(result.values[..., 0],
                             expected,
                             rtol=3e-14,
                             atol=3e-14)
  assert abs(expected[-1, -1, -1]) > 0.17


@needs_gkeyll
def test_hybrid_cache_identity_includes_configuration_velocity_split():
  basis = gpython.basis
  a = basis.get_basis("hybrid", 3, 1, cdim=1, vdim=2)
  b = basis.get_basis("hybrid", 3, 1, cdim=2, vdim=1)
  assert a.num_basis == 16 and b.num_basis == 12 and a is not b
  for c, v, count in [(1, 2, 16), (2, 1, 12)]:
    matrix = basis.interpolation_matrix("hybrid", 3, 1, 3, cdim=c, vdim=v)
    assert matrix.shape == (27, count)
    assert matrix is basis.interpolation_matrix("hybrid",
                                                3,
                                                1,
                                                3,
                                                cdim=c,
                                                vdim=v)


@needs_gkeyll
def test_ambiguous_hybrid_split_requires_explicit_metadata():
  data = pg.GData(ctx={
      "basis_type": "hybrid",
      "poly_order": 1,
      "value_form": "modal"
  })
  data.push([np.array([-1., 1.])] * 3, np.zeros((1, 1, 1, 48)))
  with pytest.raises(ValueError, match="ambiguous"):
    data.interpolate()


@needs_gkeyll
def test_clone_isolates_nested_metadata_native_values_and_grids():
  source = pg.load(GEN / "representation_affine_1d.gkyl")
  original_values = source.values.copy()
  source.ctx["nested"] = {"arrays": [np.array([1., 2.])]}
  cloned = source.clone()
  cloned.ctx["cells"][0] = 100
  cloned.ctx["nested"]["arrays"][0][0] = 9.
  cloned.ctx["_load_metadata"]["file_header"]["cells"][0] = 100
  cloned.grid[0][0] = 12.
  np.testing.assert_array_equal(source.values, original_values)
  assert source.ctx["cells"][0] == 2
  assert source.ctx["nested"]["arrays"][0][0] == 1.
  assert source.ctx["_load_metadata"]["file_header"]["cells"][0] == 2
  assert source.grid[0][0] == -1.
  cloned.ctx["cells"][0] = 2
  assert not np.shares_memory(source.values, cloned.values)
  result = source.select(comp=1)
  result.grid[0][0] = -9.
  assert source.grid[0][0] == -1.


@needs_gkeyll
@pytest.mark.parametrize("form", ["modal", "nodal", "quad"])
def test_hybrid_linear_and_pointwise_arithmetic_use_actual_split(form):
  source = pg.load(GEN / "hybrid_1x2v_highest.gkyl").represent(to=form)
  transformed = source + source + 2.
  if form == "quad":
    transformed = transformed.represent(to="modal")
  result = transformed.interpolate(num_interp=3)
  axes = [(axis[:-1] + axis[1:]) / 2 for axis in result.grid]
  x, v1, v2 = np.meshgrid(*axes, indexing="ij")
  exact = 3 * np.sqrt(5 / 8) * x * v1 * (3 * v2**2 - 1) + 2
  np.testing.assert_allclose(result.values[..., 0],
                             exact,
                             rtol=4e-14,
                             atol=4e-14)
  if form != "modal":
    product = source * source
    np.testing.assert_allclose(product.values,
                               source.values**2,
                               rtol=2e-14,
                               atol=2e-14)


@needs_gkeyll
def test_rpn_identity_owns_nested_metadata():
  source = pg.load(GEN / "representation_affine_1d.gkyl")
  source.ctx["nested"] = {"array": np.array([2., 3.])}
  result = pg.evaluate("f", source)
  result.ctx["nested"]["array"][0] = 100.
  result.ctx["_load_metadata"]["file_header"]["cells"][0] = 100
  assert source.ctx["nested"]["array"][0] == 2.
  assert source.ctx["_load_metadata"]["file_header"]["cells"][0] == 2
