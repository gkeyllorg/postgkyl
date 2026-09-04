"""Tests for ``postgkyl.dg.rep`` -- modal · nodal · quad representation changes.

This is the module's dedicated home post-relocation (``gpython/rep.py`` ->
``dg/rep.py``, layer 03-dg job 1); defensive/edge-case branches for the same
module are also exercised from ``tests/test_coverage_leaf.py`` (a shared
leaf/engine coverage file predating this move). See ``CLAUDE.md``'s "Engine
layers" section for why representation changes live in ``dg``, not ``gpython``.

Run:  PYTHONPATH=src pytest tests/test_dg_rep.py -v
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)  # dedup harmless across the shared test session

from postgkyl import gpython, dg  # noqa: E402

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")
pytestmark = needs_gkeyll


def _linear_field(basis_type, ndim, poly_order, cells, nfields=1):
  """An exactly-representable modal field: coefficient 0 (mean) = cell index,
  everything else zero -- lets every conversion be checked against a value
  known independently of the code under test."""
  nb = gpython.basis.num_basis(basis_type, ndim, poly_order)
  ncells = int(np.prod(cells))
  vals = np.zeros((ncells, nfields * nb))
  for f in range(nfields):
    vals[:, f * nb] = np.arange(ncells) + f  # only the mean coefficient
  return gpython.GkylArray.from_numpy(vals), nb


def test_modal_to_nodal_to_modal_round_trips_exactly():
  arr, nb = _linear_field("serendipity", 2, 1, [3, 3])
  nodal = dg.rep.modal_to_nodal("serendipity", 2, 1, arr)
  back = dg.rep.nodal_to_modal("serendipity", 2, 1, nodal)
  np.testing.assert_allclose(back.view(), arr.view(), atol=1e-13)


def test_modal_to_nodal_matches_a_directly_evaluated_constant_field():
  """A pure mean-coefficient field is a constant per cell -- every nodal
  value must equal that constant, checked against the analytic normalized
  constant basis function b0 = 2^(-ndim/2), independent of the shim."""
  ndim, poly_order = 1, 1
  arr, nb = _linear_field("serendipity", ndim, poly_order, [4])
  nodal = dg.rep.modal_to_nodal("serendipity", ndim, poly_order, arr)
  b0 = 2.0**(-ndim / 2.0)
  expected = (np.arange(4) * b0)[:, None] * np.ones(nb)
  np.testing.assert_allclose(nodal.view(), expected, atol=1e-13)


@pytest.mark.parametrize("num_quad", [2, 3])
def test_quad_round_trip_exact_for_in_basis_field(num_quad):
  """modal -> quad -> modal is exact whenever num_quad >= poly_order + 1."""
  arr, nb = _linear_field("serendipity", 1, 1, [5], nfields=2)
  quad = dg.rep.modal_to_quad("serendipity", 1, 1, arr, num_quad)
  back = dg.rep.quad_to_modal("serendipity", 1, 1, quad, num_quad)
  np.testing.assert_allclose(back.view(), arr.view(), atol=1e-12)


def test_wrap_round_trips_values_unchanged():
  values = np.arange(12.0).reshape(3, 4)
  wrapped = dg.rep.wrap(values)
  np.testing.assert_array_equal(wrapped.view(), values)


def test_apply_pointwise_sqrt_matches_numpy_after_interpolate():
  """fn applied via quadrature matches applying fn directly to the exact
  (interpolated) values, for an in-basis-representable nonnegative field."""
  ndim, poly_order = 1, 1
  arr, nb = _linear_field("serendipity", ndim, poly_order, [4])
  arr = gpython.kernels.shiftc(arr, 5.0, 0)  # keep the field positive for sqrt
  out = dg.rep.apply_pointwise(ndim=ndim,
                               poly_order=poly_order,
                               basis_type="serendipity",
                               arr=arr,
                               fn=np.sqrt,
                               num_quad=poly_order + 1)
  grid, direct_vals = dg.interpolate(arr.view(), [np.linspace(0, 4, 5)],
                                     poly_order=poly_order,
                                     basis_type="serendipity")
  grid, sqrt_of_applied = dg.interpolate(out.view(), [np.linspace(0, 4, 5)],
                                         poly_order=poly_order,
                                         basis_type="serendipity")
  np.testing.assert_allclose(sqrt_of_applied, np.sqrt(direct_vals), atol=1e-10)


def test_materialize_nodal_matches_modal_to_nodal_values():
  ndim, poly_order = 1, 1
  arr, nb = _linear_field("serendipity", ndim, poly_order, [3])
  nodal = dg.rep.modal_to_nodal("serendipity", ndim, poly_order, arr)
  grid = [np.linspace(0.0, 3.0, 4)]
  edges, out = dg.rep.materialize("serendipity", ndim, poly_order, nodal, grid,
                                  "nodal")
  assert out.shape == (2 * 3, 1)  # 2 tensor nodes/cell * 3 cells, 1 field
  np.testing.assert_allclose(np.sort(out[:, 0].reshape(3, 2), axis=1),
                             np.sort(nodal.view(), axis=1),
                             atol=1e-13)
