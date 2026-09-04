"""Tests for ``postgkyl.gpython.kernels`` -- weak algebra, lincomb, reduce, integrate.

Run:  PYTHONPATH=src pytest tests/test_gpython_kernels.py -v
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)  # dedup harmless across the shared test session

from postgkyl import gpython  # noqa: E402
from postgkyl.gpython import kernels as k  # noqa: E402
from postgkyl.gpython.array import GkylArray  # noqa: E402

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

pytestmark = needs_gkeyll


def _smooth_field(basis_type, ndim, p, cells, rng, shift=0.0):
  """Random-but-smooth modal coefficients: only the constant + a small
  perturbation on the higher modes, and shifted away from zero so weak
  division never divides by (near-)zero."""
  nb = gpython.basis.num_basis(basis_type, ndim, p)
  coeffs = rng.normal(scale=0.05, size=(cells, nb))
  coeffs[:, 0] += shift
  return GkylArray.from_numpy(coeffs)


# --------------------------------------------------------------- weak algebra
@pytest.mark.parametrize("ndim,p", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_weak_mul_div_are_inverses_on_smooth_fields(ndim, p):
  rng = np.random.default_rng(42)
  basis_type = "serendipity"
  cells = 6
  a = _smooth_field(basis_type, ndim, p, cells, rng, shift=3.0)
  b = _smooth_field(basis_type, ndim, p, cells, rng, shift=5.0)
  ab = k.weak_mul(basis_type, ndim, p, a, b)
  back = k.weak_div(basis_type, ndim, p, ab, b)
  np.testing.assert_allclose(back.view(), a.view(), atol=1e-10)


def test_weak_inv_matches_weak_div_by_one():
  rng = np.random.default_rng(7)
  basis_type, ndim, p, cells = "serendipity", 1, 1, 4
  a = _smooth_field(basis_type, ndim, p, cells, rng, shift=4.0)
  one = GkylArray.from_numpy(
      np.zeros((cells, gpython.basis.num_basis(basis_type, ndim, p))))
  # constant field 1: coefficient 0 is 1/normalization, i.e. sqrt(2)**ndim
  one.view()  # no-op just to document one is unused below (division test)
  inv_a = k.weak_inv(basis_type, ndim, p, a)
  back = k.weak_mul(basis_type, ndim, p, inv_a, a)
  # a * (1/a) == 1: coefficient 0 equals normalization constant, others ~ 0.
  expect = np.zeros_like(back.view())
  expect[:, 0] = np.sqrt(2.0)
  np.testing.assert_allclose(back.view(), expect, atol=1e-10)


def test_weak_mul_rejects_ncomp_not_a_multiple_of_num_basis():
  basis_type, ndim, p = "serendipity", 1, 1  # num_basis == 2
  a = GkylArray.alloc(3, 4)  # 3 is not a multiple of 2
  b = GkylArray.alloc(3, 4)
  with pytest.raises(ValueError, match="not a multiple"):
    k.weak_mul(basis_type, ndim, p, a, b)


def test_weak_mul_rejects_shape_mismatch():
  basis_type, ndim, p = "serendipity", 1, 1
  a = GkylArray.alloc(2, 4)
  b = GkylArray.alloc(2, 5)  # different size
  with pytest.raises(ValueError, match="shape mismatch"):
    k.weak_mul(basis_type, ndim, p, a, b)


def test_weak_ops_reject_unknown_basis_type():
  a = GkylArray.alloc(2, 4)
  b = GkylArray.alloc(2, 4)
  with pytest.raises(NotImplementedError, match="serendipity"):
    k.weak_mul("bogus", 1, 1, a, b)


@pytest.mark.parametrize("ndim", [4, 5, 6])
def test_weak_mul_div_refuse_ndim_above_3(ndim):
  """gkyl_dg_bin_ops' kernel tables assert(dim < 4) -- a process abort if
  this guard were missing; it must degrade to a clean exception instead."""
  basis = gpython.basis.get_basis("serendipity", ndim, 1)
  a = GkylArray.alloc(basis.num_basis, 3)
  b = GkylArray.alloc(basis.num_basis, 3)
  with pytest.raises(NotImplementedError, match="ndim 1..3"):
    k.weak_mul("serendipity", ndim, 1, a, b)
  with pytest.raises(NotImplementedError, match="ndim 1..3"):
    k.weak_div("serendipity", ndim, 1, a, b)


def test_weak_mul_div_refuse_tensor_poly_order_above_table():
  """Tensor mul/div kernels only go to p2 at ndim 2-3 (p3 slot is NULL)."""
  a = GkylArray.alloc(16, 3)  # shape irrelevant; guard fires first
  b = GkylArray.alloc(16, 3)
  with pytest.raises(NotImplementedError, match="poly_order 0..2"):
    k.weak_mul("tensor", 2, 3, a, b)


def test_weak_inv_rejects_non_p1():
  a = GkylArray.alloc(2, 3)
  with pytest.raises(NotImplementedError, match="p=1 only"):
    k.weak_inv("serendipity", 1, 2, a)


@pytest.mark.parametrize("ndim", [4, 5, 6])
def test_weak_inv_refuses_ndim_above_3(ndim):
  """gkyl_dg_inv_op's kernel table has NO bounds check at all for ndim; this
  guard is the only thing standing between a call and undefined behavior."""
  basis = gpython.basis.get_basis("serendipity", ndim, 1)
  a = GkylArray.alloc(basis.num_basis, 3)
  with pytest.raises(NotImplementedError, match="ndim"):
    k.weak_inv("serendipity", ndim, 1, a)


# --------------------------------------------------- conf-space x phase-space
def test_mul_conf_phase_by_a_unit_constant_conf_field_is_identity_hybrid():
  """Multiplying by a spatially-uniform conf field of true value 1 can never
  raise polynomial degree, so it's an EXACT identity on the phase
  coefficients regardless of what the weak cross-mul kernel computes --
  this is the 1x1v PKPM pairing (serendipity conf x hybrid phase)."""
  cbasis = gpython.basis.get_basis("serendipity", 1, 1)
  pbasis = gpython.basis.get_basis("hybrid", 2, 1)
  conf_cells, phase_cells = [3], [3, 4]
  cop_coeffs = np.zeros((3, cbasis.num_basis))
  cop_coeffs[:, 0] = np.sqrt(2.0)  # constant field value 1 (cdim=1)
  cop = GkylArray.from_numpy(cop_coeffs)
  rng = np.random.default_rng(3)
  pop_coeffs = rng.normal(size=(12, pbasis.num_basis))
  pop = GkylArray.from_numpy(pop_coeffs)
  out = k.weak_mul_conf_phase("serendipity", 1, "hybrid", 2, 1, conf_cells,
                              phase_cells, cop, pop)
  np.testing.assert_allclose(out.view(), pop_coeffs, atol=1e-10)


def test_mul_conf_phase_by_a_unit_constant_conf_field_is_identity_gkhybrid():
  """Same identity check for the 1x2v gyrokinetic pairing (serendipity conf
  x gkhybrid phase, cdim=1 vdim=2)."""
  cbasis = gpython.basis.get_basis("serendipity", 1, 1)
  pbasis = gpython.basis.get_basis("gkhybrid", 3, 1)
  conf_cells, phase_cells = [4], [4, 3, 2]
  cop_coeffs = np.zeros((4, cbasis.num_basis))
  cop_coeffs[:, 0] = np.sqrt(2.0)
  cop = GkylArray.from_numpy(cop_coeffs)
  rng = np.random.default_rng(5)
  pop_coeffs = rng.normal(size=(24, pbasis.num_basis))
  pop = GkylArray.from_numpy(pop_coeffs)
  out = k.weak_mul_conf_phase("serendipity", 1, "gkhybrid", 3, 1, conf_cells,
                              phase_cells, cop, pop)
  np.testing.assert_allclose(out.view(), pop_coeffs, atol=1e-10)


def test_mul_conf_phase_by_a_unit_constant_conf_field_is_identity_serendipity():
  """Same-family serendipity conf x serendipity phase also goes through
  gkyl_dg_mul_conf_phase_op_range (not the same-basis gkyl_dg_mul_op path,
  since cdim != pdim), so it needs its own identity check."""
  cbasis = gpython.basis.get_basis("serendipity", 1, 2)
  pbasis = gpython.basis.get_basis("serendipity", 2, 2)
  conf_cells, phase_cells = [3], [3, 5]
  cop_coeffs = np.zeros((3, cbasis.num_basis))
  cop_coeffs[:, 0] = np.sqrt(2.0)
  cop = GkylArray.from_numpy(cop_coeffs)
  rng = np.random.default_rng(9)
  pop_coeffs = rng.normal(size=(15, pbasis.num_basis))
  pop = GkylArray.from_numpy(pop_coeffs)
  out = k.weak_mul_conf_phase("serendipity", 1, "serendipity", 2, 2, conf_cells,
                              phase_cells, cop, pop)
  np.testing.assert_allclose(out.view(), pop_coeffs, atol=1e-10)


def test_mul_conf_phase_rejects_ncomp_mismatch():
  cop = GkylArray.alloc(3, 3)  # hybrid conf num_basis is 2, not 3
  pop = GkylArray.alloc(6, 12)
  with pytest.raises(ValueError, match="single-field only"):
    k.weak_mul_conf_phase("serendipity", 1, "hybrid", 2, 1, [3], [3, 4], cop,
                          pop)


def test_mul_conf_phase_rejects_non_serendipity_conf_for_hybrid():
  cop = GkylArray.alloc(2, 3)
  pop = GkylArray.alloc(6, 12)
  with pytest.raises(NotImplementedError, match="serendipity conf basis"):
    k.weak_mul_conf_phase("tensor", 1, "hybrid", 2, 1, [3], [3, 4], cop, pop)


def test_mul_conf_phase_rejects_mismatched_ser_ten_families():
  cop = GkylArray.alloc(2, 3)
  pop = GkylArray.alloc(4, 15)
  with pytest.raises(NotImplementedError, match="phase basis type alone"):
    k.weak_mul_conf_phase("tensor", 1, "serendipity", 2, 1, [3], [3, 5], cop,
                          pop)


def test_mul_conf_phase_rejects_kernel_table_gap():
  """pdim=5, cdim=1 has no serendipity cross-mul kernel at all (NULL in
  ser_cross_mul_list) -- must raise cleanly, not call through a NULL
  function pointer."""
  cop = GkylArray.alloc(2, 2)
  pop = GkylArray.alloc(32, 32)
  with pytest.raises(NotImplementedError, match="no serendipity conf\\*phase"):
    k.weak_mul_conf_phase("serendipity", 1, "serendipity", 5, 1, [2],
                          [2, 2, 2, 2, 2], cop, pop)


def test_mul_conf_phase_rejects_cells_array_size_mismatch():
  cop = GkylArray.alloc(2, 3)
  pop = GkylArray.alloc(4, 20)  # cells [3, 5] imply size 15, not 20
  with pytest.raises(ValueError, match="incompatible"):
    k.weak_mul_conf_phase("serendipity", 1, "serendipity", 2, 1, [3], [3, 5],
                          cop, pop)


def test_mul_conf_phase_rejects_phase_ndim_not_exceeding_conf_ndim():
  cop = GkylArray.alloc(4, 4)
  pop = GkylArray.alloc(4, 4)
  with pytest.raises(ValueError, match="must exceed"):
    k.weak_mul_conf_phase("serendipity", 2, "serendipity", 2, 1, [2, 2], [2, 2],
                          cop, pop)


# ---------------------------------------------------------- coefficient ops
def test_lincomb_matches_numpy():
  rng = np.random.default_rng(1)
  a = GkylArray.from_numpy(rng.normal(size=(5, 3)))
  b = GkylArray.from_numpy(rng.normal(size=(5, 3)))
  out = k.lincomb(2.0, a, -1.5, b)
  np.testing.assert_allclose(out.view(), 2.0 * a.view() - 1.5 * b.view())


def test_lincomb_rejects_shape_mismatch():
  a = GkylArray.alloc(2, 4)
  b = GkylArray.alloc(3, 4)
  with pytest.raises(ValueError, match="shape mismatch"):
    k.lincomb(1.0, a, 1.0, b)


def test_scale_matches_numpy_and_does_not_mutate_input():
  a = GkylArray.from_numpy(np.arange(6, dtype=np.float64).reshape(3, 2))
  original = a.view().copy()
  out = k.scale(a, -2.0)
  np.testing.assert_allclose(out.view(), -2.0 * original)
  np.testing.assert_allclose(a.view(), original)


def test_shiftc_matches_numpy_and_does_not_mutate_input():
  a = GkylArray.from_numpy(np.zeros((3, 2)))
  out = k.shiftc(a, 7.0, 1)
  expect = np.zeros((3, 2))
  expect[:, 1] = 7.0
  np.testing.assert_allclose(out.view(), expect)
  np.testing.assert_allclose(a.view(), np.zeros((3, 2)))


# ---------------------------------------------------------------- reductions
def test_reduce_of_constant_coefficients():
  a = GkylArray.from_numpy(np.full((4, 2), 3.0))
  np.testing.assert_allclose(k.reduce(a, k.GKYL_SUM), [12.0, 12.0])
  np.testing.assert_allclose(k.reduce(a, k.GKYL_MIN), [3.0, 3.0])
  np.testing.assert_allclose(k.reduce(a, k.GKYL_MAX), [3.0, 3.0])


def test_dg_reduce_of_constant_field_min_max_match_the_constant():
  """min/max of a truly constant field equal that constant regardless of how
  many Gauss-Legendre nodes per cell the kernel evaluates at."""
  basis_type, ndim, p = "serendipity", 1, 1
  nb = gpython.basis.num_basis(basis_type, ndim, p)
  coeffs = np.zeros((5, nb))
  coeffs[:, 0] = 3.0 * np.sqrt(2.0)  # constant mode -> field value 3.0
  a = GkylArray.from_numpy(coeffs)
  assert np.isclose(k.dg_reduce(basis_type, ndim, p, a, 0, "min"), 3.0)
  assert np.isclose(k.dg_reduce(basis_type, ndim, p, a, 0, "max"), 3.0)


def test_dg_reduce_sum_scales_with_cell_count():
  """`sum` totals the per-node field values across every cell (not divided
  by node count), so doubling identical cells must exactly double it --
  a cell-count-independent way to check the "sum over the field" semantics
  without needing to know the kernel's internal Gauss-node count."""
  basis_type, ndim, p = "serendipity", 1, 1
  nb = gpython.basis.num_basis(basis_type, ndim, p)

  def const_field(ncells, value):
    coeffs = np.zeros((ncells, nb))
    coeffs[:, 0] = value * np.sqrt(2.0)
    return GkylArray.from_numpy(coeffs)

  small = k.dg_reduce(basis_type, ndim, p, const_field(3, 3.0), 0, "sum")
  big = k.dg_reduce(basis_type, ndim, p, const_field(6, 3.0), 0, "sum")
  assert small > 0
  assert np.isclose(big, 2.0 * small)


def test_dg_reduce_min_max_at_the_gauss_legendre_nodes_for_a_linear_field():
  """min/max are evaluated at the basis's Gauss-Legendre quadrature NODES
  (interior points), not the cell edges -- so for f(z) = 3 + 2z they equal f
  at the nodes nearest each end, not the true f(-1)/f(1) domain extrema.
  Serendipity p=1 in 1D uses the 2-point rule at z = +-1/sqrt(3)."""
  basis_type, ndim, p = "serendipity", 1, 1
  # modal coefficients of 3 + 2z in the (normalized Legendre) basis:
  # b0 = 1/sqrt(2), b1 = sqrt(3/2) z  =>  c0 = 3*sqrt(2), c1 = 2/sqrt(3/2)
  c0 = 3.0 * np.sqrt(2.0)
  c1 = 2.0 / np.sqrt(1.5)
  a = GkylArray.from_numpy(np.array([[c0, c1]]))
  node = 1.0 / np.sqrt(3.0)
  assert np.isclose(k.dg_reduce(basis_type, ndim, p, a, 0, "min"),
                    3.0 - 2.0 * node)
  assert np.isclose(k.dg_reduce(basis_type, ndim, p, a, 0, "max"),
                    3.0 + 2.0 * node)


def test_dg_reduce_rejects_bad_op_and_bad_comp():
  a = GkylArray.alloc(2, 3)
  with pytest.raises(ValueError, match="op"):
    k.dg_reduce("serendipity", 1, 1, a, 0, "bogus")
  with pytest.raises(ValueError, match="comp"):
    k.dg_reduce("serendipity", 1, 1, a, 5, "sum")


# ----------------------------------------------------------------- integrate
def test_integrate_constant_field_equals_constant_times_volume():
  basis_type, ndim, p = "serendipity", 1, 1
  nb = gpython.basis.num_basis(basis_type, ndim, p)
  cells = 4
  coeffs = np.zeros((cells, nb))
  coeffs[:, 0] = 2.0 * np.sqrt(2.0)  # constant field value 2.0
  a = GkylArray.from_numpy(coeffs)
  grid = {
      "ndim": 1,
      "lower": np.array([0.0]),
      "upper": np.array([2.0]),
      "cells": np.array([cells])
  }
  result = k.integrate(grid, basis_type, p, a)
  np.testing.assert_allclose(result, [2.0 * 2.0])  # value * volume


def test_integrate_abs_and_sq_ops():
  basis_type, ndim, p = "serendipity", 1, 1
  nb = gpython.basis.num_basis(basis_type, ndim, p)
  coeffs = np.zeros((3, nb))
  coeffs[:, 0] = -2.0 * np.sqrt(2.0)  # constant field value -2.0
  a = GkylArray.from_numpy(coeffs)
  grid = {
      "ndim": 1,
      "lower": np.array([0.0]),
      "upper": np.array([3.0]),
      "cells": np.array([3])
  }
  none = k.integrate(grid, basis_type, p, a, op="none")
  absr = k.integrate(grid, basis_type, p, a, op="abs")
  sq = k.integrate(grid, basis_type, p, a, op="sq")
  np.testing.assert_allclose(none, [-6.0])
  np.testing.assert_allclose(absr, [6.0])
  np.testing.assert_allclose(sq, [12.0])  # (-2)^2 * volume(3) = 12


def test_integrate_factor_scales_the_result():
  basis_type, ndim, p = "serendipity", 1, 1
  nb = gpython.basis.num_basis(basis_type, ndim, p)
  coeffs = np.zeros((2, nb))
  coeffs[:, 0] = np.sqrt(2.0)
  a = GkylArray.from_numpy(coeffs)
  grid = {
      "ndim": 1,
      "lower": np.array([0.0]),
      "upper": np.array([2.0]),
      "cells": np.array([2])
  }
  result = k.integrate(grid, basis_type, p, a, factor=10.0)
  np.testing.assert_allclose(result, [20.0])


def test_integrate_rejects_bad_op():
  a = GkylArray.alloc(2, 2)
  grid = {"ndim": 1, "lower": [0.0], "upper": [1.0], "cells": [2]}
  with pytest.raises(ValueError, match="op"):
    k.integrate(grid, "serendipity", 1, a, op="bogus")


def test_integrate_rejects_unsupported_basis_or_poly_order():
  a = GkylArray.alloc(2, 2)
  grid = {"ndim": 1, "lower": [0.0], "upper": [1.0], "cells": [2]}
  with pytest.raises(NotImplementedError):
    k.integrate(grid, "tensor", 1, a)
  with pytest.raises(NotImplementedError):
    k.integrate(grid, "serendipity", 3, a)  # p3 unsupported by the kernel set


def test_integrate_rejects_ndim_above_3():
  basis = gpython.basis.get_basis("serendipity", 4, 1)
  a = GkylArray.alloc(basis.num_basis, 6)
  grid = {
      "ndim": 4,
      "lower": np.zeros(4),
      "upper": np.ones(4),
      "cells": np.array([1, 1, 1, 6])
  }
  with pytest.raises(NotImplementedError, match="ndim 1-3"):
    k.integrate(grid, "serendipity", 1, a)


def test_integrate_rejects_grid_array_mismatch():
  basis_type, ndim, p = "serendipity", 1, 1
  a = GkylArray.alloc(gpython.basis.num_basis(basis_type, ndim, p), 4)
  grid = {
      "ndim": 1,
      "lower": np.array([0.0]),
      "upper": np.array([1.0]),
      "cells": np.array([5])
  }  # 5 != a.size (4)
  with pytest.raises(ValueError, match="do not cover"):
    k.integrate(grid, basis_type, p, a)


# ------------------------------------------------------------------ average
def _const_field(basis_type, ndim, p, cells, value):
  nb = gpython.basis.num_basis(basis_type, ndim, p)
  b0 = 2.0**(-ndim / 2.0)
  coeffs = np.zeros((int(np.prod(cells)), nb))
  coeffs[:, 0] = value / b0
  return GkylArray.from_numpy(coeffs)


def test_array_average_partial_reduction_of_constant_field_is_exact():
  """A genuinely partial reduction (some dims survive) is a proper weak
  contraction: the surviving 1D field's coefficient 0 is the standard
  b0-normalized representation of the (unchanged, since the field is
  spatially constant) value -- no basis-dependent surprises here."""
  basis_type, p = "serendipity", 1
  cells = [4, 3]
  a = _const_field(basis_type, 2, p, cells, 3.0)
  grid = {
      "ndim": 2,
      "lower": np.array([0.0, 0.0]),
      "upper": np.array([2.0, 1.0]),
      "cells": np.array(cells)
  }
  out = k.array_average(grid,
                        basis_type,
                        p,
                        ndim_avg=1,
                        cells_avg=[cells[0]],
                        avg_dim=[0, 1],
                        a=a)
  expect = np.zeros((cells[0], gpython.basis.num_basis(basis_type, 1, p)))
  expect[:, 0] = 3.0 / (2.0**(-1 / 2.0))
  np.testing.assert_allclose(out.view(), expect, atol=1e-10)


def test_array_average_full_reduction_unweighted_writes_a_raw_value():
  """The degenerate (every dim averaged) unweighted kernel
  (gkyl_array_average_NxYY_avg<all dirs>) writes a single raw VALUE into
  coefficient 0 -- not a b0-normalized coefficient, unlike the partial-
  reduction case above. This is exactly the asymmetry
  ``dg.modal.average`` corrects for (test_dg_modal_average_* in
  test_coverage_leaf.py); this test pins the raw kernel behavior itself."""
  basis_type, p = "serendipity", 1
  cells = [4]
  a = _const_field(basis_type, 1, p, cells, 3.0)
  grid = {
      "ndim": 1,
      "lower": np.array([0.0]),
      "upper": np.array([2.0]),
      "cells": np.array(cells)
  }
  out = k.array_average(grid,
                        basis_type,
                        p,
                        ndim_avg=1,
                        cells_avg=[1],
                        avg_dim=[1],
                        a=a)
  np.testing.assert_allclose(out.view()[0, 0], 3.0, atol=1e-10)
  np.testing.assert_allclose(out.view()[0, 1:], 0.0, atol=1e-10)


def test_array_average_full_reduction_weighted_by_a_uniform_weight_matches_integrate(
):
  """With ANY weight (even spatially uniform), the kernel performs a real
  weak division, so the output IS a properly b0-normalized coefficient --
  matching gkyl_array_integrate / volume for a uniform-weight average."""
  basis_type, p = "serendipity", 1
  cells = [4]
  value = 3.0
  a = _const_field(basis_type, 1, p, cells, value)
  w = _const_field(basis_type, 1, p, cells, 2.0)
  grid = {
      "ndim": 1,
      "lower": np.array([0.0]),
      "upper": np.array([2.0]),
      "cells": np.array(cells)
  }
  out = k.array_average(grid,
                        basis_type,
                        p,
                        ndim_avg=1,
                        cells_avg=[1],
                        avg_dim=[1],
                        a=a,
                        weight=w)
  b0 = 2.0**(-1 / 2.0)
  np.testing.assert_allclose(out.view()[0, 0] * b0, value, atol=1e-10)


def test_array_average_rejects_unsupported_basis_or_poly_order():
  a = GkylArray.alloc(2, 4)
  grid = {"ndim": 1, "lower": [0.0], "upper": [1.0], "cells": [4]}
  with pytest.raises(NotImplementedError, match="serendipity p1-p2"):
    k.array_average(grid, "tensor", 1, 1, [1], [1], a)
  with pytest.raises(NotImplementedError, match="serendipity p1-p2"):
    k.array_average(grid, "serendipity", 3, 1, [1], [1], a)


def test_array_average_rejects_ndim_above_3():
  basis = gpython.basis.get_basis("serendipity", 4, 1)
  a = GkylArray.alloc(basis.num_basis, 6)
  grid = {
      "ndim": 4,
      "lower": np.zeros(4),
      "upper": np.ones(4),
      "cells": np.array([1, 1, 1, 6])
  }
  with pytest.raises(NotImplementedError, match="ndim 1-3"):
    k.array_average(grid, "serendipity", 1, 1, [1, 1, 1, 6], [1, 0, 0, 0], a)


def test_array_average_rejects_ncomp_not_single_field():
  basis_type, p = "serendipity", 1
  a = GkylArray.alloc(4,
                      4)  # 4 comps: 2 fields of num_basis=2, not single-field
  grid = {"ndim": 1, "lower": [0.0], "upper": [1.0], "cells": [4]}
  with pytest.raises(ValueError, match="single-field only"):
    k.array_average(grid, basis_type, p, 1, [1], [1], a)


def test_array_average_rejects_weight_shape_mismatch():
  basis_type, p = "serendipity", 1
  cells = [4]
  a = _const_field(basis_type, 1, p, cells, 3.0)
  w = GkylArray.alloc(2, 3)  # size 3 != a.size (4)
  grid = {"ndim": 1, "lower": [0.0], "upper": [1.0], "cells": np.array(cells)}
  with pytest.raises(ValueError, match="weight"):
    k.array_average(grid, basis_type, p, 1, [1], [1], a, weight=w)


def test_array_average_rejects_grid_array_mismatch():
  basis_type, p = "serendipity", 1
  a = GkylArray.alloc(gpython.basis.num_basis(basis_type, 1, p), 4)
  grid = {
      "ndim": 1,
      "lower": np.array([0.0]),
      "upper": np.array([1.0]),
      "cells": np.array([5])
  }  # 5 != a.size (4)
  with pytest.raises(ValueError, match="do not cover"):
    k.array_average(grid, basis_type, p, 1, [1], [1], a)


# -------------------------------------------------------------- differentiate
def test_weak_differentiate_rejects_poly_order_above_table():
  a = GkylArray.alloc(2, 3)
  with pytest.raises(NotImplementedError, match="poly_order 1..2"):
    k.weak_differentiate("serendipity", 1, 3, dir=0, diff_order=1, dx=1.0, a=a)


def test_weak_differentiate_rejects_dir_out_of_range():
  a = GkylArray.alloc(2, 3)
  with pytest.raises(ValueError, match="out of range"):
    k.weak_differentiate("serendipity", 1, 1, dir=5, diff_order=1, dx=1.0, a=a)


def test_weak_differentiate_rejects_bad_diff_order():
  a = GkylArray.alloc(2, 3)
  with pytest.raises(ValueError, match="order must be 1 or 2"):
    k.weak_differentiate("serendipity", 1, 1, dir=0, diff_order=3, dx=1.0, a=a)


# ----------------------------------------------------------- eval_at_coord_proj
def test_eval_at_coord_proj_rejects_gkhybrid_poly_order_above_1():
  a = GkylArray.alloc(2, 2)
  grid = {
      "ndim": 3,
      "lower": [0.0, 0.0, 0.0],
      "upper": [1.0, 1.0, 1.0],
      "cells": [1, 1, 1]
  }
  with pytest.raises(NotImplementedError, match="poly_order 1 only"):
    k.eval_at_coord_proj("gkhybrid", 3, 2, 1, grid, [0], [0.0], 1, [1], a)


def test_eval_at_coord_proj_rejects_unknown_basis_type():
  a = GkylArray.alloc(2, 2)
  grid = {"ndim": 1, "lower": [0.0], "upper": [1.0], "cells": [1]}
  with pytest.raises(NotImplementedError, match="serendipity/tensor/gkhybrid"):
    k.eval_at_coord_proj("hybrid", 1, 1, 1, grid, [0], [0.0], 1, [1], a)


def test_eval_at_coord_proj_rejects_tensor_ndim_above_table():
  a = GkylArray.alloc(2, 2)
  grid = {"ndim": 4, "lower": [0.0] * 4, "upper": [1.0] * 4, "cells": [1] * 4}
  with pytest.raises(NotImplementedError, match="ndim"):
    k.eval_at_coord_proj("tensor", 4, 1, 4, grid, [0], [0.0], 1, [1], a)


def test_eval_at_coord_proj_rejects_poly_order_above_table():
  a = GkylArray.alloc(2, 2)
  grid = {"ndim": 1, "lower": [0.0], "upper": [1.0], "cells": [1]}
  with pytest.raises(NotImplementedError, match="poly_order 1..2"):
    k.eval_at_coord_proj("serendipity", 1, 3, 1, grid, [0], [0.0], 1, [1], a)


def test_eval_at_coord_proj_rejects_eval_dirs_out_of_range():
  a = GkylArray.alloc(2, 2)
  grid = {"ndim": 2, "lower": [0.0, 0.0], "upper": [1.0, 1.0], "cells": [1, 1]}
  with pytest.raises(ValueError, match="out of range"):
    k.eval_at_coord_proj("serendipity", 2, 1, 2, grid, [5], [0.0], 1, [1], a)


def test_eval_at_coord_proj_rejects_grid_array_mismatch():
  basis_type, ndim, p = "serendipity", 1, 1
  nb = gpython.basis.num_basis(basis_type, ndim, p)
  a = GkylArray.alloc(nb, 4)
  grid = {"ndim": ndim, "lower": [0.0], "upper": [1.0], "cells": [5]}
  with pytest.raises(ValueError, match="do not cover"):
    k.eval_at_coord_proj(basis_type, ndim, p, ndim, grid, [0], [0.0], 1, [1], a)


def test_eval_at_coord_proj_rejects_eval_dirs_coords_length_mismatch():
  basis_type, ndim, p = "serendipity", 1, 1
  nb = gpython.basis.num_basis(basis_type, ndim, p)
  a = GkylArray.alloc(nb, 4)
  grid = {"ndim": ndim, "lower": [0.0], "upper": [1.0], "cells": [4]}
  with pytest.raises(ValueError, match="same length"):
    k.eval_at_coord_proj(basis_type, ndim, p, ndim, grid, [0], [0.1, 0.2], 1,
                         [1], a)


# ------------------------------------------------------------------ powsqrt
def test_powsqrt_of_a_constant_field_is_exact():
  basis_type, ndim, p, cells = "serendipity", 1, 1, 4
  nb = gpython.basis.num_basis(basis_type, ndim, p)
  b0 = 2.0**(-ndim / 2.0)
  coeffs = np.zeros((cells, nb))
  coeffs[:, 0] = 4.0 / b0  # constant field value 4.0
  a = GkylArray.from_numpy(coeffs)
  out = k.powsqrt(basis_type, ndim, p, [cells], a, 1.0)
  np.testing.assert_allclose(out.view()[:, 0] * b0, 2.0, atol=1e-12)


@pytest.mark.parametrize("exponent", [1.0, -1.0, 3.0])
def test_powsqrt_matches_the_apply_pointwise_quadrature_path(exponent):
  """``gkyl_proj_powsqrt_on_basis`` and ``dg.rep.apply_pointwise`` both
  project through the same modal<->quadrature matrices (``basis.py``'s
  Gauss-Legendre rule), so they must agree to quadrature precision on a
  genuinely varying (non-constant) field -- this is the cross-check
  ``REFACTOR_GKEYLL_FFI.md``'s ``.apply()`` verb already exercises,
  independent of the compiled kernel."""
  from postgkyl import dg

  basis_type, ndim, p, cells = "serendipity", 1, 1, 4
  rng = np.random.default_rng(3)
  nb = gpython.basis.num_basis(basis_type, ndim, p)
  coeffs = rng.normal(scale=0.05, size=(cells, nb))
  coeffs[:, 0] += 4.0  # shifted positive so pow(sqrt(.), .) stays well-defined
  a = GkylArray.from_numpy(coeffs)
  num_quad = p + 1

  out = k.powsqrt(basis_type, ndim, p, [cells], a, exponent, num_quad=num_quad)
  expect = dg.rep.apply_pointwise(
      basis_type, ndim, p, a,
      lambda v: np.power(np.sqrt(np.where(v < 0, 1e-40, v)), exponent),
      num_quad)
  np.testing.assert_allclose(out.view(), expect.view(), atol=1e-10)


def test_powsqrt_rejects_multi_component_input():
  """The kernel has no field-index argument, so a multi-component (vector)
  field must be refused here (looping per field is ``dg.modal.powsqrt``'s
  job, not this thin binding's)."""
  basis_type, ndim, p, cells = "serendipity", 1, 1, 4
  nb = gpython.basis.num_basis(basis_type, ndim, p)
  a = GkylArray.alloc(3 * nb, cells)  # 3 physical components
  with pytest.raises(ValueError, match="single-field only"):
    k.powsqrt(basis_type, ndim, p, [cells], a, 1.0)


def test_powsqrt_rejects_num_quad_below_one():
  basis_type, ndim, p, cells = "serendipity", 1, 1, 4
  nb = gpython.basis.num_basis(basis_type, ndim, p)
  a = GkylArray.alloc(nb, cells)
  with pytest.raises(ValueError, match="num_quad"):
    k.powsqrt(basis_type, ndim, p, [cells], a, 1.0, num_quad=0)


def test_powsqrt_rejects_cells_not_covering_the_array():
  basis_type, ndim, p = "serendipity", 1, 1
  nb = gpython.basis.num_basis(basis_type, ndim, p)
  a = GkylArray.alloc(nb, 4)
  with pytest.raises(ValueError, match="do not cover"):
    k.powsqrt(basis_type, ndim, p, [5], a, 1.0)
