"""Tests for ``postgkyl.gpython.basis`` -- Gkeyll basis objects + matrices.

Run:  PYTHONPATH=src pytest tests/test_gpython_basis.py -v
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)  # dedup harmless across the shared test session

from postgkyl import gpython  # noqa: E402
from postgkyl.gpython import basis as fb  # noqa: E402

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

pytestmark = needs_gkeyll


def _analytic_num_basis(basis_type: str, ndim: int, poly_order: int) -> int:
  """Independent (from-scratch) count, NOT derived from the shim's table."""
  if basis_type == "tensor":
    return (poly_order + 1)**ndim
  # Serendipity: the standard tensor-product-hypercube serendipity finite
  # element counts (Arnold & Awanou 2011); 1D collapses to the full
  # polynomial space p+1, and 2D matches the textbook 4/8/12-node quad
  # elements (bilinear / quadratic-without-center / cubic serendipity).
  if ndim == 1:
    return poly_order + 1
  if ndim == 2:
    return {0: 1, 1: 4, 2: 8, 3: 12}[poly_order]
  raise NotImplementedError("no independent closed form wired up for this case")


@pytest.mark.parametrize("basis_type,ndim,poly_order", [
    ("serendipity", 1, 0),
    ("serendipity", 1, 1),
    ("serendipity", 1, 2),
    ("serendipity", 1, 3),
    ("serendipity", 2, 0),
    ("serendipity", 2, 1),
    ("serendipity", 2, 2),
    ("serendipity", 2, 3),
    ("tensor", 1, 2),
    ("tensor", 2, 2),
    ("tensor", 3, 1),
])
def test_num_basis_matches_independent_formula(basis_type, ndim, poly_order):
  got = fb.num_basis(basis_type, ndim, poly_order)
  assert got == _analytic_num_basis(basis_type, ndim, poly_order)


def test_analytic_num_basis_helper_rejects_serendipity_3d():
  """The independent reference formula only has closed forms for 1-D/2-D
  serendipity; the parametrized cases above never reach 3-D, so this checks
  the helper's own guard directly."""
  with pytest.raises(NotImplementedError, match="no independent closed form"):
    _analytic_num_basis("serendipity", 3, 1)


def test_get_basis_caches_the_same_object():
  a = fb.get_basis("serendipity", 2, 1)
  b = fb.get_basis("serendipity", 2, 1)
  assert a is b
  # Case-insensitivity shares the same cache entry.
  c = fb.get_basis("SERENDIPITY", 2, 1)
  assert a is c


def test_basis_repr():
  b = fb.get_basis("serendipity", 1, 1)
  r = repr(b)
  assert "serendipity" in r and "ndim=1" in r and "p=1" in r and "N=2" in r


# --------------------------------------------------------- boundary guards
@pytest.mark.parametrize(
    "basis_type,ndim,poly_order",
    [
        ("serendipity", 7, 1),  # ndim above Gkeyll's cart_modal_serendip cap
        ("serendipity", 0, 1),
        ("serendipity", -1, 1),
        ("serendipity", 1, 4),  # poly_order above the ev[4] table
        ("serendipity", 5, 3),  # 5D serendipity tops out at p2
        ("serendipity", 6, 2),  # 6D serendipity tops out at p1
        ("tensor", 3, 3),  # 3D tensor tops out at p2
        ("tensor", 8, 1),
        ("bogus", 1, 1),
    ])
def test_unsupported_combinations_raise_cleanly(basis_type, ndim, poly_order):
  """These would abort the process or read out-of-bounds C tables if the
  Python-side guard were missing (see basis.py's _MAX_POLY_ORDER comment) --
  a clean ValueError, not a crash, is exactly what is being tested here."""
  with pytest.raises(ValueError):
    fb.get_basis(basis_type, ndim, poly_order)


@pytest.mark.parametrize("basis_type,ndim,poly_order", [
    ("serendipity", 5, 2),
    ("serendipity", 6, 1),
    ("tensor", 3, 2),
])
def test_boundary_combinations_that_ARE_supported(basis_type, ndim, poly_order):
  b = fb.get_basis(basis_type, ndim, poly_order)
  assert (b.ndim, b.poly_order) == (ndim, poly_order)


# --------------------------------------------------------------- eval_matrix
def test_eval_matrix_at_cell_center_is_the_constant_mode():
  """b_0 at z=0 is the normalized constant mode 1/sqrt(2)**ndim for
  serendipity/tensor (orthonormal on [-1,1]^ndim with respect to dz)."""
  for ndim in (1, 2, 3):
    m = fb.eval_matrix("serendipity", ndim, 1, np.zeros((1, ndim)))
    assert np.isclose(m[0, 0], (1.0 / np.sqrt(2.0))**ndim)


def test_eval_matrix_reproduces_an_in_basis_polynomial():
  """Build modal coefficients for f(z) = 1 + 2z + 3z^2 (degree <= p=2) via
  nodal_to_modal, then check eval_matrix reproduces f exactly at arbitrary
  points (not just the nodes used to build it)."""
  basis_type, ndim, p = "serendipity", 1, 2
  nodes = fb.node_coords(basis_type, ndim, p)[:, 0]

  def f(z):
    return 1.0 + 2.0 * z + 3.0 * z**2

  fnodal = f(nodes)
  n2m = fb.nodal_to_modal_matrix(basis_type, ndim, p)
  coeffs = n2m @ fnodal

  probe = np.linspace(-1, 1, 11).reshape(-1, 1)
  m = fb.eval_matrix(basis_type, ndim, p, probe)
  got = m @ coeffs
  np.testing.assert_allclose(got, f(probe[:, 0]), atol=1e-12)


def test_nodal_to_modal_and_modal_to_nodal_are_exact_inverses():
  for basis_type, ndim, p in [("serendipity", 1, 2), ("serendipity", 2, 1),
                              ("tensor", 2, 2)]:
    n2m = fb.nodal_to_modal_matrix(basis_type, ndim, p)
    m2n = fb.modal_to_nodal_matrix(basis_type, ndim, p)
    nb = fb.num_basis(basis_type, ndim, p)
    np.testing.assert_allclose(n2m @ m2n, np.eye(nb), atol=1e-12)
    np.testing.assert_allclose(m2n @ n2m, np.eye(nb), atol=1e-12)


def test_modal_quad_round_trip_exact_for_in_degree_polynomials():
  """quad_to_modal(modal_to_quad(c)) == c whenever num_quad >= p+1: the
  q2m projection integrates b_j(z)*f(z), degree <= 2p, and an n-point
  Gauss rule is exact to degree 2n-1, so 2*num_quad-1 >= 2p needs
  num_quad >= p+1 (not merely p, as a naive reading of "degree <= p" would
  suggest -- this is exactly why the num_quad choice matters here)."""
  basis_type, ndim, p, num_quad = "serendipity", 1, 2, 3
  rng = np.random.default_rng(42)
  nb = fb.num_basis(basis_type, ndim, p)
  coeffs = rng.normal(size=nb)

  m2q = fb.modal_to_quad_matrix(basis_type, ndim, p, num_quad)
  q2m = fb.quad_to_modal_matrix(basis_type, ndim, p, num_quad)
  back = q2m @ (m2q @ coeffs)
  np.testing.assert_allclose(back, coeffs, atol=1e-12)


def test_interpolation_matrix_layout_matches_fortran_tensor_order():
  """Row i of a 2D interpolation matrix corresponds to np.unravel_index(i,
  [n,n], order='F') -- dimension 0 fastest."""
  n = 3
  pts_1d = fb.interpolation_points_1d(n)
  pts_2d = fb.tensor_points(pts_1d, 2)
  for i in range(n * n):
    idx = np.unravel_index(i, (n, n), order="F")
    expected = [pts_1d[idx[0]], pts_1d[idx[1]]]
    np.testing.assert_allclose(pts_2d[i], expected)


def test_interpolation_matrix_is_cached_and_read_only():
  m1 = fb.interpolation_matrix("serendipity", 1, 1, 2)
  m2 = fb.interpolation_matrix("serendipity", 1, 1, 2)
  assert m1 is m2
  with pytest.raises(ValueError):
    m1[0, 0] = 5.0


def test_gauss_quad_weights_sum_to_domain_volume():
  for ndim in (1, 2, 3):
    _, w = fb.gauss_quad(ndim, 3)
    assert np.isclose(w.sum(), 2.0**ndim)


def test_node_coords_shape():
  coords = fb.node_coords("serendipity", 2, 1)
  nb = fb.num_basis("serendipity", 2, 1)
  assert coords.shape == (nb, 2)


# --------------------------------------------------------------- hybrid/gkhybrid
@pytest.mark.parametrize("basis_type,ndim,expected_num_basis", [
    ("hybrid", 2, 6),
    ("hybrid", 3, 12),
    ("hybrid", 4, 24),
    ("gkhybrid", 2, 6),
    ("gkhybrid", 3, 12),
    ("gkhybrid", 4, 24),
    ("gkhybrid", 5, 48),
])
def test_hybrid_num_basis_matches_gkeyll_kernel_tables(basis_type, ndim,
                                                       expected_num_basis):
  """Independent counts from gkeyll's own num_basis_list tables in
  gkyl_cart_modal_{hybrid,gkhybrid}_priv.h (and its unit tests
  ctest_basis.c), for the (cdim, vdim) split basis.py derives from ndim."""
  b = fb.get_basis(basis_type, ndim, 1)
  assert (b.ndim, b.poly_order, b.num_basis,
          b.id) == (ndim, 1, expected_num_basis, basis_type)


@pytest.mark.parametrize(
    "basis_type,ndim,poly_order",
    [
        ("hybrid", 1, 1),  # below Gkeyll's ndim>1 assert
        ("hybrid", 5, 1),  # no (cdim, vdim) split Gkeyll compiles kernels for
        ("hybrid", 2, 2),  # hybrid only exists at poly_order 1
        ("gkhybrid", 1, 1),
        ("gkhybrid", 6, 1),
        ("gkhybrid", 3, 2),
    ])
def test_hybrid_unsupported_combinations_raise_cleanly(basis_type, ndim,
                                                       poly_order):
  with pytest.raises(ValueError):
    fb.get_basis(basis_type, ndim, poly_order)


def test_cdim_vdim_raises_for_unsupported_hybrid_ndim():
  with pytest.raises(ValueError, match="Gkeyll's hybrid basis supports ndim"):
    fb.cdim_vdim("hybrid", 5)


def test_hybrid_and_gkhybrid_are_distinct_bases_at_the_same_ndim():
  """ndim=2 exists for both; they must not collide in the cache or alias
  the same compiled basis."""
  hyb = fb.get_basis("hybrid", 2, 1)
  gkhyb = fb.get_basis("gkhybrid", 2, 1)
  assert hyb.id == "hybrid" and gkhyb.id == "gkhybrid"
  assert hyb is not gkhyb


@pytest.mark.parametrize("basis_type,ndim", [
    ("hybrid", 2),
    ("hybrid", 3),
    ("gkhybrid", 2),
    ("gkhybrid", 3),
    ("gkhybrid", 4),
])
def test_hybrid_nodal_to_modal_and_modal_to_nodal_are_exact_inverses(
    basis_type, ndim):
  n2m = fb.nodal_to_modal_matrix(basis_type, ndim, 1)
  m2n = fb.modal_to_nodal_matrix(basis_type, ndim, 1)
  nb = fb.num_basis(basis_type, ndim, 1)
  np.testing.assert_allclose(n2m @ m2n, np.eye(nb), atol=1e-12)
  np.testing.assert_allclose(m2n @ n2m, np.eye(nb), atol=1e-12)
