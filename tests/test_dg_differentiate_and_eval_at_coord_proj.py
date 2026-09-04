"""Correctness tests for the ``differentiate``/``eval_at_coord_proj`` DG
operations ported from the old ``GkeyllDGops`` (ctypes) class -- now backed
by the compiled shim (``gpython.kernels``), orchestrated in ``dg.modal``, and
exposed as the ``eval_at_coord_proj`` verb/CLI command (``differentiate`` has
no dedicated verb: it is used internally, the way the old class's
``differentiate`` method fed gk-quantity math, not as a user-facing CLI verb).

Run:  PYTHONPATH=src pytest tests/test_dg_differentiate_and_eval_at_coord_proj.py -v
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(ROOT), "src")
sys.path.insert(0, SRC)  # dedup harmless across the shared test session

import postgkyl as pg  # noqa: E402
from postgkyl import dg, gpython  # noqa: E402

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

DATA = os.path.join(ROOT, "test_data")
GKHYB = os.path.join(DATA, "rt_gk_tcv_iwl_1x2v_p1-elc_250.gkyl")


# ============================================================ differentiate
@needs_gkeyll
def test_differentiate_first_order_on_a_linear_field():
  """f(x) = x on a single cell covering physical x in [0, 2] (dx=2); df/dx
  should be the constant field 1 everywhere."""
  # f(z) = c0*b0 + c1*b1 = c0/sqrt(2) + c1*sqrt(1.5)*z; matching f(z) = 1+z.
  c0 = np.sqrt(2.0)
  c1 = 1.0 / np.sqrt(1.5)
  a = gpython.GkylArray.from_numpy(np.array([[c0, c1]]))

  out = dg.modal.differentiate("serendipity",
                               1,
                               1,
                               a,
                               dir=0,
                               diff_order=1,
                               dx=2.0)

  # A constant field 1 has coefficient c0' = sqrt(2), c1' = 0.
  np.testing.assert_allclose(out.view(), [[np.sqrt(2.0), 0.0]], atol=1e-12)


@needs_gkeyll
def test_differentiate_second_order_on_a_quadratic_field():
  """f(z) = z^2 (p2 serendipity, single cell, dx=2 so d/dx = d/dz);
  d^2f/dz^2 = 2 exactly, independent of position."""
  # Reference basis for 1x p2 serendipity: b0=1/sqrt(2), b1=sqrt(1.5)*z,
  # b2 = sqrt(5/8)*(3z^2-1). f(z)=z^2 => project onto b2 (mean-subtracted
  # quadratic) plus a constant: z^2 = 1/3 + (1/3)*(3z^2-1).
  # c0/sqrt(2) = 1/3 => c0 = sqrt(2)/3; c2*sqrt(5/8) = 1/3 => c2 = 1/(3*sqrt(5/8)).
  c0 = np.sqrt(2.0) / 3.0
  c2 = 1.0 / (3.0 * np.sqrt(5.0 / 8.0))
  a = gpython.GkylArray.from_numpy(np.array([[c0, 0.0, c2]]))

  out = dg.modal.differentiate("serendipity",
                               1,
                               2,
                               a,
                               dir=0,
                               diff_order=2,
                               dx=2.0)

  expected_c0 = 2.0 * np.sqrt(2.0)  # constant field "2" -> coeff0 = 2*sqrt(2)
  np.testing.assert_allclose(out.view()[0, 0], expected_c0, atol=1e-10)
  np.testing.assert_allclose(out.view()[0, 1:], [0.0, 0.0], atol=1e-10)


@needs_gkeyll
def test_differentiate_rejects_out_of_table_combinations():
  a = gpython.GkylArray.alloc(4, 1)  # 1x p1 serendipity-shaped, but wrong basis
  with pytest.raises(NotImplementedError, match="serendipity/tensor"):
    dg.modal.differentiate("gkhybrid", 2, 1, a, dir=0, diff_order=1, dx=1.0)

  a3 = gpython.GkylArray.alloc(gpython.basis.num_basis("tensor", 3, 1), 1)
  with pytest.raises(NotImplementedError, match="ndim"):
    dg.modal.differentiate("tensor", 3, 1, a3, dir=0, diff_order=1, dx=1.0)


# ======================================================== eval_at_coord_proj
@needs_gkeyll
def test_eval_at_coord_proj_matches_direct_polynomial_evaluation():
  """Cross-check: the target's reconstructed value at an arbitrary surviving
  point must equal the donor's own reconstruction at the same point (with
  the eliminated coordinate substituted) -- exactly, since both sides
  evaluate the same underlying polynomial."""
  ndim, poly_order = 2, 1
  basis_type = "serendipity"
  nb = gpython.basis.num_basis(basis_type, ndim, poly_order)
  cells = [2, 1]
  lower, upper = [0.0, 0.0], [2.0, 1.0]

  rng = np.random.default_rng(0)
  coeffs = np.zeros((cells[0] * cells[1], nb))
  coeffs[0, :] = rng.normal(size=nb)
  a = gpython.GkylArray.from_numpy(coeffs)

  grid = {"ndim": ndim, "lower": lower, "upper": upper, "cells": cells}
  y0 = 0.3
  keep_dirs, cells_tar, out, btype, po_tar, cdim_tar, vdim_tar = (
      dg.modal.eval_at_coord_proj(grid,
                                  basis_type,
                                  ndim,
                                  poly_order,
                                  a,
                                  eval_dirs=[1],
                                  eval_coords=[y0]))

  assert keep_dirs == [0]
  assert cells_tar == [2]
  assert btype == "serendipity"
  assert po_tar == 1

  # cell 0 spans x in [0, 1], y in [0, 1]; center (0.5, 0.5), dx=dy=1.
  x0 = 0.2
  zx = 2 * (x0 - 0.5) / 1.0
  zy = 2 * (y0 - 0.5) / 1.0
  donor_val = gpython.basis.eval_matrix(basis_type, ndim, poly_order,
                                        np.array([[zx, zy]])) @ coeffs[0]
  target_val = gpython.basis.eval_matrix(btype, 1, po_tar, np.array(
      [[zx]])) @ out.view()[0]
  np.testing.assert_allclose(target_val, donor_val, atol=1e-10)


@needs_gkeyll
def test_eval_at_coord_proj_full_reduction_uses_the_degenerate_1d_target():
  ndim, poly_order = 1, 1
  basis_type = "serendipity"
  a = gpython.GkylArray.from_numpy(np.array([[1.0, 0.5]]))
  grid = {"ndim": ndim, "lower": [0.0], "upper": [1.0], "cells": [1]}

  keep_dirs, cells_tar, out, btype, po_tar, cdim_tar, vdim_tar = (
      dg.modal.eval_at_coord_proj(grid,
                                  basis_type,
                                  ndim,
                                  poly_order,
                                  a,
                                  eval_dirs=[0],
                                  eval_coords=[0.5]))

  assert keep_dirs == []
  assert cells_tar == [1]
  assert out.size == 1


@needs_gkeyll
def test_eval_at_coord_proj_rejects_eval_dirs_out_of_range():
  a = gpython.GkylArray.alloc(gpython.basis.num_basis("serendipity", 2, 1), 1)
  grid = {"ndim": 2, "lower": [0.0, 0.0], "upper": [1.0, 1.0], "cells": [1, 1]}
  with pytest.raises(ValueError, match="out of range"):
    dg.modal.eval_at_coord_proj(grid,
                                "serendipity",
                                2,
                                1,
                                a,
                                eval_dirs=[2],
                                eval_coords=[0.0])


@needs_gkeyll
def test_eval_at_coord_proj_on_real_gkhybrid_data_matches_donor_reconstruction(
):
  """Real 1x2v gkhybrid data: eliminating the vpar direction should produce
  a target whose basis Gkeyll reports (possibly a different TYPE than the
  donor's, since no gkhybrid convention has a mu-only velocity space), and
  whose reconstructed value matches the donor's at the same physical point."""
  d = pg.load(GKHYB)
  assert d.ctx["basis_type"] == "gkhybrid"
  lo, up = d.bounds
  y0 = 0.5 * (lo[1] + up[1])  # a vpar coordinate inside the domain

  out = d.eval_at_coord_proj([1], [y0])
  assert out.num_dims == 2

  cells = np.asarray(d.ctx["cells"])
  dx = (up - lo) / cells
  # Pick a point in cell (0, 0, 0): conf x and mu at their cell centers.
  zx = 0.0  # cell center -> reference coordinate 0
  zy = 2.0 * (y0 - (lo[1] + 0.5 * dx[1])) / dx[1]
  zmu = 0.0

  donor_val = gpython.basis.eval_matrix("gkhybrid", 3, 1,
                                        np.array([[zx, zy, zmu]])) @ np.asarray(
                                            d.native.view())[0]
  target_val = gpython.basis.eval_matrix(
      out.ctx["basis_type"], 2, out.ctx["poly_order"], np.array(
          [[zx, zmu]])) @ np.asarray(out.native.view())[0]
  np.testing.assert_allclose(target_val, donor_val, atol=1e-8)


# =================================================== operations.eval_at_coord_proj
@needs_gkeyll
def test_ops_eval_at_coord_proj_rejects_numpy_backed_and_non_modal():
  interpolated = pg.load(GKHYB).interpolate()
  with pytest.raises(ValueError, match="native modal data"):
    interpolated.eval_at_coord_proj([1], [0.0])

  nodal = pg.load(GKHYB).to_nodal()
  with pytest.raises(ValueError, match="modal value_form"):
    nodal.eval_at_coord_proj([1], [0.0])


@needs_gkeyll
def test_ops_eval_at_coord_proj_rejects_missing_basis_metadata():
  a = pg.load(GKHYB)
  del a.ctx["poly_order"]
  with pytest.raises(ValueError, match="basis_type/poly_order"):
    a.eval_at_coord_proj([1], [0.0])


@needs_gkeyll
def test_ops_eval_at_coord_proj_tag_label_inplace():
  a = pg.load(GKHYB)
  lo, up = a.bounds
  y0 = 0.5 * (lo[1] + up[1])
  out = a.eval_at_coord_proj([1], [y0], tag="reduced", label="my label")
  assert out.tag == "reduced"
  assert out.label == "my label"
  assert a.num_dims == 3  # original untouched

  b = pg.load(GKHYB)
  mutated = b.eval_at_coord_proj([1], [y0], inplace=True)
  assert mutated is b
  assert b.num_dims == 2


@needs_gkeyll
def test_ops_eval_at_coord_proj_can_eliminate_every_dimension():
  data = pg.load(GKHYB)
  lower, upper = data.bounds
  coordinates = 0.5 * (lower + upper)

  out = data.eval_at_coord_proj([0, 1, 2], coordinates)

  assert out.num_dims == 1
  assert out.ctx["cells"].tolist() == [1]
  np.testing.assert_array_equal(out.grid[0], [0.0, 1.0])
