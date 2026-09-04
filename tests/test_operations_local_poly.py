"""Tests for the ``local_poly`` verb -- the discontinuity-preserving DG
plotting mesh (see ``dg.interpolate.local_poly``), ported from the old
Typer-era ``dg_local_poly`` command (``PLAN.md``/``14-cli.md``: initially
deferred, since the old implementation depended on hand-derived per-order
polynomial tables (``modalDG/kernels/expand_*d.py``, serendipity only) with
no equivalent in the new engine -- superseded here by
``gpython.basis.eval_matrix``, which evaluates any basis at arbitrary
points through Gkeyll's own compiled basis-eval).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")
pytestmark = needs_gkeyll

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
GEN = os.path.join(DATA, "generated")
F1D = os.path.join(GEN, "1d_ms_p1.gkyl")
F2D = os.path.join(GEN, "2d_ms_p2.gkyl")
F3D = os.path.join(GEN, "3d_ms_p1.gkyl")

# 1D p1 serendipity basis, evaluated at the reference-cell faces.
_B0 = 0.7071067811865475
_B1 = 1.224744871391589


def test_matches_hand_evaluated_basis_at_cell_faces():
  d = pg.load(F1D)
  c0, c1 = d.get_values()[0]
  expect_left = _B0 * c0 - _B1 * c1
  expect_right = _B0 * c0 + _B1 * c1

  lp = d.local_poly(npoints=2)
  np.testing.assert_allclose(lp.get_values()[0, 0], expect_left)
  np.testing.assert_allclose(lp.get_values()[1, 0], expect_right)
  assert lp.grid[0][0] == pytest.approx(0.0)
  assert lp.grid[0][1] == pytest.approx(1.0 / d.num_cells[0])


def test_inserts_nan_at_every_interior_cell_interface():
  d = pg.load(F1D)
  npoints = 3
  lp = d.local_poly(npoints=npoints)
  values = lp.get_values()[:, 0]
  num_cells = int(d.num_cells[0])

  # One NaN spliced in after every cell's block of `npoints` raw points,
  # except the last cell (no interface after the final face).
  assert values.shape[0] == npoints * num_cells + (num_cells - 1)
  nan_positions = np.flatnonzero(np.isnan(values))
  # np.insert's k-th insertion index shifts by k in the resulting array
  # (each prior insertion pushes it one further along).
  pre_insertion = np.arange(npoints, npoints * num_cells, npoints)
  expected_positions = pre_insertion + np.arange(len(pre_insertion))
  np.testing.assert_array_equal(nan_positions, expected_positions)

  # The grid coordinate at a NaN repeats the preceding (cell-right-face)
  # coordinate, so plotting breaks the line without leaving a coordinate gap.
  for pos in nan_positions:
    assert lp.grid[0][pos] == pytest.approx(lp.grid[0][pos - 1])


def test_backend_and_flags_after_local_poly():
  lp = pg.load(F1D).local_poly()
  assert lp.backend == "numpy"
  assert lp.is_interpolated
  assert lp.ctx["interpolated"] is True


def test_default_npoints_is_two():
  d = pg.load(F1D)
  num_cells = int(d.num_cells[0])
  lp = d.local_poly()
  # 2 raw points/cell + one NaN at each of the (num_cells - 1) interior faces.
  assert lp.get_values().shape[0] == 2 * num_cells + (num_cells - 1)


def test_2d_and_3d_shapes():
  d2 = pg.load(F2D)
  lp2 = d2.local_poly(npoints=4)
  nx, ny = (int(c) for c in d2.num_cells)
  assert lp2.grid[0].shape == (4 * nx + (nx - 1), )
  assert lp2.grid[1].shape == (4 * ny + (ny - 1), )
  # `d2.num_comps` counts raw modal coefficients (fields * num_basis); this
  # fixture holds a single field.
  assert lp2.get_values().shape == (4 * nx + (nx - 1), 4 * ny + (ny - 1), 1)

  d3 = pg.load(F3D)
  lp3 = d3.local_poly()
  assert lp3.num_dims == 3
  assert not np.all(np.isnan(lp3.get_values()))


def test_missing_poly_order_raises():
  d = pg.load(F1D)
  del d.ctx["poly_order"]
  with pytest.raises(ValueError, match="poly_order"):
    d.local_poly()


def test_missing_basis_type_raises():
  d = pg.load(F1D)
  del d.ctx["basis_type"]
  with pytest.raises(ValueError, match="basis_type"):
    d.local_poly()


def test_rejects_non_modal_value_form():
  d = pg.load(F1D).to_nodal()
  with pytest.raises(ValueError, match="modal value_form"):
    d.local_poly()


def test_inplace_and_tag_label():
  d = pg.load(F1D)
  same = d.local_poly(inplace=True, tag="lp", label="lp-label")
  assert same is d
  assert d.tag == "lp"
  assert d.get_label() == "lp-label"

  d2 = pg.load(F1D)
  new = d2.local_poly(tag="lp2")
  assert new is not d2
  assert new.tag == "lp2"
