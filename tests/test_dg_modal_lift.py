"""Correctness tests for ``dg.modal.lift`` -- the operator that maps a field
over the kept directions (the output of ``dg.modal.average``) back onto the
full basis and cell layout, constant along the averaged directions.

Closed-form references use the normalization of Gkeyll's orthonormal
serendipity basis: on the reference cell the constant mode is
``b_0 = 2**(-ndim/2)`` and the linear modes are ``sqrt(3) * 2**(-ndim/2) z_d``,
ordered ``1, z_0, z_1, ...`` (the same convention the differentiate and
eval_at_coord_proj tests rely on). Lifting a function from ``ndim_red`` to
``ndim`` dimensions therefore multiplies its coefficients by
``2**((ndim - ndim_red)/2)`` and places them on the modes that depend only on
the kept coordinates.

Run:  PYTHONPATH=src pytest tests/test_dg_modal_lift.py -v
"""

import numpy as np
import pytest

from postgkyl import dg, gpython

pytestmark = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

# Coefficients are O(1); products of two orthonormal basis functions are
# integrated exactly, so only double round-off remains.
ROUND_OFF = dict(rtol=1e-13, atol=1e-13)

_CELLS = (4, 6, 3)
_GRID = {
    "ndim": 3,
    "lower": np.array([0.0, 0.0, -1.0]),
    "upper": np.array([1.0, 2.0, 1.0]),
    "cells": np.array(_CELLS),
}
_DIRS = [[1], [1, 2], [0], [0, 1, 2]]


def _random_coefficients(cells, ncomp, seed):
  return np.random.default_rng(seed).standard_normal((*cells, ncomp))


def _y_independent_weight(poly_order) -> gpython.GkylArray:
  """A positive 3-D weight w(x, z): coefficient 0 (constant) in 2..3 and a
  small coefficient on mode 1 (the x mode), identical for every y cell."""
  rng = np.random.default_rng(7)
  nb = gpython.basis.num_basis("serendipity", 3, poly_order)
  values = np.zeros((*_CELLS, nb))
  values[..., 0] = 2.0 + rng.random((_CELLS[0], 1, _CELLS[2]))
  values[..., 1] = 0.1 * rng.random((_CELLS[0], 1, _CELLS[2]))
  return gpython.GkylArray.from_numpy(values)


# ------------------------------------------------------- closed-form lifts
@pytest.mark.parametrize("avg_dir, lifted_mode", [(1, 1), (0, 2)])
def test_lift_1d_p1_into_2d_places_the_linear_mode(avg_dir, lifted_mode):
  """g(z) = c0 b0' + c1 b1' on the 1-D p1 basis lifts to C0 = sqrt(2) c0 on
  the 2-D constant and C = sqrt(2) c1 on the 2-D mode linear in the kept
  coordinate (mode 1 for x, mode 2 for y); every other mode is zero and
  every cell along the averaged direction repeats the kept cell's values."""
  cells = [3, 2]
  keep = 1 - avg_dir
  reduced = _random_coefficients([cells[keep]], 2, seed=0)

  out = dg.modal.lift("serendipity", 2, 1,
                      gpython.GkylArray.from_numpy(reduced), [avg_dir], cells)

  assert (out.size, out.ncomp) == (6, 4)
  expected = np.zeros((cells[keep], 4))
  expected[:, 0] = np.sqrt(2.0) * reduced[:, 0]
  expected[:, lifted_mode] = np.sqrt(2.0) * reduced[:, 1]
  expected = np.expand_dims(expected, axis=avg_dir)
  np.testing.assert_allclose(out.view(cells),
                             np.broadcast_to(expected, (*cells, 4)),
                             **ROUND_OFF)


@pytest.mark.parametrize("poly_order", [1, 2])
def test_lift_of_a_full_average_is_a_constant_field(poly_order):
  """A full average of two fields with means 5 and -2 is stored on a 1-cell
  1-D basis as coefficient 0 = mean * sqrt(2) (and nothing else). Lifted
  to 3-D the constant mode is mean * 2**(3/2) in every cell."""
  nb_1d = poly_order + 1
  reduced = np.zeros((1, 2 * nb_1d))
  reduced[0, 0] = 5.0 * np.sqrt(2.0)
  reduced[0, nb_1d] = -2.0 * np.sqrt(2.0)

  out = dg.modal.lift("serendipity", 3, poly_order,
                      gpython.GkylArray.from_numpy(reduced), [0, 1, 2], _CELLS)

  nb = gpython.basis.num_basis("serendipity", 3, poly_order)
  assert (out.size, out.ncomp) == (int(np.prod(_CELLS)), 2 * nb)
  expected = np.zeros((*_CELLS, 2 * nb))
  expected[..., 0] = 5.0 * 2.0**1.5
  expected[..., nb] = -2.0 * 2.0**1.5
  np.testing.assert_allclose(out.view(_CELLS), expected, **ROUND_OFF)


# ------------------------------------------------ the defining point identity
@pytest.mark.parametrize("poly_order", [1, 2])
@pytest.mark.parametrize("dirs", [[1], [1, 2], [0], [0, 2]])
def test_lifted_field_takes_the_reduced_values_at_every_point(poly_order, dirs):
  """The lift is exact: at any point of any cell, the lifted field equals
  the reduced field at the kept coordinates of that point. Both sides are
  evaluated with Gkeyll's own basis (``eval_matrix``), independent of the
  projection that builds the lift."""
  keep = [d for d in range(3) if d not in dirs]
  cells_red = [_CELLS[d] for d in keep]
  nb_red = gpython.basis.num_basis("serendipity", len(keep), poly_order)
  reduced = _random_coefficients(cells_red, 2 * nb_red, seed=3)

  out = dg.modal.lift("serendipity", 3, poly_order,
                      gpython.GkylArray.from_numpy(reduced), dirs, _CELLS)

  pts = np.random.default_rng(4).uniform(-1.0, 1.0, size=(7, 3))
  full = gpython.basis.eval_matrix("serendipity", 3, poly_order, pts)
  red = gpython.basis.eval_matrix("serendipity", len(keep), poly_order,
                                  pts[:, keep])
  nb = full.shape[1]
  lifted = out.view(_CELLS).reshape(*_CELLS, 2, nb)
  for idx in np.ndindex(*_CELLS):
    idx_red = tuple(idx[d] for d in keep)
    np.testing.assert_allclose(lifted[idx] @ full.T,
                               reduced[idx_red].reshape(2, nb_red) @ red.T,
                               **ROUND_OFF)


# ------------------------------------------------------- average round trip
@pytest.mark.parametrize("poly_order", [1, 2])
@pytest.mark.parametrize("dirs", _DIRS)
@pytest.mark.parametrize("weighted", [False, True])
def test_average_of_lift_round_trips(poly_order, dirs, weighted):
  """``average(lift(average(f))) == average(f)`` -- the lifted field is
  constant along ``dirs``, so (weighted or not) averaging it over ``dirs``
  returns exactly the field that was lifted."""
  nb = gpython.basis.num_basis("serendipity", 3, poly_order)
  field = gpython.GkylArray.from_numpy(
      _random_coefficients(_CELLS, 2 * nb, seed=1))
  weight = _y_independent_weight(poly_order) if weighted else None

  def average(a):
    return dg.modal.average(_GRID,
                            "serendipity",
                            3,
                            poly_order,
                            a,
                            dirs,
                            weight=weight)

  _keep, _cells, mean = average(field)
  lifted = dg.modal.lift("serendipity", 3, poly_order, mean, dirs, _CELLS)
  _keep, _cells, again = average(lifted)

  np.testing.assert_allclose(again.view(), mean.view(), **ROUND_OFF)


# ------------------------------------------------------------- error paths
def test_lift_rejects_empty_or_out_of_range_dirs():
  reduced = gpython.GkylArray.from_numpy(np.ones((4, 2)))
  with pytest.raises(ValueError, match="out of range"):
    dg.modal.lift("serendipity", 2, 1, reduced, [], [4, 3])
  with pytest.raises(ValueError, match="out of range"):
    dg.modal.lift("serendipity", 2, 1, reduced, [2], [4, 3])


def test_lift_rejects_a_component_count_that_is_not_whole_fields():
  reduced = gpython.GkylArray.from_numpy(np.ones((4, 3)))
  with pytest.raises(ValueError, match="not a multiple"):
    dg.modal.lift("serendipity", 2, 1, reduced, [1], [4, 3])
