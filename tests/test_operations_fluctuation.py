"""Tests for the ``fluctuation`` verb: ``f - <f>_dims`` on the field's own
grid and basis, through the API, the fluent alias, and the generated CLI.

The algebraic identities hold to round-off on arbitrary (random) DG data
because the lift of the average is exact:
  - a fluctuation averages to zero under the same (weighted) average,
  - a field constant along ``dims`` has no fluctuation.
The analytic case uses an exactly representable ``f(x, y) = a(x) + b(y)``,
whose fluctuation about any subset of directions is known in closed form.

Run:  PYTHONPATH=src pytest tests/test_operations_fluctuation.py -v
"""

from __future__ import annotations

from pathlib import Path

import click
from click.testing import CliRunner
import numpy as np
from numpy.polynomial import Polynomial
import pytest

import postgkyl as pg
from postgkyl import dg, gpython
from postgkyl.cli.app import cli, COMMANDS
from postgkyl.cli.state import DataSpace

pytestmark = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

ROOT = Path(__file__).parents[1]
FIELD_3D = ROOT / "tests" / "test_data" / "generated" / "3d_ms_p1.gkyl"

# Random coefficients and polynomial values are O(1); the average, lift, and
# subtraction are exact operations, so only double round-off remains.
ROUND_OFF = dict(rtol=1e-12, atol=1e-12)

_CELLS = (4, 6, 3)
_GRID = [
    np.linspace(0.0, 1.0, _CELLS[0] + 1),
    np.linspace(0.0, 2.0, _CELLS[1] + 1),
    np.linspace(-1.0, 1.0, _CELLS[2] + 1),
]
_DIRS = [[1], [1, 2], [0], [0, 1, 2]]


def _state(values, poly_order, value_form="modal", tag="default") -> pg.GData:
  data = pg.GData(tag=tag,
                  ctx={
                      "basis_type": "serendipity",
                      "poly_order": poly_order,
                      "value_form": value_form,
                      "cells": np.array(_CELLS),
                  })
  data.push(_GRID, gpython.GkylArray.from_numpy(values))
  return data


def _random(poly_order, nfields=2, seed=0) -> pg.GData:
  nb = gpython.basis.num_basis("serendipity", 3, poly_order)
  rng = np.random.default_rng(seed)
  return _state(rng.standard_normal((*_CELLS, nfields * nb)), poly_order)


def _jacobian(poly_order) -> pg.GData:
  """A positive weight independent of y, like a field-aligned Jacobian
  J(x, z): coefficient 0 (the constant mode) in 2..3 plus a small
  coefficient on mode 1 (the x mode), identical in every y cell."""
  rng = np.random.default_rng(7)
  nb = gpython.basis.num_basis("serendipity", 3, poly_order)
  values = np.zeros((*_CELLS, nb))
  values[..., 0] = 2.0 + rng.random((_CELLS[0], 1, _CELLS[2]))
  values[..., 1] = 0.1 * rng.random((_CELLS[0], 1, _CELLS[2]))
  return _state(values, poly_order, tag="w")


def _average_coefficients(data, dims, weight=None):
  """``data``'s average over ``dims`` as an array: the modal coefficients of
  a partial average (all zero exactly when the averaged field is zero), or
  the physical mean(s) returned by a full average."""
  mean = data.average(dims, weight=weight)
  return mean.values if isinstance(mean, pg.GData) else np.asarray(mean)


def _constant_along(dirs, poly_order, seed) -> pg.GData:
  """A field independent of ``dirs``: a random field averaged over ``dirs``
  and lifted back (``dg.modal.lift``)."""
  field = _random(poly_order, seed=seed)
  grid = {
      "ndim": 3,
      "lower": np.array([g[0] for g in _GRID]),
      "upper": np.array([g[-1] for g in _GRID]),
      "cells": np.array(_CELLS),
  }
  _keep, _cells, mean = dg.modal.average(grid, "serendipity", 3, poly_order,
                                         field.native, dirs)
  lifted = dg.modal.lift("serendipity", 3, poly_order, mean, dirs, _CELLS)
  return _state(lifted.view(_CELLS), poly_order)


# ------------------------------------------------------- algebraic identities
@pytest.mark.parametrize("poly_order", [1, 2])
@pytest.mark.parametrize("dirs", _DIRS)
@pytest.mark.parametrize("weighted", [False, True])
def test_fluctuation_averages_to_zero(poly_order, dirs, weighted):
  field = _random(poly_order, seed=1)
  weight = _jacobian(poly_order) if weighted else None

  fluct = field.fluctuation(dirs, weight=weight)

  assert fluct.values.shape == field.values.shape
  np.testing.assert_allclose(_average_coefficients(fluct, dirs, weight), 0.0,
                             **ROUND_OFF)
  assert np.abs(fluct.values).max() > 0.1


@pytest.mark.parametrize("poly_order", [1, 2])
@pytest.mark.parametrize("dirs", _DIRS)
def test_field_constant_along_dirs_has_no_fluctuation(poly_order, dirs):
  constant = _constant_along(dirs, poly_order, seed=2)
  np.testing.assert_allclose(
      constant.fluctuation(dirs).values, 0.0, **ROUND_OFF)


@pytest.mark.parametrize("poly_order", [1, 2])
@pytest.mark.parametrize("dirs", _DIRS)
@pytest.mark.parametrize("weighted", [False, True])
def test_uniform_field_has_no_fluctuation(poly_order, dirs, weighted):
  """f = 5 everywhere (coefficient 0 = 5 * 2**(3/2), nothing else) is its own
  average under any weight, so its fluctuation vanishes identically."""
  nb = gpython.basis.num_basis("serendipity", 3, poly_order)
  values = np.zeros((*_CELLS, nb))
  values[..., 0] = 5.0 * 2.0**1.5
  weight = _jacobian(poly_order) if weighted else None
  fluct = _state(values, poly_order).fluctuation(dirs, weight=weight)
  np.testing.assert_allclose(fluct.values, 0.0, **ROUND_OFF)


# -------------------------------------------------------------- analytic case
def _polynomials(poly_order):
  """a(x), b(y) of degree ``poly_order``: exactly representable per cell."""
  a = Polynomial([1.0, 2.0, 0.5][:poly_order + 1])
  b = Polynomial([0.0, 3.0, -1.0][:poly_order + 1])
  return a, b


def _mean(polynomial, edges):
  primitive = polynomial.integ()
  return (primitive(edges[-1]) - primitive(edges[0])) / (edges[-1] - edges[0])


def _nodal(poly_order, function) -> pg.GData:
  """``function(x, y, z)`` sampled at Gkeyll's nodes of every cell, as a
  nodal dataset; ``represent(to='modal')`` is exact for polynomials in the
  basis."""
  nodes = gpython.basis.node_coords("serendipity", 3, poly_order)
  centers = [(g[:-1] + g[1:]) / 2 for g in _GRID]
  widths = [np.diff(g) for g in _GRID]
  values = np.empty((*_CELLS, nodes.shape[0]))
  for idx in np.ndindex(*_CELLS):
    coords = [
        centers[d][idx[d]] + widths[d][idx[d]] / 2 * nodes[:, d]
        for d in range(3)
    ]
    values[idx] = function(*coords)
  return _state(values, poly_order, value_form="nodal").represent(to="modal")


@pytest.mark.parametrize("poly_order", [1, 2])
@pytest.mark.parametrize("dims", [[1], [0], [2], [0, 1], [0, 1, 2]])
def test_fluctuation_of_a_separable_sum_matches_the_closed_form(
    poly_order, dims):
  """f = a(x) + b(y): averaging over a direction replaces that direction's
  term by its mean, so the fluctuation is ``(a - mean a)`` if 0 is in
  ``dims`` plus ``(b - mean b)`` if 1 is in ``dims`` -- and zero for
  ``dims == [2]``, since f does not depend on z."""
  a, b = _polynomials(poly_order)
  field = _nodal(poly_order, lambda x, y, z: a(x) + b(y))

  fluct = field.fluctuation(dims)

  num_interp = 3
  points = fluct.interpolate(num_interp=num_interp)
  axes = [
      np.linspace(g[0], g[-1], (len(g) - 1) * num_interp + 1) for g in _GRID
  ]
  x, y, _z = np.meshgrid(*[(ax[:-1] + ax[1:]) / 2 for ax in axes],
                         indexing="ij")
  expected = np.zeros_like(x)
  if 0 in dims:
    expected += a(x) - _mean(a, _GRID[0])
  if 1 in dims:
    expected += b(y) - _mean(b, _GRID[1])
  np.testing.assert_allclose(points.values[..., 0], expected, **ROUND_OFF)


@pytest.mark.parametrize("poly_order", [1, 2])
def test_y_independent_weight_leaves_the_y_fluctuation_unchanged(poly_order):
  """With w = w(x, z) the weighted y-average of a(x) + b(y) is still
  a(x) + mean b, so the weighted fluctuation about y is b(y) - mean b."""
  a, b = _polynomials(poly_order)
  field = _nodal(poly_order, lambda x, y, z: a(x) + b(y))
  weight = _nodal(poly_order, lambda x, y, z: 2.0 + 0.5 * x * z)

  points = field.fluctuation([1], weight=weight).interpolate(num_interp=2)

  y_axis = np.linspace(_GRID[1][0], _GRID[1][-1], 2 * _CELLS[1] + 1)
  y = ((y_axis[:-1] + y_axis[1:]) / 2)[None, :, None]
  expected = np.broadcast_to(
      b(y) - _mean(b, _GRID[1]), points.values.shape[:-1])
  np.testing.assert_allclose(points.values[..., 0], expected, **ROUND_OFF)


# --------------------------------------------------------- dataset contract
def test_result_keeps_backend_value_form_grid_and_cells():
  field = _random(1, seed=5)
  before = field.values.copy()

  out = field.fluctuation([1])

  assert out is not field
  assert out.backend == "gkyl"
  assert out.ctx["value_form"] == "modal"
  assert out.ctx["basis_type"] == "serendipity"
  assert out.ctx["poly_order"] == 1
  assert out.ctx["cells"].tolist() == list(_CELLS)
  assert out.num_comps == field.num_comps
  for actual, expected in zip(out.grid, _GRID):
    np.testing.assert_array_equal(actual, expected)
  np.testing.assert_array_equal(field.values, before)


def test_tag_label_and_inplace_behave_like_other_verbs():
  field = _random(1, seed=6)
  out = field.fluctuation([1], tag="fluct", label="my label")
  assert out.tag == "fluct"
  assert out.label == "my label"
  assert field.tag == "default"

  mutated = field.fluctuation([1], inplace=True)
  assert mutated is field
  np.testing.assert_allclose(_average_coefficients(field, [1]), 0.0,
                             **ROUND_OFF)


def test_fluent_functional_and_facade_spellings_are_one_object():
  assert pg.GData.fluctuation is pg.operations.fluctuation
  assert pg.fluctuation is pg.operations.fluctuation


# ---------------------------------------------------------------- rejections
def test_rejects_numpy_backed_and_non_modal_inputs():
  field = _random(1, seed=8)
  with pytest.raises(ValueError, match="native modal data"):
    field.interpolate().fluctuation([1])
  with pytest.raises(ValueError, match="modal value_form"):
    field.represent(to="nodal").fluctuation([1])


def test_rejects_empty_and_out_of_range_dims():
  field = _random(1, seed=9)
  with pytest.raises(ValueError, match="out of range"):
    field.fluctuation([])
  with pytest.raises(ValueError, match="out of range"):
    field.fluctuation([3])


def test_rejects_a_mismatched_weight():
  field = _random(1, seed=10)
  with pytest.raises(ValueError, match="poly_order"):
    field.fluctuation([1], weight=_jacobian(2))
  with pytest.raises(ValueError, match="native modal data"):
    field.fluctuation([1], weight=_jacobian(1).interpolate())


# ---------------------------------------------------------------------- CLI
def _run(*args):
  return CliRunner().invoke(cli, [str(arg) for arg in args])


def test_cli_fluctuation_runs_on_a_file():
  result = _run(FIELD_3D, "fluctuation", "--dims", "1", "info")
  assert result.exit_code == 0, result.output
  assert "Number of dimensions: 3" in result.output
  assert "serendipity p1 (modal)" in result.output


def test_cli_weight_resolves_a_tag(tmp_path):
  field_path = pg.save(_random(1, nfields=1, seed=12), str(tmp_path / "f.gkyl"))
  weight_path = pg.save(_jacobian(1), str(tmp_path / "jacobian.gkyl"))
  result = _run(weight_path, "--tag", "w", field_path, "--tag", "f",
                "fluctuation", "--dims", "1", "--weight", "w", "info")
  assert result.exit_code == 0, result.output
  assert "(f#" in result.output

  missing = _run(field_path, "fluctuation", "--dims", "1", "--weight", "nope")
  assert missing.exit_code != 0
  assert "no dataset tagged 'nope'" in missing.output


def test_cli_command_is_the_canonical_verb():
  """Through the generic adapter, ``--weight w`` resolves the tagged
  dataset and the result replaces the input in the working set; it is the
  same computation as the Python verb and averages to zero."""
  command = next(c for c in COMMANDS if c.name == "fluctuation")
  field, weight = _random(1, nfields=1, seed=11), _jacobian(1)
  space = DataSpace(datasets=[field, weight])
  with click.Context(command, obj=space) as context:
    context.invoke(command,
                   dims=[[1]],
                   weight="w",
                   inplace=False,
                   tag="out",
                   label=None)

  out = space.datasets[0]
  assert out.tag == "out"
  np.testing.assert_allclose(out.values,
                             pg.fluctuation(field, [1], weight=weight).values,
                             **ROUND_OFF)
  np.testing.assert_allclose(_average_coefficients(out, [1], weight), 0.0,
                             **ROUND_OFF)
