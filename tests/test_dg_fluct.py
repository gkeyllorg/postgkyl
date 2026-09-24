"""Postgkyl module for testing DG fluctuations: GkeyllDGops.expand/fluctuation
and the dg-fluct command.

A fluctuation dA = A - <A> is built by averaging A over some directions with
gkyl_array_average and lifting the result back onto the full basis. The checks
below are exact identities of that construction, so they hold to round-off on
arbitrary (random) DG data:
  - averaging a lifted field gives the field back,
  - a field that does not depend on the averaged directions has no fluctuation,
  - a fluctuation averages to zero, with or without a weight.
"""
import click
import numpy as np
import pytest

import postgkyl.commands as cmd
from postgkyl.data import GData
from postgkyl.pgkyl import cli

try:
  from postgkyl.tools.gkeyll_dg_ops import GkeyllDGops
  GkeyllDGops()
  _DGOPS_AVAILABLE = True
except Exception:  # noqa: BLE001 - any failure means the lib is unusable here
  _DGOPS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _DGOPS_AVAILABLE, reason="requires the gkylsoft DG library")

_NUM_BASIS = {1: 8, 2: 20}  # 3D serendipity.
_CELLS = (4, 6, 3)
_GRID = [np.linspace(0.0, 1.0, _CELLS[0] + 1), np.linspace(0.0, 2.0, _CELLS[1] + 1),
         np.linspace(-1.0, 1.0, _CELLS[2] + 1)]
_DIRS = [[1], [1, 2], [0], [0, 1, 2]]


def _gdata(values, poly_order) -> GData:
  gdata = GData(ctx={"poly_order": poly_order, "basis_type": "serendipity"})
  gdata.push(_GRID, values)
  return gdata

def _random(poly_order, num_comps=2, seed=0) -> GData:
  rng = np.random.default_rng(seed)
  return _gdata(rng.standard_normal((*_CELLS, num_comps*_NUM_BASIS[poly_order])), poly_order)

def _jacobian(poly_order) -> GData:
  """A positive weight independent of y, like a field-aligned Jacobian J(x,z)."""
  rng = np.random.default_rng(7)
  values = np.zeros((*_CELLS, _NUM_BASIS[poly_order]))
  values[..., 0] = 2.0 + rng.random((_CELLS[0], 1, _CELLS[2]))
  values[..., 1] = 0.1*rng.random((_CELLS[0], 1, _CELLS[2]))
  return _gdata(values, poly_order)


@pytest.mark.parametrize("poly_order", [1, 2])
@pytest.mark.parametrize("dirs", _DIRS)
class TestFluctuation:

  @pytest.mark.parametrize("weighted", [False, True])
  def test_average_of_expand_round_trips(self, poly_order, dirs, weighted):
    ops = GkeyllDGops()
    field = _random(poly_order)
    weight = _jacobian(poly_order) if weighted else None
    mean = ops.average(dirs, field, weight=weight)
    again = ops.average(dirs, ops.expand(mean, field, dirs), weight=weight)
    assert np.allclose(again.get_values(), mean.get_values(), atol=1e-13)

  @pytest.mark.parametrize("weighted", [False, True])
  def test_fluctuation_averages_to_zero(self, poly_order, dirs, weighted):
    ops = GkeyllDGops()
    field = _random(poly_order, seed=1)
    weight = _jacobian(poly_order) if weighted else None
    fluct = ops.fluctuation(dirs, field, weight=weight)
    assert fluct.get_values().shape == field.get_values().shape
    assert np.allclose(ops.average(dirs, fluct, weight=weight).get_values(), 0.0, atol=1e-13)
    assert np.abs(fluct.get_values()).max() > 0.1

  def test_field_constant_along_dirs_has_no_fluctuation(self, poly_order, dirs):
    ops = GkeyllDGops()
    field = _random(poly_order, seed=2)
    lifted = ops.expand(ops.average(dirs, field), field, dirs)
    assert np.allclose(ops.fluctuation(dirs, lifted).get_values(), 0.0, atol=1e-13)


def test_full_average_is_a_1d_basis_coefficient():
  """With or without a weight, a full average of 5 is the 1D coefficient 5*sqrt(2)."""
  ops = GkeyllDGops()
  values = np.zeros((*_CELLS, _NUM_BASIS[1]))
  values[..., 0] = 5.0*2.0**1.5
  field = _gdata(values, 1)
  unweighted = ops.average([0, 1, 2], field).get_values()
  weighted = ops.average([0, 1, 2], field, weight=_jacobian(1)).get_values()
  assert np.allclose(unweighted, [[5.0*np.sqrt(2.0), 0.0]])
  assert np.allclose(weighted, unweighted)


def test_dg_fluct_command():
  ctx = click.core.Context(cli)
  ctx.obj = {"data": cmd.DataSpace(), "verbose": False, "compgrid": None}
  field = _random(1, num_comps=1, seed=3)
  field.set_tag("den")
  ctx.obj["data"].add(field)

  ctx.invoke(cmd.dg_fluct, z1=True, weight="none")

  out = list(ctx.obj["data"].iterator())
  assert len(out) == 1
  assert out[0].get_tag() == "den"
  assert out[0].get_num_dims() == 3
  expected = GkeyllDGops().fluctuation([1], field)
  assert np.allclose(out[0].get_values(), expected.get_values())
  assert not field.get_status()


def test_dg_fluct_needs_a_direction():
  ctx = click.core.Context(cli)
  ctx.obj = {"data": cmd.DataSpace(), "verbose": False, "compgrid": None}
  with pytest.raises(click.exceptions.UsageError):
    ctx.invoke(cmd.dg_fluct)


def test_full_average_collects_into_a_time_trace():
  """dg-avg over every direction, interpolate, collect gives a 1D time trace.

  A full average is stored on a dummy single cell; interpolate must keep a
  single value there, or collect builds a 2D (time x dummy) dataset.
  """
  ctx = click.core.Context(cli)
  ctx.obj = {"data": cmd.DataSpace(), "verbose": False, "compgrid": None}
  for frame in range(3):
    values = np.zeros((*_CELLS, _NUM_BASIS[1]))
    values[..., 0] = np.sqrt(2.0)**3*(frame + 1)  # f = frame + 1
    field = _gdata(values, 1)
    field.ctx.update({"time": 0.1*frame, "frame": frame, "is_modal": True,
                      "grid_type": "uniform"})
    ctx.obj["data"].add(field)

  ctx.invoke(cmd.dg_avg, z0=True, z1=True, z2=True, weight="none")
  ctx.invoke(cmd.interpolate, interp=2)
  ctx.invoke(cmd.collect)

  out = list(ctx.obj["data"].iterator())
  assert len(out) == 1
  assert out[0].get_num_dims(squeeze=True) == 1
  assert np.allclose(out[0].get_grid()[0], [0.0, 0.1, 0.2])
  assert np.allclose(out[0].get_values().ravel(), [1.0, 2.0, 3.0])
