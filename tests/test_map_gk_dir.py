"""Normalized poloidal-flux coordinate mapping."""

import importlib
import numpy as np
import pytest

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.operations import map_gk_dir

map_module = importlib.import_module("postgkyl.operations.map_gk_dir")


def test_rho_pol_maps_sample_positions_and_preserves_values(monkeypatch):
  values = np.arange(12.0).reshape(3, 4, 1)
  data = GDataState(ctx={"interpolated": True, "geqdsk_file": "eq.geqdsk"})
  data.push([np.array([2., 3., 6., 11.]), np.arange(5.)], values)
  monkeypatch.setattr(map_module, "_flux_reference", lambda *args: (1., 17.))

  result = map_gk_dir(data, dir=0, coord="rho_pol")

  np.testing.assert_allclose(result.grid[0],
                             np.sqrt((np.array([2.5, 4.5, 8.5]) - 1.) / 16.))
  np.testing.assert_array_equal(result.grid[1], data.grid[1])
  np.testing.assert_array_equal(result.values, values)
  np.testing.assert_array_equal(data.grid[0], [2., 3., 6., 11.])
  assert result.ctx["coordinate_labels"][0] == r"$\rho_{\mathrm{pol}}$"


def test_flux_reference_uses_equilibrium_basename_and_lcfs(monkeypatch):
  seen = []

  def fake_read(path, ctx):
    seen.append(path)
    ctx.update(geometry_type=1,
               tokamak_geo_type=20,
               simag=5.,
               psi_lcfs=1.,
               sibry=1.,
               psi_sep=0.)

  monkeypatch.setattr(map_module, "read", fake_read)
  assert map_module._flux_reference("run/field.gkyl", "other/eq.geqdsk",
                                    None) == (5., 1.)
  assert seen == ["run/eq-psi.gkyl"]
  assert map_module._flux_reference("", None, "elsewhere/psi.gkyl") == (5., 1.)
  assert seen[-1] == "elsewhere/psi.gkyl"


def test_flux_reference_uses_separatrix_for_diverted_tokamak(monkeypatch):

  def fake_read(path, ctx):
    ctx.update(geometry_type=1,
               tokamak_geo_type=2,
               simag=0.,
               psi_sep=4.,
               sibry=3.)

  monkeypatch.setattr(map_module, "read", fake_read)
  assert map_module._flux_reference("", None, "psi.gkyl") == (0., 4.)


def test_flux_reference_requires_explicit_tokamak_subtype(monkeypatch):

  def fake_read(path, ctx):
    ctx.update(geometry_type=1, simag=0., psi_sep=4., sibry=3.)

  monkeypatch.setattr(map_module, "read", fake_read)
  with pytest.raises(ValueError, match="tokamak_geo_type"):
    map_module._flux_reference("", None, "old-psi.gkyl")


def test_invalid_direction_coord_and_domain(monkeypatch):
  data = GDataState(ctx={"interpolated": True})
  data.push([np.array([0., 1.])], np.ones((1, 1)))
  monkeypatch.setattr(map_module, "_flux_reference", lambda *args: (2., 3.))
  with pytest.raises(ValueError, match="supports only"):
    map_gk_dir(data, dir=1, coord="rho_pol")
  with pytest.raises(ValueError, match="undefined"):
    map_gk_dir(data, dir=0, coord="rho_pol")
