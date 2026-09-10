"""Explicit R-Z mapping and Gkeyll geometry compositions."""

from __future__ import annotations

from importlib import import_module
import os
from types import SimpleNamespace

import click
import numpy as np
import pytest
from click.testing import CliRunner

import postgkyl as pg
from postgkyl import gpython
from postgkyl.cli.app import COMMANDS

gk_rz_command = next(command for command in COMMANDS if command.name == "gk_rz")
from postgkyl.cli.state import DataSpace
from postgkyl.diagnostics.gk.geometry import resolve_geometry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
F1D = os.path.join(
    DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")
F2D = os.path.join(DATA, "gk_ltx_iwl_2x2v_p1-elc_M2par_10.gkyl")
F2D_GEO = os.path.join(DATA, "gk_ltx_iwl_2x2v_p1-geo_int_mapc2p.gkyl")
F3D = os.path.join(DATA, "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl")

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


def test_geometry_prefers_nodes_and_honors_explicit_modal_override(
    tmp_path, monkeypatch):
  from postgkyl.diagnostics.gk import geometry as geometry_module

  source = tmp_path / "sim-field_0.gkyl"
  nodes = tmp_path / "sim-geo_int_nodes.gkyl"
  modal = tmp_path / "sim-geo_int_mapc2p.gkyl"
  nodes.touch()
  modal.touch()
  coords = [np.array([0.0, 1.0]), np.array([-1.0, 1.0])]
  arrays = np.ones((2, 2))
  calls = []
  monkeypatch.setattr(
      geometry_module, "_read_nodes_geometry", lambda path: (calls.append(
          ("nodes", path)) or (coords, arrays, arrays, None)))
  monkeypatch.setattr(
      geometry_module, "_read_mapc2p_geometry", lambda path: (calls.append(
          ("mapc2p", path)) or (coords, arrays, arrays, None)))

  resolve_geometry(str(source))
  assert calls[-1] == ("nodes", str(nodes))
  resolve_geometry(str(source), mapc2p="")
  assert calls[-1] == ("mapc2p", str(modal))


@needs_gkeyll
def test_geometry_overrides_and_validation_errors(tmp_path):
  data = pg.load(F2D)
  with pytest.raises(ValueError, match="either mapc2p=.*nodes_file"):
    pg.gk.rz(data, mapc2p=F2D_GEO, nodes_file=F2D_GEO)
  explicit = pg.gk.rz(data, mapc2p=F2D_GEO, nz_interp=2)
  inferred = pg.gk.rz(data, nz_interp=2)
  np.testing.assert_allclose(explicit.values, inferred.values)

  missing = data.clone()
  missing._file_name = str(tmp_path / "absent-field_0.gkyl")
  with pytest.raises(ValueError, match="Could not find a geometry file"):
    pg.gk.rz(missing)
  with pytest.raises(ValueError, match="positive integer"):
    pg.gk.rz(data, nz_interp=0)
  with pytest.raises(ValueError, match="out of bounds"):
    pg.gk.rz(data, comp=1)
  with pytest.raises(ValueError, match="requires 2-D or 3-D"):
    pg.gk.rz(pg.load(F1D), mapc2p=F2D_GEO)
  with pytest.raises(ValueError, match="un-interpolated modal DG"):
    pg.gk.rz(data.interpolate())


def test_rz_projection_collection_caches_by_geometry_prefix(monkeypatch):
  rz = import_module("postgkyl.diagnostics.gk.rz")
  monkeypatch.setattr(rz, "validate_mapping_grid", lambda *_args: None)
  first = SimpleNamespace(file_name="block-one", ctx={"block": 1})
  repeated = SimpleNamespace(file_name="block-one", ctx={"block": 1})
  second = SimpleNamespace(file_name="block-two", ctx={"block": 2})
  calls = []
  monkeypatch.setattr(rz, "geometry_prefix", lambda path: path)
  monkeypatch.setattr(
      rz, "resolve_geometry", lambda path, **kwargs: calls.append(
          (path, kwargs)) or path)
  monkeypatch.setattr(
      rz, "resolve_rz_projection",
      lambda data, geo, **_kwargs: SimpleNamespace(computational_grid=(),
                                                   geometry=geo))

  projections = rz.rz_projections([first, repeated, second],
                                  mapc2p="map-*.gkyl",
                                  nodes_file="nodes-*.gkyl")
  assert projections == {
      "block-one": SimpleNamespace(computational_grid=(), geometry="block-one"),
      "block-two": SimpleNamespace(computational_grid=(), geometry="block-two"),
  }
  assert calls == [
      ("block-one", {
          "mapc2p": "map-1.gkyl",
          "nodes_file": "nodes-1.gkyl"
      }),
      ("block-two", {
          "mapc2p": "map-2.gkyl",
          "nodes_file": "nodes-2.gkyl"
      }),
  ]
  assert rz.projection_for(projections,
                           first) == SimpleNamespace(computational_grid=(),
                                                     geometry="block-one")


@needs_gkeyll
def test_group_mapping_cli_and_help_section():
  from postgkyl.cli.app import cli

  group = pg.GDataGroup([pg.load(F2D), pg.load(F2D)])
  projection = pg.resolve_rz_projection(group[0],
                                        resolve_geometry(F2D),
                                        nz_interp=2)
  mapped = group.map_to_rz(projection=projection)
  assert isinstance(mapped, pg.GDataGroup) and len(mapped) == 2

  space = DataSpace(datasets=[pg.load(F2D)])
  with click.Context(gk_rz_command, obj=space) as ctx:
    ctx.invoke(gk_rz_command,
               mapc2p=F2D_GEO,
               nodes_file=None,
               z_axis=0.0,
               phi_tor=0.0,
               nz_interp=2,
               use=None,
               tag="rz",
               label=None)
  expected = pg.gk.rz(pg.load(F2D), mapc2p=F2D_GEO, nz_interp=2, tag="rz")
  np.testing.assert_allclose(space.datasets[0].values, expected.values)

  help_text = CliRunner().invoke(cli, ["--help"]).output
  verbs = help_text.split("Diagnostics:", 1)[0]
  diagnostics = help_text.split("Diagnostics:", 1)[1].split("Render:", 1)[0]
  assert "gk_rz" not in verbs and "gk_rz" in diagnostics


@needs_gkeyll
@pytest.mark.parametrize("module_name, builder_name", [
    ("rz", "rz_projections"),
    ("fluxsurf", "flux_surface_grids"),
])
def test_cached_mapping_rejects_changed_grid_with_same_prefix(
    module_name, builder_name):
  module = import_module(f"postgkyl.diagnostics.gk.{module_name}")
  first = pg.load(F3D)
  changed = first.clone()
  changed.grid[0] = changed.grid[0] + 0.1
  with pytest.raises(ValueError, match="computational grid does not match"):
    getattr(module, builder_name)([first, changed], nz_interp=2)


@needs_gkeyll
def test_flux_surface_diagnostic_cli_matches_explicit_core():
  from postgkyl.cli.app import cli
  data = pg.load(F3D)
  geometry = pg.gk.resolve_geometry(data.file_name)
  grid = pg.resolve_flux_surface_grid(data, geometry, nphi=4, nz_interp=2)
  expected = data.extract_flux_surface(fs_grid=grid)
  actual = pg.gk.fluxsurf(data, nphi=4, nz_interp=2)
  np.testing.assert_allclose(actual.values, expected.values)
  command = next(c for c in COMMANDS if c.name == "gk_fluxsurf")
  space = DataSpace(datasets=[data])
  with click.Context(command, obj=space) as ctx:
    ctx.invoke(command, nphi=4, nz_interp=2, use=None, tag=None, label=None)
  np.testing.assert_allclose(space.datasets[0].values, expected.values)
  result = CliRunner().invoke(cli, ["--help"])
  assert result.exit_code == 0
  assert "gk_fluxsurf" in result.output.split("Diagnostics:", 1)[1]
  assert not hasattr(pg, "gk_fluxsurf")
  assert not hasattr(pg.GData, "gk_fluxsurf")
