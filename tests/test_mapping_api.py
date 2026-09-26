"""Explicit R-Z mapping and Gkeyll geometry compositions."""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path

import click
import numpy as np
import pytest
from click.testing import CliRunner

import postgkyl as pg
from postgkyl import gpython
from postgkyl.cli.app import COMMANDS

rz_command = next(command for command in COMMANDS
                  if command.name == "map_to_rz")
from postgkyl.cli.state import DataSpace
from postgkyl.operations.geometry import resolve_geometry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
F1D = os.path.join(
    DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")
F2D = os.path.join(DATA, "gk_ltx_iwl_2x2v_p1-elc_M2par_10.gkyl")
F2D_GEO = os.path.join(DATA, "gk_ltx_iwl_2x2v_p1-geo_int_mapc2p.gkyl")
F3D = os.path.join(DATA, "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl")
GENERATED = Path(DATA) / "generated"

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


def test_geometry_prefers_nodes_and_honors_explicit_modal_override(
    tmp_path, monkeypatch):
  from postgkyl.operations import geometry as geometry_module

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
    pg.map_to_rz(data, mapc2p=F2D_GEO, nodes_file=F2D_GEO)
  explicit = pg.map_to_rz(data, mapc2p=F2D_GEO, nz_interp=2)
  inferred = pg.map_to_rz(data, nz_interp=2)
  np.testing.assert_allclose(explicit.values, inferred.values)

  missing = data.clone()
  missing._file_name = str(tmp_path / "absent-field_0.gkyl")
  with pytest.raises(ValueError, match="Could not find a geometry file"):
    pg.map_to_rz(missing)
  with pytest.raises(ValueError, match="positive integer"):
    pg.map_to_rz(data, nz_interp=0)
  with pytest.raises(ValueError, match="out of bounds"):
    pg.map_to_rz(data, comp=1)
  with pytest.raises(ValueError, match="requires 2-D or 3-D"):
    pg.map_to_rz(pg.load(F1D), mapc2p=F2D_GEO)
  with pytest.raises(ValueError, match="un-interpolated modal DG"):
    pg.map_to_rz(data.interpolate())


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
  with click.Context(rz_command, obj=space) as ctx:
    ctx.invoke(rz_command,
               mapc2p=F2D_GEO,
               nodes_file=None,
               z_axis=0.0,
               phi_tor=0.0,
               nz_interp=2,
               use=None,
               tag="rz",
               label=None)
  expected = pg.map_to_rz(pg.load(F2D), mapc2p=F2D_GEO, nz_interp=2, tag="rz")
  np.testing.assert_allclose(space.datasets[0].values, expected.values)

  help_text = CliRunner().invoke(cli, ["--help"]).output
  verbs = help_text.split("Diagnostics:", 1)[0]
  diagnostics = help_text.split("Diagnostics:", 1)[1].split("Render:", 1)[0]
  assert "map_to_rz" in verbs and "map_to_rz" not in diagnostics
  assert "gk_rz" not in help_text


@needs_gkeyll
@pytest.mark.parametrize("module_name, builder_name", [
    ("map_to_rz", "rz_projections"),
    ("extract_flux_surface", "flux_surface_grids"),
])
def test_cached_mapping_rejects_changed_grid_with_same_prefix(
    module_name, builder_name):
  module = import_module(f"postgkyl.operations.{module_name}")
  first = pg.load(F3D)
  changed = first.clone()
  changed.grid[0] = changed.grid[0] + 0.1
  with pytest.raises(ValueError, match="computational grid does not match"):
    getattr(module, builder_name)([first, changed], nz_interp=2)


@needs_gkeyll
def test_flux_surface_cli_matches_explicit_projection():
  from postgkyl.cli.app import cli
  data = pg.load(F3D)
  geometry = pg.resolve_geometry(data.file_name)
  grid = pg.resolve_flux_surface_grid(data, geometry, nphi=4, nz_interp=2)
  expected = data.extract_flux_surface(fs_grid=grid)
  actual = pg.extract_flux_surface(data, nphi=4, nz_interp=2)
  np.testing.assert_allclose(actual.values, expected.values)
  command = next(c for c in COMMANDS if c.name == "extract_flux_surface")
  space = DataSpace(datasets=[data])
  with click.Context(command, obj=space) as ctx:
    ctx.invoke(command, nphi=4, nz_interp=2, use=None, tag=None, label=None)
  np.testing.assert_allclose(space.datasets[0].values, expected.values)
  result = CliRunner().invoke(cli, ["--help"])
  assert result.exit_code == 0
  assert "extract_flux_surface" in result.output.split("Diagnostics:", 1)[0]
  assert "gk_fluxsurf" not in result.output
  assert not hasattr(pg, "gk_fluxsurf")
  assert not hasattr(pg.GData, "gk_fluxsurf")


@needs_gkeyll
@pytest.mark.parametrize("ndim", [2, 3])
def test_fluid_output_uses_shared_geometry_in_api_and_cli(ndim):
  source = GENERATED / f"mapped_{ndim}d_b1-fluid_0.gkyl"
  data = pg.load(str(source))
  result = data.map_to_rz(nz_interp=2, z_axis=0.5)
  assert result.get_grid_type() == "mapped"
  assert result.ctx["mapped_axes"] == {0: 0, 1: 0}
  assert result.values.shape == (4, 8 if ndim == 3 else 4, 1)
  np.testing.assert_allclose(result.values, 8.0)
  np.testing.assert_allclose(result.grid[0][:, 0], np.linspace(3.0, 4.0, 5))
  np.testing.assert_allclose(result.grid[1][0],
                             np.linspace(0.5, 1.5, result.values.shape[1] + 1))
  np.testing.assert_allclose(result.differentiate().values, 0.0, atol=1e-12)
  np.testing.assert_allclose(result.integrate("0,1"), 8.0)

  space = DataSpace(datasets=[data])
  command = next(c for c in COMMANDS if c.name == "map_to_rz")
  cli_result = CliRunner().invoke(command,
                                  ["--nz_interp", "2", "--z_axis", "0.5"],
                                  obj=space)
  assert cli_result.exit_code == 0, cli_result.output
  np.testing.assert_allclose(space.datasets[0].values, result.values)
  for actual, expected in zip(space.datasets[0].grid, result.grid):
    np.testing.assert_allclose(actual, expected)


@needs_gkeyll
def test_geometry_node_and_modal_files_agree_on_fluid_coordinates():
  source = str(GENERATED / "mapped_2d_b1-fluid_0.gkyl")
  nodes = pg.load(source).map_to_rz()
  modal = pg.load(source).map_to_rz(mapc2p="")
  explicit = pg.load(source).map_to_rz(
      nodes_file=str(GENERATED / "mapped_2d_b*-geo_int_nodes.gkyl"))
  np.testing.assert_allclose(modal.values, nodes.values)
  for expected, actual, override in zip(nodes.grid, modal.grid, explicit.grid):
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(override, expected, atol=1e-14)


@needs_gkeyll
@pytest.mark.parametrize(
    "module_name, builder_name, lookup_name, verb, keyword", [
        ("map_to_rz", "rz_projections", "projection_for", "map_to_rz",
         "projection"),
        ("extract_flux_surface", "flux_surface_grids", "grid_for",
         "extract_flux_surface", "fs_grid"),
    ])
def test_mapping_reuse_loads_each_block_once_and_preserves_frames(
    monkeypatch, module_name, builder_name, lookup_name, verb, keyword):
  from postgkyl.operations import _mapping

  module = import_module(f"postgkyl.operations.{module_name}")
  datasets = [
      pg.load(str(GENERATED / f"mapped_3d_b{block}-fluid_{frame}.gkyl"))
      for block in (0, 1) for frame in (0, 1)
  ]
  calls = []
  resolve = _mapping.resolve_geometry

  def record(*args, **kwargs):
    calls.append(args[0])
    return resolve(*args, **kwargs)

  monkeypatch.setattr(_mapping, "resolve_geometry", record)
  mappings = getattr(module, builder_name)(
      datasets,
      nodes_file=str(GENERATED / "mapped_3d_b*-geo_int_nodes.gkyl"),
      nz_interp=2)
  assert calls == [datasets[0].file_name, datasets[2].file_name]
  assert len(mappings) == 2
  for data in datasets:
    mapping = getattr(module, lookup_name)(mappings, data)
    result = getattr(data, verb)(**{keyword: mapping})
    np.testing.assert_allclose(result.values,
                               7.0 + data.ctx["block"] + 2 * data.ctx["frame"])
    if verb == "map_to_rz":
      np.testing.assert_allclose(
          result.grid[0][:, 0],
          np.linspace(2 + data.ctx["block"], 3 + data.ctx["block"], 5))


@needs_gkeyll
@pytest.mark.parametrize("verb, build, keyword, options", [
    ("map_to_rz", "resolve_rz_projection", "projection", {
        "z_axis": 1.0
    }),
    ("extract_flux_surface", "resolve_flux_surface_grid", "fs_grid", {
        "nphi": 2
    }),
])
def test_reusable_mapping_excludes_geometry_and_sampling_overrides(
    verb, build, keyword, options):
  data = pg.load(str(GENERATED / "mapped_3d_b0-fluid_0.gkyl"))
  geometry = pg.resolve_geometry(data.file_name)
  mapping = getattr(pg, build)(data, geometry)
  for overrides in (options, {"mapc2p": ""}, {"nodes_file": "unused.gkyl"}):
    with pytest.raises(ValueError, match="cannot be combined"):
      getattr(data, verb)(**{keyword: mapping}, **overrides)


def test_mapping_callables_have_one_equation_agnostic_public_owner():
  from postgkyl import operations
  from postgkyl.cli.app import MODELS

  for name in ("map_to_rz", "extract_flux_surface"):
    operation = getattr(operations, name)
    assert getattr(pg, name) is getattr(pg.GData, name) is operation
    assert next(model for model in MODELS
                if model.name == name).canonical is operation
  assert pg.resolve_geometry is operations.resolve_geometry
  assert not hasattr(pg.gk, "rz")
  assert not hasattr(pg.gk, "fluxsurf")
  assert not hasattr(pg.gk, "resolve_geometry")
