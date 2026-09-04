"""Gyrokinetic R-Z operation, public surfaces, and compatibility paths."""

from __future__ import annotations

import os

import click
import numpy as np
import pytest
from click.testing import CliRunner

import postgkyl as pg
from postgkyl import gpython
from postgkyl.cli.app import COMMANDS

gk_rz_command = next(command for command in COMMANDS if command.name == "gk-rz")
from postgkyl.cli.state import DataSpace
from postgkyl.operations import gyrokinetics as gk_ops

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
F1D = os.path.join(DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")
F2D = os.path.join(DATA, "gk_ltx_iwl_2x2v_p1-elc_M2par_10.gkyl")
F2D_GEO = os.path.join(DATA, "gk_ltx_iwl_2x2v_p1-geo_int_mapc2p.gkyl")
F3D = os.path.join(DATA, "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl")

needs_gkeyll = pytest.mark.skipif(not gpython.available(),
    reason="no compiled Gkeyll (libg0core.so) found")


@needs_gkeyll
def test_2d_mapping_reference_grid_and_values():
  mapped = pg.gk_rz(pg.load(F2D), nz_interp=2)
  assert [axis.shape for axis in mapped.grid] == [(33, 33), (33, 33)]
  assert mapped.values.shape == (32, 32, 1)
  np.testing.assert_allclose(mapped.values.flat[:5], [
      3.12774618e30, 3.15719364e30, 3.23307916e30,
      3.18034806e30, 3.15422838e30], rtol=2e-9)
# end


@needs_gkeyll
def test_3d_mapping_reference_and_fft_phase():
  data = pg.load(F3D)
  geometry = gk_ops.resolve_geometry(data.file_name)
  projection = gk_ops.resolve_rz_projection(data, geometry, nz_interp=2)
  at_zero = gk_ops.map_to_rz(data, projection, phi_tor=0.0)
  at_quarter = gk_ops.map_to_rz(data, projection, phi_tor=np.pi / 2)
  assert [axis.shape for axis in at_zero.grid] == [(97, 65), (97, 65)]
  assert at_zero.values.shape == (96, 64, 1)
  np.testing.assert_allclose(at_zero.values.flat[:5], [
      9.97245134e18, 9.71438453e18, 9.57553863e18,
      9.50610482e18, 9.81316530e18], rtol=2e-9)
  assert not np.allclose(at_zero.values, at_quarter.values)
# end


def test_geometry_prefers_nodes_and_honors_explicit_modal_override(tmp_path, monkeypatch):
  from postgkyl.operations.gyrokinetics import geometry as geometry_module

  source = tmp_path / "sim-field_0.gkyl"
  nodes = tmp_path / "sim-geo_int_nodes.gkyl"
  modal = tmp_path / "sim-geo_int_mapc2p.gkyl"
  nodes.touch()
  modal.touch()
  coords = [np.array([0.0, 1.0]), np.array([-1.0, 1.0])]
  arrays = np.ones((2, 2))
  calls = []
  monkeypatch.setattr(geometry_module, "_read_nodes_geometry",
      lambda path: (calls.append(("nodes", path)) or (coords, arrays, arrays, None)))
  monkeypatch.setattr(geometry_module, "_read_mapc2p_geometry",
      lambda path: (calls.append(("mapc2p", path)) or (coords, arrays, arrays, None)))

  gk_ops.resolve_geometry(str(source))
  assert calls[-1] == ("nodes", str(nodes))
  gk_ops.resolve_geometry(str(source), mapc2p="")
  assert calls[-1] == ("mapc2p", str(modal))
# end


@needs_gkeyll
def test_geometry_overrides_and_validation_errors(tmp_path):
  data = pg.load(F2D)
  with pytest.raises(ValueError, match="either mapc2p=.*nodes_file"):
    pg.gk_rz(data, mapc2p=F2D_GEO, nodes_file=F2D_GEO)
  # end
  explicit = pg.gk_rz(data, mapc2p=F2D_GEO, nz_interp=2)
  inferred = pg.gk_rz(data, nz_interp=2)
  np.testing.assert_allclose(explicit.values, inferred.values)

  missing = data.clone()
  missing._file_name = str(tmp_path / "absent-field_0.gkyl")
  with pytest.raises(ValueError, match="Could not find a geometry file"):
    pg.gk_rz(missing)
  # end
  with pytest.raises(ValueError, match="positive integer"):
    pg.gk_rz(data, nz_interp=0)
  # end
  with pytest.raises(ValueError, match="out of bounds"):
    pg.gk_rz(data, comp=1)
  # end
  with pytest.raises(ValueError, match="requires 2-D or 3-D"):
    pg.gk_rz(pg.load(F1D))
  # end
  with pytest.raises(ValueError, match="un-interpolated modal DG"):
    pg.gk_rz(data.interpolate())
  # end
# end


@needs_gkeyll
def test_missing_toroidal_geometry_and_incompatible_projection_fail_clearly():
  data = pg.load(F3D)
  coords = [np.array([0.0, 1.0])] * 3
  values = np.ones((2, 2, 2))
  no_phi = gk_ops.Geometry(coords=coords, major_r=values, vert_z=values,
      phi=None, corner=None)
  with pytest.raises(ValueError, match="no toroidal-angle component"):
    gk_ops.resolve_rz_projection(data, no_phi)
  # end

  geometry = gk_ops.resolve_geometry(data.file_name)
  projection = gk_ops.resolve_rz_projection(data, geometry, nz_interp=2)
  shifted = data.clone()
  shifted.grid[0] = shifted.grid[0] + 0.01
  with pytest.raises(ValueError, match="computational grid does not match"):
    gk_ops.map_to_rz(shifted, projection)
  # end
# end


@needs_gkeyll
def test_state_propagation_projection_reuse_and_public_surfaces():
  class DerivedData(pg.GData):
    pass
  # end

  source = DerivedData(F2D, tag="source", label="original")
  original = source.values.copy()
  geometry = gk_ops.resolve_geometry(source.file_name)
  projection = gk_ops.resolve_rz_projection(source, geometry, nz_interp=2)
  first = gk_ops.map_to_rz(source, projection, tag="rz", label="mapped")
  second = gk_ops.map_to_rz(source.clone(), projection)
  assert isinstance(first, DerivedData)
  assert first is not source
  assert first.file_name == source.file_name
  assert (first.tag, first.label, first.ctx["interpolated"]) == ("rz", "mapped", True)
  np.testing.assert_array_equal(source.values, original)
  np.testing.assert_allclose(first.values, second.values)

  fluent = source.gk_rz(mapc2p=F2D_GEO, nz_interp=2)
  functional = pg.gk_rz(source, mapc2p=F2D_GEO, nz_interp=2)
  np.testing.assert_allclose(fluent.values, functional.values)
  assert pg.gk_rz is gk_ops.gk_rz

  inplace = source.clone()
  result = pg.gk_rz(inplace, mapc2p=F2D_GEO, nz_interp=2, inplace=True)
  assert result is inplace and result.ctx["interpolated"] is True
# end


@needs_gkeyll
def test_comp_selects_an_explicit_physical_field():
  source = pg.load(F2D)
  multi = pg.GData(ctx={key: value for key, value in source.ctx.items()
      if key != "num_comps"})
  multi.push([axis.copy() for axis in source.grid],
      np.concatenate([source.values, 2.0 * source.values], axis=-1))
  multi._file_name = source.file_name
  first = pg.gk_rz(multi, comp=0, nz_interp=2)
  second = pg.gk_rz(multi, comp=1, nz_interp=2)
  np.testing.assert_allclose(second.values, 2.0 * first.values)
# end


@needs_gkeyll
def test_group_compatibility_cli_and_help_section():
  from postgkyl.cli.app import cli
  from postgkyl.diagnostics.gk import fluxsurf as old_fluxsurf
  from postgkyl.diagnostics.gk import rz as old_rz

  group = pg.GDataGroup([pg.load(F2D), pg.load(F2D)])
  mapped = group.gk_rz(mapc2p=F2D_GEO, nz_interp=2)
  assert isinstance(mapped, pg.GDataGroup) and len(mapped) == 2

  assert old_rz.gk_rz is gk_ops.gk_rz
  assert old_rz.RzProjection is gk_ops.RzProjection
  assert old_fluxsurf.FluxSurfaceGrid is gk_ops.FluxSurfaceGrid
  assert old_fluxsurf.extract_flux_surface is gk_ops.extract_flux_surface

  space = DataSpace(datasets=[pg.load(F2D)])
  with click.Context(gk_rz_command, obj=space) as ctx:
    ctx.invoke(gk_rz_command, mapc2p=F2D_GEO, nodes_file=None,
        z_axis=0.0, phi_tor=0.0, nz_interp=2, use=None, tag="rz", label=None)
  # end
  expected = pg.gk_rz(pg.load(F2D), mapc2p=F2D_GEO, nz_interp=2, tag="rz")
  np.testing.assert_allclose(space.datasets[0].values, expected.values)

  help_text = CliRunner().invoke(cli, ["--help"]).output
  verbs = help_text.split("Diagnostics:", 1)[0]
  diagnostics = help_text.split("Diagnostics:", 1)[1].split("Render:", 1)[0]
  assert "gk-rz" in verbs and "gk-rz" not in diagnostics
# end
