"""Toroidal-wedge boundary mapping: analytic surfaces and reusable contracts."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import postgkyl as pg
from postgkyl import gpython
from postgkyl import operations
from postgkyl.cli.state import DataSpace

from generate_test_data import TWISTED_SPANS, twisted_fields

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
F2D = os.path.join(DATA, "gk_ltx_iwl_2x2v_p1-elc_M2par_10.gkyl")
F3D = os.path.join(DATA, "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl")

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

GENERATED = Path(DATA) / "generated"


def _boundary_path(rows, angles):
  """(row, angle) pairs in the documented boundary order, closed."""
  inner, outer = rows[0], rows[-1]
  return ([(r, angles[0]) for r in rows] + [(outer, a) for a in angles[1:]] +
          [(r, angles[-1]) for r in rows[-2::-1]] + [(inner, a)
                                                     for a in angles[-2::-1]])


@needs_gkeyll
def test_surface_values_and_points_follow_the_analytic_field():
  # Exact p1 polynomials on a twisted geometry with R = 3 + x and Z = z.
  data = pg.load(str(GENERATED / "twisted_pos_3d_p1-field_0.gkyl"))
  torus = data.map_to_torus(x_idx=(1, -1),
                            phi_tor=(0.3, 2.0),
                            nphi=5,
                            nz_interp=1)

  # p1 samples two subcell centers per cell: x cells have width 1 on
  # [-1, 2] and z cells width 2 on [1, 5].
  xs = -1.0 + 0.5 * (np.arange(6) + 0.5)
  zs = 1.0 + (np.arange(4) + 0.5)
  path = _boundary_path([1, 2, 3, 4, 5], np.linspace(0.3, 2.0, 5))
  z = np.append(zs, zs[0])  # The parallel seam closes on the first column.
  x = np.array([xs[row] for row, _ in path])[:, None]
  phi = np.array([angle for _, angle in path])[:, None]
  expected = twisted_fields(x, z, phi, TWISTED_SPANS["twisted_pos_3d_p1"])[0]

  assert torus.values.shape == (len(path), zs.size + 1, 1, 1)
  np.testing.assert_allclose(torus.values[..., 0, 0],
                             expected,
                             rtol=1e-13,
                             atol=0)
  r = 3.0 + x
  for actual, coordinate in zip(
      torus.grid, np.broadcast_arrays(r * np.cos(phi), r * np.sin(phi), z)):
    assert actual.shape == (len(path), zs.size + 1, 1)
    np.testing.assert_allclose(actual[..., 0], coordinate, rtol=0, atol=1e-12)
  assert torus.ctx["grid_type"] == "mapped"
  assert torus.ctx["mapped_axes"] == {0: 0, 1: 0, 2: 0}
  assert torus.backend == "numpy" and torus.is_interpolated


@needs_gkeyll
def test_caps_are_the_poloidal_projection_at_the_wedge_angles():
  data = pg.load(F3D)
  geometry = pg.resolve_geometry(data.file_name)
  projection = pg.resolve_torus_projection(data,
                                           geometry,
                                           x_idx=(5, 40),
                                           phi_tor=(0.4, 2.9),
                                           nphi=6,
                                           nz_interp=2)
  torus = data.map_to_torus(projection=projection)
  rz = pg.resolve_rz_projection(data, geometry, nz_interp=2)
  rows = np.arange(5, 41)
  # The last cap starts on the outer wall's final sample, after the
  # nphi - 2 interior outer-wall angles.
  last = rows.size + 4
  for angle, cap in ((0.4, torus.values[:rows.size]),
                     (2.9, torus.values[last:last + rows.size])):
    poloidal = data.map_to_rz(projection=rz, phi_tor=angle).values[rows, :, 0]
    if angle == 2.9:
      poloidal = poloidal[::-1]
    np.testing.assert_allclose(cap[:, :-1, 0, 0], poloidal, rtol=1e-12)
    np.testing.assert_allclose(cap[:, -1, 0, 0], poloidal[:, 0], rtol=1e-12)


@needs_gkeyll
def test_inferred_geometry_reuse_and_public_spellings_agree():
  data = pg.load(F3D)
  inferred = pg.map_to_torus(data, x_idx=(3, -3), nphi=4, nz_interp=2)
  projection = pg.resolve_torus_projection(data,
                                           pg.resolve_geometry(F3D),
                                           x_idx=(3, 93),
                                           nphi=4,
                                           nz_interp=2)
  reused = data.map_to_torus(projection=projection, tag="torus", label="wedge")
  np.testing.assert_allclose(reused.values, inferred.values)
  for actual, expected in zip(reused.grid, inferred.grid):
    np.testing.assert_allclose(actual, expected)
  assert (reused.tag, reused.label) == ("torus", "wedge")
  assert pg.map_to_torus is operations.map_to_torus is pg.GData.map_to_torus

  inplace = data.clone()
  assert pg.map_to_torus(inplace, projection=projection,
                         inplace=True) is inplace
  np.testing.assert_allclose(inplace.values, inferred.values)


@needs_gkeyll
@pytest.mark.parametrize("options, message", [
    ({
        "x_idx": (40, 5)
    }, "inner index below the outer"),
    ({
        "x_idx": (0, 96)
    }, "out of bounds for data with Nx=96"),
    ({
        "x_idx": (0, 1.5)
    }, "integer radial index"),
    ({
        "x_idx": (3, )
    }, "two radial indices"),
    ({
        "phi_tor": (1.0, 1.0)
    }, "increase"),
    ({
        "phi_tor": (0.0, 7.0)
    }, "at most 2\\*pi"),
    ({
        "nphi": 1
    }, "at least 2"),
    ({
        "nz_interp": 0
    }, "positive integer"),
])
def test_projection_options_are_validated(options, message):
  data = pg.load(F3D)
  with pytest.raises(ValueError, match=message):
    pg.resolve_torus_projection(data, pg.resolve_geometry(F3D), **options)


@needs_gkeyll
def test_reusable_projection_contracts():
  data = pg.load(F3D)
  projection = pg.resolve_torus_projection(data,
                                           pg.resolve_geometry(F3D),
                                           x_idx=(0, -1),
                                           nphi=3,
                                           nz_interp=2)
  for overrides in ({"nphi": 4}, {"x_idx": (1, 2)}, {"z_axis": 1.0}):
    with pytest.raises(ValueError, match="cannot be combined"):
      data.map_to_torus(projection=projection, **overrides)

  shifted = data.clone()
  shifted.grid[0] = shifted.grid[0] + 0.01
  with pytest.raises(ValueError, match="computational grid does not match"):
    pg.map_to_torus(shifted, projection=projection)
  with pytest.raises(ValueError, match="radial rows exceed"):
    pg.map_to_torus(data,
                    projection=replace(projection, rows=projection.rows + 1))
  with pytest.raises(ValueError, match="surface points and samples differ"):
    pg.map_to_torus(data,
                    projection=replace(projection,
                                       phi_tor=projection.phi_tor[:2]))
  with pytest.raises(ValueError, match="requires 3-D"):
    pg.map_to_torus(pg.load(F2D))
  with pytest.raises(ValueError, match="un-interpolated modal DG"):
    pg.map_to_torus(data.interpolate())


@needs_gkeyll
def test_cli_parses_negative_indices_and_matches_the_api():
  from postgkyl.cli.app import COMMANDS, cli

  data = pg.load(F3D)
  expected = pg.map_to_torus(data,
                             x_idx=(4, -4),
                             phi_tor=(0.0, 3.0),
                             nphi=4,
                             nz_interp=2)
  command = next(c for c in COMMANDS if c.name == "map_to_torus")
  space = DataSpace(datasets=[data])
  result = CliRunner().invoke(command, [
      "--x_idx", "4", "-4", "--phi_tor", "0", "3", "--nphi", "4", "--nz_interp",
      "2"
  ],
                              obj=space)
  assert result.exit_code == 0, result.output
  np.testing.assert_allclose(space.datasets[0].values, expected.values)

  help_text = CliRunner().invoke(cli, ["--help"]).output
  assert "map_to_torus" in help_text.split("Diagnostics:", 1)[0]
