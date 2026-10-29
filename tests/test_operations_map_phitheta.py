"""Flux-surface (phi, theta) mapping: analytic values and reusable contracts."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl import operations

from generate_test_data import TWISTED_SPANS, twisted_fields

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
F2D = os.path.join(DATA, "gk_ltx_iwl_2x2v_p1-elc_M2par_10.gkyl")
F3D = os.path.join(DATA, "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl")
GENERATED = Path(DATA) / "generated"

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


@needs_gkeyll
@pytest.mark.parametrize("prefix", TWISTED_SPANS)
def test_surface_follows_the_analytic_field_at_every_angle(prefix):
  # Exact p1 polynomials on a twisted geometry whose binormal domain fills
  # 1/4.19 of the torus; the full turn mixes simulated points and copies.
  data = pg.load(str(GENERATED / f"{prefix}-field_0.gkyl"))
  surface = data.map_to_phitheta(x_idx=-2, nphi=7, nz_interp=4, comp=1)

  phi = np.linspace(0.0, 2 * np.pi, 7, endpoint=False)
  # Sixteen parallel samples on [1, 5]; x_idx=-2 is the fifth of six
  # radial subcell centers on [-1, 2].
  theta = 1.0 + 0.25 * (np.arange(16) + 0.5)
  expected = twisted_fields(1.25, theta, phi[:, None], TWISTED_SPANS[prefix])
  np.testing.assert_allclose(surface.grid[0], phi, rtol=0, atol=1e-15)
  np.testing.assert_allclose(surface.grid[1], theta, rtol=0, atol=1e-15)
  assert surface.values.shape == (7, 16, 1)
  np.testing.assert_allclose(surface.values[..., 0],
                             expected[1],
                             rtol=1e-13,
                             atol=0)
  assert surface.backend == "numpy" and surface.is_interpolated


@needs_gkeyll
def test_each_column_is_a_row_of_the_poloidal_slice():
  data = pg.load(F3D)
  geometry = pg.resolve_geometry(data.file_name)
  projection = pg.resolve_phitheta_projection(data,
                                              geometry,
                                              x_idx=20,
                                              nphi=6,
                                              nz_interp=2)
  surface = data.map_to_phitheta(projection=projection)
  rz = pg.resolve_rz_projection(data, geometry, nz_interp=2)
  for column, angle in zip(surface.values[..., 0], projection.phi_tor):
    poloidal = data.map_to_rz(projection=rz, phi_tor=angle)
    np.testing.assert_allclose(column, poloidal.values[20, :, 0], rtol=1e-12)


@needs_gkeyll
def test_inferred_geometry_reuse_and_public_spellings_agree():
  data = pg.load(F3D)
  inferred = pg.map_to_phitheta(data, x_idx=-3, nphi=4, nz_interp=2)
  projection = pg.resolve_phitheta_projection(data,
                                              pg.resolve_geometry(F3D),
                                              x_idx=93,
                                              nphi=4,
                                              nz_interp=2)
  reused = data.map_to_phitheta(projection=projection, tag="fs", label="lcfs")
  np.testing.assert_allclose(reused.values, inferred.values)
  for actual, expected in zip(reused.grid, inferred.grid):
    np.testing.assert_allclose(actual, expected)
  assert (reused.tag, reused.label) == ("fs", "lcfs")
  assert pg.map_to_phitheta is operations.map_to_phitheta
  assert pg.GData.map_to_phitheta is operations.map_to_phitheta

  inplace = data.clone()
  assert pg.map_to_phitheta(inplace, projection=projection,
                            inplace=True) is inplace
  np.testing.assert_allclose(inplace.values, inferred.values)


@needs_gkeyll
@pytest.mark.parametrize(("options", "message"), [
    ({
        "nphi": 0
    }, "nphi must be a positive integer"),
    ({
        "nz_interp": 0
    }, "nz_interp must be a positive integer"),
    ({
        "x_idx": 96
    }, "out of bounds for data with Nx=96"),
    ({
        "x_idx": -97
    }, "out of bounds for data with Nx=96"),
    ({
        "x_idx": True
    }, "integer radial index"),
    ({
        "x_idx": 1.5
    }, "integer radial index"),
])
def test_projection_options_are_validated(options, message):
  data = pg.load(F3D)
  with pytest.raises(ValueError, match=message):
    pg.resolve_phitheta_projection(data, pg.resolve_geometry(F3D), **options)


@needs_gkeyll
def test_reusable_projection_contracts():
  data = pg.load(F3D)
  geometry = pg.resolve_geometry(F3D)
  with pytest.raises(ValueError, match="no toroidal-angle component"):
    pg.resolve_phitheta_projection(data, replace(geometry, phi=None))

  projection = pg.resolve_phitheta_projection(data,
                                              geometry,
                                              nphi=4,
                                              nz_interp=2)
  shifted = data.clone()
  shifted.grid[0] = shifted.grid[0] + 0.1
  with pytest.raises(ValueError, match="computational grid does not match"):
    pg.map_to_phitheta(shifted, projection=projection)
  with pytest.raises(ValueError, match="out of bounds"):
    pg.map_to_phitheta(data, projection=replace(projection, x_idx=10_000))
  zero_span = replace(projection.rz, span=np.zeros_like(projection.rz.span))
  with pytest.raises(ValueError, match="span must be finite and nonzero"):
    pg.map_to_phitheta(data, projection=replace(projection, rz=zero_span))
  with pytest.raises(ValueError, match="requires 3-D"):
    pg.map_to_phitheta(pg.load(F2D))
  with pytest.raises(ValueError, match="un-interpolated modal DG"):
    pg.map_to_phitheta(data.interpolate())
