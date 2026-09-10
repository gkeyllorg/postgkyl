"""Explicit flux-surface mapping and reusable grid contracts."""

from __future__ import annotations

from dataclasses import replace
import os

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl import operations as mapping
from postgkyl.diagnostics.gk.geometry import resolve_geometry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELD = os.path.join(ROOT, "tests", "test_data",
                     "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl")

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


@needs_gkeyll
def test_flux_surface_move_preserves_output_and_projection_reuse():
  data = pg.load(FIELD)
  geometry = resolve_geometry(data.file_name)
  grid = mapping.resolve_flux_surface_grid(data,
                                           geometry,
                                           x_idx=0,
                                           nphi=4,
                                           nz_interp=2)
  first = mapping.extract_flux_surface(data, fs_grid=grid)
  second = mapping.extract_flux_surface(data.clone(), fs_grid=grid)
  assert first.values.shape == (4, 64, 1)
  assert first.ctx["interpolated"] is True
  np.testing.assert_allclose(first.values, second.values)


@needs_gkeyll
@pytest.mark.parametrize(("kwargs", "message"), [
    ({
        "nphi": 0
    }, "nphi must be a positive integer"),
    ({
        "nz_interp": 0
    }, "nz_interp must be a positive integer"),
    ({
        "x_idx": -1
    }, "out of bounds"),
])
def test_flux_surface_public_validation(kwargs, message):
  data = pg.load(FIELD)
  geometry = resolve_geometry(data.file_name)
  with pytest.raises(ValueError, match=message):
    mapping.resolve_flux_surface_grid(data, geometry, **kwargs)


@needs_gkeyll
def test_flux_surface_grid_requires_toroidal_geometry_and_integer_index():
  data = pg.load(FIELD)
  geometry = resolve_geometry(data.file_name)
  with pytest.raises(ValueError, match="no toroidal-angle component"):
    mapping.resolve_flux_surface_grid(data, replace(geometry, phi=None))
  with pytest.raises(ValueError, match="x_idx must be an integer"):
    mapping.resolve_flux_surface_grid(data, geometry, x_idx=True)


@needs_gkeyll
def test_flux_surface_grid_requires_two_binormal_and_parallel_points():
  data = pg.load(FIELD).clone()
  data.ctx["poly_order"] = 0
  data._grid[1] = np.array([0.0, 1.0])
  geometry = resolve_geometry(data.file_name)
  with pytest.raises(ValueError, match="at least two interpolated y and z"):
    mapping.resolve_flux_surface_grid(data, geometry)


@needs_gkeyll
def test_extract_flux_surface_validates_reusable_grid_metadata():
  data = pg.load(FIELD)
  geometry = resolve_geometry(data.file_name)
  grid = mapping.resolve_flux_surface_grid(data, geometry, nphi=4, nz_interp=2)

  shifted = data.clone()
  shifted.grid[0] = shifted.grid[0] + 0.1
  with pytest.raises(ValueError, match="computational grid does not match"):
    mapping.extract_flux_surface(shifted, fs_grid=grid)

  with pytest.raises(ValueError, match="out of bounds"):
    mapping.extract_flux_surface(data, fs_grid=replace(grid, x_idx=10_000))

  with pytest.raises(ValueError, match="projection and data grid shapes"):
    mapping.extract_flux_surface(data,
                                 fs_grid=replace(grid, phi_2d=np.ones((1, 1))))


@needs_gkeyll
def test_extract_flux_surface_rejects_zero_toroidal_span():
  data = pg.load(FIELD)
  geometry = resolve_geometry(data.file_name)
  grid = mapping.resolve_flux_surface_grid(data, geometry, nphi=4, nz_interp=2)
  zero_span = replace(grid, phi_2d=np.zeros_like(grid.phi_2d))
  with pytest.raises(ValueError, match="zero or non-finite"):
    mapping.extract_flux_surface(data, fs_grid=zero_span)
