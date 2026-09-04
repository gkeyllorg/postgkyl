"""Characterization tests for the moved gyrokinetic flux-surface operation."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
import os
from types import SimpleNamespace

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.operations import gyrokinetics as gk_ops

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELD = os.path.join(ROOT, "tests", "test_data",
                     "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl")

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


@needs_gkeyll
def test_flux_surface_move_preserves_output_and_projection_reuse():
  data = pg.load(FIELD)
  geometry = gk_ops.resolve_geometry(data.file_name)
  grid = gk_ops.resolve_flux_surface_grid(data,
                                          geometry,
                                          x_idx=0,
                                          nphi=4,
                                          nz_interp=2)
  first = gk_ops.extract_flux_surface(data, grid)
  second = gk_ops.extract_flux_surface(data.clone(), grid)
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
  geometry = gk_ops.resolve_geometry(data.file_name)
  with pytest.raises(ValueError, match=message):
    gk_ops.resolve_flux_surface_grid(data, geometry, **kwargs)


@needs_gkeyll
def test_flux_surface_grid_requires_toroidal_geometry_and_integer_index():
  data = pg.load(FIELD)
  geometry = gk_ops.resolve_geometry(data.file_name)
  with pytest.raises(ValueError, match="no toroidal-angle component"):
    gk_ops.resolve_flux_surface_grid(data, replace(geometry, phi=None))
  with pytest.raises(ValueError, match="x_idx must be an integer"):
    gk_ops.resolve_flux_surface_grid(data, geometry, x_idx=True)


@needs_gkeyll
def test_flux_surface_grid_requires_two_binormal_and_parallel_points():
  data = pg.load(FIELD).clone()
  data.ctx["poly_order"] = 0
  data._grid[1] = np.array([0.0, 1.0])
  geometry = gk_ops.resolve_geometry(data.file_name)
  with pytest.raises(ValueError, match="at least two interpolated y and z"):
    gk_ops.resolve_flux_surface_grid(data, geometry)


@needs_gkeyll
def test_extract_flux_surface_validates_reusable_grid_metadata():
  data = pg.load(FIELD)
  geometry = gk_ops.resolve_geometry(data.file_name)
  grid = gk_ops.resolve_flux_surface_grid(data,
                                          geometry,
                                          nphi=4,
                                          nz_interp=2)

  shifted = data.clone()
  shifted.grid[0] = shifted.grid[0] + 0.1
  with pytest.raises(ValueError, match="computational grid does not match"):
    gk_ops.extract_flux_surface(shifted, grid)

  with pytest.raises(ValueError, match="out of bounds"):
    gk_ops.extract_flux_surface(data, replace(grid, x_idx=10_000))

  with pytest.raises(ValueError, match="projection and data grid shapes"):
    gk_ops.extract_flux_surface(data,
                                replace(grid, phi_2d=np.ones((1, 1))))


@needs_gkeyll
def test_extract_flux_surface_rejects_zero_toroidal_span():
  data = pg.load(FIELD)
  geometry = gk_ops.resolve_geometry(data.file_name)
  grid = gk_ops.resolve_flux_surface_grid(data,
                                          geometry,
                                          nphi=4,
                                          nz_interp=2)
  zero_span = replace(grid, phi_2d=np.zeros_like(grid.phi_2d))
  with pytest.raises(ValueError, match="zero or non-finite"):
    gk_ops.extract_flux_surface(data, zero_span)


def test_flux_surface_grid_collection_caches_by_geometry_prefix(monkeypatch):
  fluxsurf = import_module("postgkyl.operations.gyrokinetics.fluxsurf")
  first = SimpleNamespace(file_name="block-one", ctx={"block": 1})
  repeated = SimpleNamespace(file_name="block-one", ctx={"block": 1})
  second = SimpleNamespace(file_name="block-two", ctx={"block": 2})
  calls = []
  monkeypatch.setattr(fluxsurf, "geometry_prefix", lambda path: path)
  monkeypatch.setattr(
      fluxsurf, "resolve_geometry",
      lambda path, **kwargs: calls.append((path, kwargs)) or path)
  monkeypatch.setattr(
      fluxsurf, "resolve_flux_surface_grid",
      lambda data, geo, **_kwargs: f"grid:{geo}")

  grids = fluxsurf.flux_surface_grids([first, repeated, second],
                                      mapc2p="map-*.gkyl",
                                      nodes_file="nodes-*.gkyl")
  assert grids == {
      "block-one": "grid:block-one",
      "block-two": "grid:block-two",
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
  assert fluxsurf.grid_for(grids, first) == "grid:block-one"


def test_gk_fluxsurf_composes_geometry_grid_and_extraction(monkeypatch):
  fluxsurf = import_module("postgkyl.operations.gyrokinetics.fluxsurf")
  data = SimpleNamespace(file_name="field.gkyl")
  calls = []
  monkeypatch.setattr(
      fluxsurf, "resolve_geometry",
      lambda path, **kwargs: calls.append(("geometry", path, kwargs)) or "geo")
  monkeypatch.setattr(
      fluxsurf, "resolve_flux_surface_grid",
      lambda source, geo, **kwargs: calls.append(
          ("grid", source, geo, kwargs)) or "grid")
  monkeypatch.setattr(
      fluxsurf, "extract_flux_surface",
      lambda source, grid, **kwargs: calls.append(
          ("extract", source, grid, kwargs)) or "result")

  result = fluxsurf.gk_fluxsurf(data,
                                mapc2p="map.gkyl",
                                x_idx=2,
                                nphi=16,
                                nz_interp=3,
                                comp=4,
                                inplace=True,
                                tag="surface",
                                label="flux")
  assert result == "result"
  assert [call[0] for call in calls] == ["geometry", "grid", "extract"]
  assert calls[-1][-1] == {
      "comp": 4,
      "inplace": True,
      "tag": "surface",
      "label": "flux"
  }
