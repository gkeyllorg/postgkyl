"""Gkeyll flux-surface composition and per-block discovery."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace


def test_flux_surface_grid_collection_caches_by_geometry_prefix(monkeypatch):
  fluxsurf = import_module("postgkyl.diagnostics.gk.fluxsurf")
  monkeypatch.setattr(fluxsurf, "validate_mapping_grid", lambda *_args: None)
  first = SimpleNamespace(file_name="block-one", ctx={"block": 1})
  repeated = SimpleNamespace(file_name="block-one", ctx={"block": 1})
  second = SimpleNamespace(file_name="block-two", ctx={"block": 2})
  calls = []
  monkeypatch.setattr(fluxsurf, "geometry_prefix", lambda path: path)
  monkeypatch.setattr(
      fluxsurf, "resolve_geometry", lambda path, **kwargs: calls.append(
          (path, kwargs)) or path)
  monkeypatch.setattr(
      fluxsurf, "resolve_flux_surface_grid",
      lambda data, geo, **_kwargs: SimpleNamespace(computational_grid=(),
                                                   geometry=geo))

  grids = fluxsurf.flux_surface_grids([first, repeated, second],
                                      mapc2p="map-*.gkyl",
                                      nodes_file="nodes-*.gkyl")
  assert grids == {
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
  assert fluxsurf.grid_for(grids,
                           first) == SimpleNamespace(computational_grid=(),
                                                     geometry="block-one")


def test_gk_fluxsurf_composes_geometry_grid_and_extraction(monkeypatch):
  fluxsurf = import_module("postgkyl.diagnostics.gk.fluxsurf")
  monkeypatch.setattr(fluxsurf, "validate_mapping_grid", lambda *_args: None)
  data = SimpleNamespace(file_name="field.gkyl")
  calls = []
  monkeypatch.setattr(
      fluxsurf, "resolve_geometry", lambda path, **kwargs: calls.append(
          ("geometry", path, kwargs)) or "geo")
  monkeypatch.setattr(
      fluxsurf, "resolve_flux_surface_grid",
      lambda source, geo, **kwargs: calls.append(
          ("grid", source, geo, kwargs)) or "grid")
  monkeypatch.setattr(
      fluxsurf, "extract_flux_surface",
      lambda source, fs_grid, **kwargs: calls.append(
          ("extract", source, fs_grid, kwargs)) or "result")

  result = fluxsurf.fluxsurf(data,
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
