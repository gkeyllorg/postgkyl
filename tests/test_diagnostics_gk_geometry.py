"""Gkeyll auxiliary geometry discovery and decoding."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

geometry = import_module("postgkyl.diagnostics.gk.geometry")


def test_gauss_nodes_are_ordered_inside_each_cell():
  nodes = geometry._gauss_nodes(np.array([0.0, 2.0, 4.0]))
  assert nodes.shape == (4, )
  assert np.all(np.diff(nodes) > 0.0)
  assert nodes[0] > 0.0 and nodes[-1] < 4.0


def test_pointwise_file_squeezes_grid_and_values(monkeypatch):
  state = SimpleNamespace(grid=[np.array([[0.0, 1.0, 2.0]])],
                          values=np.ones((1, 2, 1)))
  monkeypatch.setattr(geometry, "GDataState", lambda _path: state)
  grid, values, returned = geometry._pointwise_file("geometry.gkyl")
  np.testing.assert_array_equal(grid[0], [0.0, 1.0, 2.0])
  np.testing.assert_array_equal(values, [1.0, 1.0])
  assert returned is state


def test_geometry_components_support_cartesian_and_rz_layouts():
  mapc2p = SimpleNamespace(ctx={"geometry_type": geometry._MAPC2P_IDX})
  values = np.array([[[3.0, 4.0, 2.0], [0.0, 2.0, 5.0]]])
  major_r, vert_z, phi = geometry._geometry_components(values, mapc2p,
                                                       "map.gkyl")
  np.testing.assert_allclose(major_r, [[5.0, 2.0]])
  np.testing.assert_allclose(vert_z, [[2.0, 5.0]])
  assert phi.shape == (1, 2)

  rz = SimpleNamespace(ctx={"geometry_type": 1})
  values_3d = np.ones((2, 2, 2, 3))
  major_r, vert_z, phi = geometry._geometry_components(values_3d, rz, "rz.gkyl")
  assert major_r.shape == vert_z.shape == phi.shape == (2, 2, 2)


@pytest.mark.parametrize(("ctx", "values", "message"), [
    ({
        "geometry_type": geometry._MAPC2P_IDX
    }, np.ones((2, 2, 2)), "at least 3 Cartesian"),
    ({
        "geometry_type": 1
    }, np.ones(2), "at least 2 R/Z"),
])
def test_geometry_components_reject_short_layouts(ctx, values, message):
  with pytest.raises(ValueError, match=message):
    geometry._geometry_components(values, SimpleNamespace(ctx=ctx), "bad.gkyl")


def test_read_nodes_geometry_recovers_gauss_coordinates(monkeypatch):
  grid = [np.array([0.0, 0.5, 1.0]), np.array([-1.0, 0.0, 1.0])]
  values = np.ones((2, 2, 3))
  data = SimpleNamespace(ctx={"geometry_type": geometry._MAPC2P_IDX})
  monkeypatch.setattr(geometry, "_pointwise_file", lambda _path:
                      (grid, values, data))
  coords, major_r, vert_z, phi = geometry._read_nodes_geometry("nodes.gkyl")
  assert [axis.shape for axis in coords] == [(2, ), (2, )]
  assert major_r.shape == vert_z.shape == phi.shape == (2, 2)


def test_read_nodes_geometry_rejects_unknown_layout(monkeypatch):
  grid = [np.array([0.0, 1.0])]
  values = np.ones((2, 3))
  data = SimpleNamespace(ctx={"geometry_type": geometry._MAPC2P_IDX})
  monkeypatch.setattr(geometry, "_pointwise_file", lambda _path:
                      (grid, values, data))
  with pytest.raises(ValueError, match="Unrecognized nodal geometry layout"):
    geometry._read_nodes_geometry("nodes.gkyl")


def test_read_corner_geometry_builds_point_coordinates(monkeypatch):
  grid = [np.array([0.0, 3.0]), np.array([-2.0, 2.0])]
  values = np.ones((3, 4, 3))
  data = SimpleNamespace(ctx={"geometry_type": geometry._MAPC2P_IDX})
  monkeypatch.setattr(geometry, "_pointwise_file", lambda _path:
                      (grid, values, data))
  coords, major_r, vert_z = geometry._read_corner_rz("corner.gkyl")
  assert [axis.size for axis in coords] == [3, 4]
  assert major_r.shape == vert_z.shape == (3, 4)


def test_resolve_geometry_honors_explicit_nodes_and_loads_corner(
    monkeypatch, tmp_path):
  source = tmp_path / "sim-field_0.gkyl"
  nodes = tmp_path / "nodes.gkyl"
  corner = tmp_path / "sim-geo_corn_nodes.gkyl"
  coords = [np.array([0.0, 1.0]), np.array([-1.0, 1.0])]
  values = np.ones((2, 2))
  calls = []
  monkeypatch.setattr(geometry.os.path, "exists",
                      lambda path: path in {str(nodes), str(corner)})
  monkeypatch.setattr(
      geometry, "_read_nodes_geometry", lambda path:
      (calls.append(path) or (coords, values, values, None)))
  monkeypatch.setattr(
      geometry, "_read_corner_rz", lambda path:
      (calls.append(path) or (coords, values, values)))
  resolved = geometry.resolve_geometry(str(source), nodes_file=str(nodes))
  assert calls == [str(nodes), str(corner)]
  assert resolved.corner is not None


def test_resolve_geometry_without_a_name_requires_an_override():
  with pytest.raises(ValueError, match="Could not find a geometry file"):
    geometry.resolve_geometry(None)
