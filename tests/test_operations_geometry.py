"""Contracts for explicit coordinate geometry and modal mapping inputs."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython

from postgkyl.operations import geometry as core_geometry
from postgkyl.operations import _mapping
from postgkyl.io import geometry as file_geometry

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


def _valid_geometry(num_dims=2, *, phi=False, corner=None):
  coords = [np.array([0.0, 1.0]) for _ in range(num_dims)]
  shape = (2, ) * num_dims
  values = np.ones(shape)
  return core_geometry.Geometry(coords, values, 2.0 * values,
                                values if phi else None, corner)


@pytest.mark.parametrize(("candidate", "num_dims", "message"), [
    (_valid_geometry(1), 2, "Geometry has 1 dimensions"),
    (core_geometry.Geometry([np.array([[0.0, 1.0]])], np.ones(
        (2, )), np.ones((2, )), None, None), 1, "one-dimensional arrays"),
    (core_geometry.Geometry([np.array([0.0, 1.0, 0.5])], np.ones(
        (3, )), np.ones((3, )), None, None), 1, "strictly monotonic"),
    (core_geometry.Geometry([np.array([0.0, 1.0])], np.ones(
        (3, )), np.ones(
            (2, )), None, None), 1, "R/Z array shapes are incompatible"),
    (replace(_valid_geometry(2), phi=np.ones(
        (2, ))), 2, "toroidal-angle shape"),
    (_valid_geometry(2,
                     corner=([np.array([0.0, 1.0])], np.ones(
                         (2, )), np.ones(
                             (2, )))), 2, "Corner geometry has 1 dimensions"),
    (_valid_geometry(
        2,
        corner=([np.array([0.0]), np.array([0.0, 1.0])], np.ones(
            (1, 2)), np.ones(
                (1, 2)))), 2, "Corner geometry coordinate and R/Z"),
])
def test_validate_geometry_rejects_each_shape_invariant(candidate, num_dims,
                                                        message):
  with pytest.raises(ValueError, match=message):
    core_geometry._validate_geometry(candidate, num_dims)


def test_validate_modal_data_reports_missing_data_and_metadata():
  empty = pg.GData()
  with pytest.raises(ValueError, match="loaded dataset"):
    _mapping._validate_modal_data(empty, "projection", (0, ))

  no_basis = pg.GData()
  no_basis.push([np.array([0.0, 1.0])], np.ones((1, 1)))
  with pytest.raises(ValueError, match="basis_type"):
    _mapping._validate_modal_data(no_basis, "projection", (1, ))

  no_basis.ctx["basis_type"] = "serendipity"
  no_basis.ctx["poly_order"] = True
  with pytest.raises(ValueError, match="nonnegative integer"):
    _mapping._validate_modal_data(no_basis, "projection", (1, ))


def test_validate_modal_data_reports_grid_shape_and_monotonicity():
  data = pg.GData(ctx={
      "basis_type": "serendipity",
      "poly_order": 0,
      "value_form": "modal",
  })
  data.push([np.array([0.0, 1.0])], np.ones((1, 1)))
  data._grid = [np.array([0.0])]
  with pytest.raises(ValueError, match="one-dimensional edge grid"):
    _mapping._validate_modal_data(data, "projection", (1, ))

  data._grid = [np.array([0.0, 1.0, 0.5])]
  with pytest.raises(ValueError, match="strictly monotonic"):
    _mapping._validate_modal_data(data, "projection", (1, ))


@needs_gkeyll
def test_num_fields_rejects_incompatible_coefficient_count():
  data = pg.GData(ctx={
      "basis_type": "serendipity",
      "poly_order": 1,
      "value_form": "modal",
  })
  data.push([np.array([0.0, 1.0])], np.ones((1, 3)))
  with pytest.raises(ValueError, match="incompatible"):
    _mapping._num_fields(data)


def test_validate_component_rejects_boolean_before_basis_lookup():
  with pytest.raises(ValueError, match="integer component"):
    _mapping._validate_component(pg.GData(), True)


def test_same_grid_rejects_dimension_and_shape_mismatches():
  axis = np.array([0.0, 1.0])
  assert not _mapping._same_grid([axis], [axis, axis])
  assert not _mapping._same_grid([axis], [np.array([0.0, 0.5, 1.0])])


def test_pointwise_file_squeezes_grid_and_values(monkeypatch):
  state = SimpleNamespace(grid=[np.array([[0.0, 1.0, 2.0]])],
                          values=np.ones((1, 2, 1)))
  monkeypatch.setattr(core_geometry, "GDataState", lambda _path: state)
  grid, values, returned = core_geometry._pointwise_file("geometry.gkyl")
  np.testing.assert_array_equal(grid[0], [0.0, 1.0, 2.0])
  np.testing.assert_array_equal(values, [1.0, 1.0])
  assert returned is state


def test_read_nodes_geometry_recovers_gauss_coordinates(monkeypatch):
  grid = [np.array([0.0, 0.5, 1.0]), np.array([-1.0, 0.0, 1.0])]
  values = np.ones((2, 2, 3))
  data = SimpleNamespace(ctx={"geometry_type": file_geometry._MAPC2P_IDX})
  monkeypatch.setattr(core_geometry, "_pointwise_file", lambda _path:
                      (grid, values, data))
  coords, major_r, vert_z, phi = core_geometry._read_nodes_geometry(
      "nodes.gkyl")
  assert [axis.shape for axis in coords] == [(2, ), (2, )]
  assert major_r.shape == vert_z.shape == phi.shape == (2, 2)


def test_read_nodes_geometry_rejects_unknown_layout(monkeypatch):
  grid = [np.array([0.0, 1.0])]
  values = np.ones((2, 3))
  data = SimpleNamespace(ctx={"geometry_type": file_geometry._MAPC2P_IDX})
  monkeypatch.setattr(core_geometry, "_pointwise_file", lambda _path:
                      (grid, values, data))
  with pytest.raises(ValueError, match="Unrecognized nodal geometry layout"):
    core_geometry._read_nodes_geometry("nodes.gkyl")


def test_read_corner_geometry_builds_point_coordinates(monkeypatch):
  grid = [np.array([0.0, 3.0]), np.array([-2.0, 2.0])]
  values = np.ones((3, 4, 3))
  data = SimpleNamespace(ctx={"geometry_type": file_geometry._MAPC2P_IDX})
  monkeypatch.setattr(core_geometry, "_pointwise_file", lambda _path:
                      (grid, values, data))
  coords, major_r, vert_z = core_geometry._read_corner_rz("corner.gkyl")
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
  monkeypatch.setattr(file_geometry.os.path, "exists",
                      lambda path: path in {str(nodes), str(corner)})
  monkeypatch.setattr(
      core_geometry, "_read_nodes_geometry", lambda path:
      (calls.append(path) or (coords, values, values, None)))
  monkeypatch.setattr(
      core_geometry, "_read_corner_rz", lambda path:
      (calls.append(path) or (coords, values, values)))
  resolved = core_geometry.resolve_geometry(str(source), nodes_file=str(nodes))
  assert calls == [str(nodes), str(corner)]
  assert resolved.corner is not None


def test_resolve_geometry_without_a_name_requires_an_override():
  with pytest.raises(ValueError, match="Could not find a geometry file"):
    core_geometry.resolve_geometry(None)
