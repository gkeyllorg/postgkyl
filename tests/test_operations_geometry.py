"""Contracts for explicit coordinate geometry and modal mapping inputs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython

from postgkyl.operations import geometry as core_geometry

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
    core_geometry._validate_modal_data(empty, "projection", (0, ))

  no_basis = pg.GData()
  no_basis.push([np.array([0.0, 1.0])], np.ones((1, 1)))
  with pytest.raises(ValueError, match="basis_type"):
    core_geometry._validate_modal_data(no_basis, "projection", (1, ))

  no_basis.ctx["basis_type"] = "serendipity"
  no_basis.ctx["poly_order"] = True
  with pytest.raises(ValueError, match="nonnegative integer"):
    core_geometry._validate_modal_data(no_basis, "projection", (1, ))


def test_validate_modal_data_reports_grid_shape_and_monotonicity():
  data = pg.GData(ctx={
      "basis_type": "serendipity",
      "poly_order": 0,
      "value_form": "modal",
  })
  data.push([np.array([0.0, 1.0])], np.ones((1, 1)))
  data._grid = [np.array([0.0])]
  with pytest.raises(ValueError, match="one-dimensional edge grid"):
    core_geometry._validate_modal_data(data, "projection", (1, ))

  data._grid = [np.array([0.0, 1.0, 0.5])]
  with pytest.raises(ValueError, match="strictly monotonic"):
    core_geometry._validate_modal_data(data, "projection", (1, ))


@needs_gkeyll
def test_num_fields_rejects_incompatible_coefficient_count():
  data = pg.GData(ctx={
      "basis_type": "serendipity",
      "poly_order": 1,
      "value_form": "modal",
  })
  data.push([np.array([0.0, 1.0])], np.ones((1, 3)))
  with pytest.raises(ValueError, match="incompatible"):
    core_geometry._num_fields(data)


def test_validate_component_rejects_boolean_before_basis_lookup():
  with pytest.raises(ValueError, match="integer component"):
    core_geometry._validate_component(pg.GData(), True)


def test_same_grid_rejects_dimension_and_shape_mismatches():
  axis = np.array([0.0, 1.0])
  assert not core_geometry._same_grid([axis], [axis, axis])
  assert not core_geometry._same_grid([axis], [np.array([0.0, 0.5, 1.0])])
