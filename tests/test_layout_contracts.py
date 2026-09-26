"""Invalid geometry and metadata must fail before entering native kernels."""

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.gdatastate.guards import quadrature_order
from postgkyl.gdatastate.layout import dg_layout, resolve_basis_split
from postgkyl.numerics.elementwise import grids_compatible
from postgkyl.numerics.grid_centering import sample_coordinates
from postgkyl.numerics.idx_parser import idx_parser
from postgkyl.operations._compatibility import uniform_cartesian_grid

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


@pytest.mark.parametrize("left, right, expected", [
    (np.arange(3.), np.arange(4.), False),
    (np.empty(0), np.empty(0), True),
    (np.empty((0, 2)), np.empty((0, 3)), False),
])
def test_grid_compatibility_requires_shapes_even_for_empty_axes(
    left, right, expected):
  assert grids_compatible([left], [right]) is expected


def test_sample_coordinates_rejects_joint_coordinate_arrays():
  with pytest.raises(ValueError, match="one-dimensional axis"):
    sample_coordinates(np.zeros((3, 4)), 3)


def test_coordinate_selection_rejects_empty_domain():
  with pytest.raises(ValueError, match="empty coordinate array"):
    idx_parser(1.5, np.empty(0), nodal=True)


@pytest.mark.parametrize("ctx, message", [
    (dict(basis_type="serendipity", poly_order=1,
          value_form="invalid"), "unknown value_form"),
    (dict(basis_type="serendipity"), "basis_type/poly_order"),
])
def test_layout_rejects_incomplete_or_unknown_representation(ctx, message):
  data = pg.GData(ctx=ctx).push([np.arange(4.)], np.zeros((3, 2)))
  with pytest.raises(ValueError, match=message):
    dg_layout(data)


def test_layout_rejects_missing_values():
  data = pg.GData(ctx=dict(basis_type="serendipity", poly_order=1))
  with pytest.raises(ValueError, match="dataset has no values"):
    dg_layout(data)


def test_quadrature_metadata_rejects_unknown_rule():
  data = pg.GData(ctx=dict(num_quad=2, quad_rule="invalid"))
  with pytest.raises(ValueError, match="unknown quadrature rule"):
    quadrature_order(data)


@pytest.mark.parametrize("coordinates", [
    [0., 0., 1.],
    [2., 1., 0.],
    [0., np.nan, 2.],
    [0., np.inf, 2.],
])
def test_native_grid_rejects_nonfinite_or_nonincreasing_edges(coordinates):
  data = pg.GData().push([np.array(coordinates)], np.zeros((2, 1)))
  with pytest.raises(ValueError, match="uniform Cartesian cell edges"):
    uniform_cartesian_grid(data)


def test_native_grid_rejects_inconsistent_cell_dimensions():
  data = pg.GData().push([np.arange(3.)], np.zeros((2, 1)))
  data.ctx["cells"] = [2, 3]
  with pytest.raises(ValueError, match="uniform Cartesian cell edges"):
    uniform_cartesian_grid(data)


@needs_gkeyll
@pytest.mark.parametrize("rule, ncomp", [("gauss", 8), ("gkeyll", 72)])
def test_hybrid_quadrature_cannot_infer_an_ambiguous_split(rule, ncomp):
  ctx = dict(basis_type="hybrid",
             poly_order=1,
             value_form="quad",
             quad_rule=rule,
             num_quad=2)
  with pytest.raises(ValueError, match="split is ambiguous"):
    resolve_basis_split(ctx, 3, ncomp)


@needs_gkeyll
def test_hybrid_split_rejects_dimensions_without_a_producer_basis():
  with pytest.raises(ValueError, match="split is ambiguous"):
    resolve_basis_split(dict(basis_type="hybrid", poly_order=1), 7, 1)


@pytest.mark.parametrize("split", [dict(cdim=2), dict(vdim=1)])
def test_hybrid_split_infers_only_the_missing_dimension(split):
  assert gpython.basis.cdim_vdim("hybrid", 3, **split) == (2, 1)


def test_gkhybrid_rejects_nonproducer_split():
  with pytest.raises(ValueError, match="unsupported gkhybrid split"):
    gpython.basis.cdim_vdim("gkhybrid", 4, cdim=1, vdim=3)


@pytest.mark.parametrize("order", [0, -1])
def test_gauss_quadrature_requires_positive_order(order):
  with pytest.raises(ValueError, match="num_quad must be >= 1"):
    gpython.basis.gauss_quad(2, order)
