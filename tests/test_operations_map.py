"""Tests for the ``map`` verb (grid mapping) and the ``select`` curvilinear
guard it motivates. See ``MAPPING.md`` for the design; ``postgkyl.dg.map`` is
the (already-tested, layer-03) engine this verb delegates to.

Mapping fields are built two ways:

- **synthetically** (``_synthetic_map``/``_project_1d``/``_project_2d``,
  mirroring ``tests/test_dg_map.py``): exact per-cell coefficients of a
  chosen physical-coordinate function, so the expected grid is computable
  independently of the code under test.
- **from the real generated fixtures** (``generated/2d_c2p_*.gkyl``) for a
  genuine file-based conf-space integration test.

**Conf-space maps (``mapc2p``/``mc2nu``) are one joint ``m``-D curvilinear
map**: every physical coordinate is evaluated over all ``m`` mapped
dimensions, so a non-separable map (e.g. a rotation) is representable.
**Velocity-space maps (``mapc2p_vel``) are diagonal instead**: Gkeyll writes
each mapped dimension as its own independent 1-D map (basis dimensionality
1, not ``m``), broadcast across the other velocity dimensions' cells --
gyrokinetic velocity coordinates never couple. A real vel-space fixture
(``rt_gk_tcv_iwl_1x2v_p1-elc_mapc2p_vel.gkyl``) is laid out exactly this
way: its 4 components live on a 2-D (16, 8) grid, matching
``m * num_basis_1d == 2 * 2 == 4`` for 1-D serendipity p1, and it carries no
``basis_type``/``poly_order`` metadata of its own -- callers must supply it
(see ``load_distf``, which passes ``basis_type="serendipity",
poly_order=1``).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython, operations
from postgkyl.gdatastate.gdatastate import GDataState

needs_gkeyll = pytest.mark.skipif(not gpython.available(),
    reason="no compiled Gkeyll (libg0core.so) found")
pytestmark = needs_gkeyll

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
GEN = os.path.join(DATA, "generated")
F_ELC = os.path.join(DATA, "rt_gk_tcv_iwl_1x2v_p1-elc_250.gkyl")
F_MAPC2P_VEL = os.path.join(DATA, "rt_gk_tcv_iwl_1x2v_p1-elc_mapc2p_vel.gkyl")


# --------------------------------------------------------------- test helpers
def _project_1d(fn, lower, upper, cells, basis_type, poly_order):
  """Exact per-cell modal coefficients of ``fn(z)`` for a 1-D basis (see
  ``tests/test_dg_map.py`` for the same helper at the engine level)."""
  node_eta = gpython.basis.node_coords(basis_type, 1, poly_order)[:, 0]
  n2m = gpython.basis.nodal_to_modal_matrix(basis_type, 1, poly_order)
  dz = (upper - lower) / cells
  centers = lower + (np.arange(cells) + 0.5) * dz
  nodal_z = centers[:, None] + 0.5 * dz * node_eta[None, :]
  return fn(nodal_z) @ n2m.T
# end


def _project_2d(fn, lower, upper, cells, basis_type, poly_order):
  """Exact per-cell modal coefficients of ``fn(z0, z1)`` for a 2-D basis."""
  node_eta = gpython.basis.node_coords(basis_type, 2, poly_order)
  n2m = gpython.basis.nodal_to_modal_matrix(basis_type, 2, poly_order)
  dz = [(upper[d] - lower[d]) / cells[d] for d in range(2)]
  c0 = lower[0] + (np.arange(cells[0]) + 0.5) * dz[0]
  c1 = lower[1] + (np.arange(cells[1]) + 0.5) * dz[1]
  centers = np.stack(np.meshgrid(c0, c1, indexing="ij"), axis=-1)
  node_phys = (centers[:, :, None, :]
      + 0.5 * np.array(dz)[None, None, None, :] * node_eta[None, None, :, :])
  nodal_vals = fn(node_phys[..., 0], node_phys[..., 1])
  return np.einsum("ij,...j->...i", n2m, nodal_vals)
# end


def _synthetic_map(coeffs, lower, upper, cells, *, basis_type="serendipity",
    poly_order=1, value_form="modal"):
  """A gkyl-backed mapping dataset holding ``coeffs`` directly -- no mapc2p
  file needed, per the layer instructions. ``cells`` must be set in ``ctx``
  before ``push`` (``GDataState.set_grid`` needs it to know ``num_dims``,
  and a flat ``GkylArray`` carries no cell layout of its own)."""
  d = GDataState()
  d.ctx.update(basis_type=basis_type, poly_order=poly_order,
      value_form=value_form, cells=np.asarray(cells, dtype=np.int64))
  grid = [np.linspace(lower[i], upper[i], int(cells[i]) + 1)
      for i in range(len(cells))]
  d.push(grid, gpython.GkylArray.from_numpy(coeffs))
  return d
# end


def _numpy_target(grid, values):
  """A NumPy-backed (field-domain) target dataset, built directly."""
  d = GDataState()
  d.push(list(grid), values)
  return d
# end


# ----------------------------------------------------------------- identity
class TestIdentityMap:
  def test_1d_conf_identity_leaves_grid_unchanged(self):
    lower, upper, cells = 0.0, 4.0, 4
    modal = _project_1d(lambda z: z, lower, upper, cells, "serendipity", 1)
    mapping = _synthetic_map(modal, [lower], [upper], [cells])

    target_axis = np.linspace(lower, upper, 17)  # finer than the map's grid
    target = _numpy_target([target_axis], np.zeros((16, 1)))
    out = operations.map(target, mapping, space="conf")

    np.testing.assert_allclose(out.grid[0], target_axis, atol=1e-12)
    assert out.ctx["grid_type"] == "mapped"
  # end

  def test_values_are_untouched(self):
    lower, upper, cells = 0.0, 2.0, 2
    modal = _project_1d(lambda z: z, lower, upper, cells, "serendipity", 1)
    mapping = _synthetic_map(modal, [lower], [upper], [cells])
    values = np.arange(8.0).reshape(4, 2)
    target = _numpy_target([np.linspace(lower, upper, 5)], values)
    out = operations.map(target, mapping, space="conf")
    np.testing.assert_array_equal(out.values, values)
  # end

  def test_new_dataset_by_default_source_grid_untouched(self):
    lower, upper, cells = 0.0, 2.0, 2
    modal = _project_1d(lambda z: z, lower, upper, cells, "serendipity", 1)
    mapping = _synthetic_map(modal, [lower], [upper], [cells])
    target = _numpy_target([np.linspace(lower, upper, 5)], np.zeros((4, 1)))
    out = operations.map(target, mapping, space="conf")
    assert out is not target
    assert "grid_type" not in target.ctx
  # end

  def test_inplace_mutates(self):
    lower, upper, cells = 0.0, 2.0, 2
    modal = _project_1d(lambda z: z, lower, upper, cells, "serendipity", 1)
    mapping = _synthetic_map(modal, [lower], [upper], [cells])
    target = _numpy_target([np.linspace(lower, upper, 5)], np.zeros((4, 1)))
    out = operations.map(target, mapping, space="conf", inplace=True)
    assert out is target
  # end
# end


# ------------------------------------------------------------ conf, 2-D real
class TestConfMapRealFixture:
  """The real generated ``2d_c2p_*`` fixtures for conf-space."""

  def _mapped(self, mapfile):
    # operations.map, not the fluent .map() -- api/gdata.py's fluent wiring for the
    # new physics/map verbs is a different layer's job (out of this layer's
    # scope; see the report).
    data = pg.load(os.path.join(GEN, "2d_ms_p1.gkyl")).interpolate()
    return operations.map(data, os.path.join(GEN, mapfile), space="conf")
  # end

  def test_grid_becomes_curvilinear_with_shape_of_the_axes_it_replaces(self):
    before = pg.load(os.path.join(GEN, "2d_ms_p1.gkyl")).interpolate()
    mapped = self._mapped("2d_c2p_stretch_ms_p1.gkyl")
    expected_shape = (before.grid[0].shape[0], before.grid[1].shape[0])
    assert mapped.grid[0].shape == expected_shape
    assert mapped.grid[1].shape == expected_shape
    assert mapped.grid[0].ndim == 2  # curvilinear: full N-D nodal array
  # end

  def test_values_untouched_by_stretch_map(self):
    before = pg.load(os.path.join(GEN, "2d_ms_p1.gkyl")).interpolate()
    mapped = self._mapped("2d_c2p_stretch_ms_p1.gkyl")
    np.testing.assert_array_equal(mapped.values, before.values)
  # end

  def test_rotation_is_non_separable(self):
    """A rotation map produces coordinates that vary along both axes."""
    mapped = self._mapped("2d_c2p_rot45_ms_p1.gkyl")
    assert np.std(mapped.grid[0], axis=1).max() > 1e-6
  # end
# end


# --------------------------------------------------------------------- vel
class TestVelMap:
  def test_1d_vel_deforms_only_the_trailing_axis(self):
    """m=1: offset = num_dims - m puts the map on the last axis."""
    lower, upper, cells = -1.0, 1.0, 4
    scale = 2.0
    modal = _project_1d(lambda v: scale * v, lower, upper, cells,
        "serendipity", 1)
    mapping = _synthetic_map(modal, [lower], [upper], [cells])

    x_edges = np.linspace(0.0, 1.0, 5)
    v0_edges = np.linspace(0.0, 1.0, 5)
    v1_edges = np.linspace(lower, upper, 9)
    target = _numpy_target([x_edges, v0_edges, v1_edges],
        np.zeros((4, 4, 8, 1)))
    out = operations.map(target, mapping, space="vel")

    np.testing.assert_allclose(out.grid[0], x_edges)   # untouched
    np.testing.assert_allclose(out.grid[1], v0_edges)  # untouched
    np.testing.assert_allclose(out.grid[2], scale * v1_edges, atol=1e-12)
  # end

  def test_2d_vel_is_separable_per_dimension(self):
    """Gkeyll's real ``mapc2p_vel`` files (see the module docstring) store
    each velocity dimension as its own independent 1-D map, broadcast
    across the other dimensions' cells -- not a joint m-D curvilinear map
    like a conf-space map. Both dimensions evaluate independently even
    though m == 2, and the resulting grid arrays stay 1-D."""
    lower, upper, cells = [-1.0, -1.0], [1.0, 1.0], [4, 3]
    m0 = _project_1d(lambda v: 2.0 * v, lower[0], upper[0], cells[0],
        "serendipity", 1)
    m1 = _project_1d(lambda v: 3.0 * v + 1.0, lower[1], upper[1], cells[1],
        "serendipity", 1)
    # Broadcast each 1-D map's coefficients across the other axis' cells,
    # matching Gkeyll's on-disk mapc2p_vel layout.
    coeffs0 = np.broadcast_to(m0[:, None, :], (cells[0], cells[1], m0.shape[-1]))
    coeffs1 = np.broadcast_to(m1[None, :, :], (cells[0], cells[1], m1.shape[-1]))
    mapping = _synthetic_map(np.concatenate([coeffs0, coeffs1], axis=-1),
        lower, upper, cells)

    x_edges = np.linspace(0.0, 1.0, 3)
    v0_edges = np.linspace(lower[0], upper[0], 9)
    v1_edges = np.linspace(lower[1], upper[1], 7)
    target = _numpy_target([x_edges, v0_edges, v1_edges],
        np.zeros((2, 8, 6, 1)))
    out = operations.map(target, mapping, space="vel")

    assert out.grid[1].ndim == 1  # separable: stays 1-D unlike a conf map
    assert out.grid[2].ndim == 1
    np.testing.assert_allclose(out.grid[1], 2.0 * v0_edges, atol=1e-12)
    np.testing.assert_allclose(out.grid[2], 3.0 * v1_edges + 1.0, atol=1e-12)
    np.testing.assert_allclose(out.grid[0], x_edges)  # conf axis untouched
  # end
# end


# --------------------------------------------------------------------- errors
class TestMapErrors:
  def test_rejects_modal_target(self):
    target = pg.load(os.path.join(GEN, "2d_ms_p1.gkyl"))  # not interpolated
    mapping_path = os.path.join(GEN, "2d_c2p_stretch_ms_p1.gkyl")
    with pytest.raises(ValueError, match=r"\.interpolate\(\)"):
      operations.map(target, mapping_path, space="conf")
    # end
  # end

  def test_bad_space_raises(self):
    target = pg.load(os.path.join(GEN, "2d_ms_p1.gkyl")).interpolate()
    with pytest.raises(ValueError, match="'space'"):
      operations.map(target, os.path.join(GEN, "2d_c2p_stretch_ms_p1.gkyl"),
          space="bogus")
    # end
  # end

  def test_map_too_large_for_dataset(self):
    target = pg.load(os.path.join(GEN, "1d_ms_p1.gkyl")).interpolate()  # 1-D
    with pytest.raises(ValueError, match="does not fit"):
      operations.map(target, os.path.join(GEN, "2d_c2p_stretch_ms_p1.gkyl"),
          space="conf")  # a 2-D map does not fit 1-D data
    # end
  # end

  def test_num_comps_validation_error(self):
    lower, upper, cells = 0.0, 1.0, 2
    bad = np.zeros((cells, 3))  # serendipity p1 1-D needs num_basis=2, not 3
    mapping = _synthetic_map(bad, [lower], [upper], [cells])
    target = _numpy_target([np.linspace(lower, upper, 5)], np.zeros((4, 1)))
    with pytest.raises(ValueError, match="component"):
      operations.map(target, mapping, space="conf")
    # end
  # end

  def test_missing_basis_metadata_raises(self):
    d = GDataState()
    d.ctx.update(cells=np.array([2]))
    d.push([np.linspace(0.0, 1.0, 3)], gpython.GkylArray.from_numpy(np.zeros((2, 2))))
    target = _numpy_target([np.linspace(0.0, 1.0, 5)], np.zeros((4, 1)))
    with pytest.raises(ValueError, match="basis_type"):
      operations.map(target, d, space="conf")
    # end
  # end

  def test_vel_map_real_fixture_fits_the_separable_algorithm(self):
    """See the module docstring: this real fixture carries no basis
    metadata of its own (callers must supply it at load time, as
    ``load_distf`` does), and its 4 components match m * num_basis_1d
    == 2 * 2 == 4 for 1-D serendipity p1 -- not m * num_basis_2d (== 2 * 4
    == 8), which is what the joint-curvilinear (conf-style) contract would
    require."""
    mapping = pg.load(F_MAPC2P_VEL, basis_type="serendipity", poly_order=1)
    assert mapping.ctx.get("basis_type") == "serendipity"
    assert mapping.num_dims == 2 and mapping.num_comps == 4
    assert gpython.basis.num_basis("serendipity", 1, 1) == 2
    assert gpython.basis.num_basis("serendipity", 2, 1) == 4

    target = pg.load(F_ELC).interpolate()
    out = operations.map(target, mapping, space="vel")
    assert out.grid[-1].ndim == 1 and out.grid[-2].ndim == 1
    assert np.all(np.isfinite(out.grid[-1]))
    assert np.all(np.isfinite(out.grid[-2]))
  # end
# end


# --------------------------------------- select on curvilinear (conf) grids
class TestSelectCurvilinearGuard:
  def _mapped(self):
    data = pg.load(os.path.join(GEN, "2d_ms_p1.gkyl")).interpolate()
    return operations.map(data, os.path.join(GEN, "2d_c2p_rot45_ms_p1.gkyl"),
        space="conf")
  # end

  def test_coordinate_selector_on_non_separable_axis_refuses(self):
    """A rotation map genuinely couples both axes -- a bare coordinate
    value has no single answer until the other axis is pinned first."""
    mapped = self._mapped()
    with pytest.raises(ValueError, match="varies along another axis"):
      mapped.select(z0=0.0)
    # end
  # end

  def test_slice_selector_on_non_separable_axis_refuses(self):
    mapped = self._mapped()
    with pytest.raises(ValueError, match="varies along another axis"):
      mapped.select(z0="1:3")
    # end
  # end

  def test_coordinate_selector_works_once_the_other_axis_is_pinned(self):
    """Selecting z1 by index first narrows its *values* to one cell (even
    though the curvilinear grid array itself keeps 2 bounding edges); a
    later, separate select() call's z0 coordinate curve is then read off
    that specific (now unambiguous) cross-section -- resolved purely from
    the dataset's own values shape, with no extra state to thread through."""
    mapped = self._mapped()
    pinned = operations.select(mapped, z1=2)
    assert pinned.values.shape[1] == 1  # z1 resolved to a single cell
    assert pinned.grid[0].shape[1] == 2  # edges: 2 bound that one cell
    out = operations.select(pinned, z0=pinned.grid[0][3, 0])
    assert out.values.shape[0] == 1
  # end

  def test_integer_index_selector_still_works(self):
    mapped = self._mapped()
    out = mapped.select(z0=1)
    assert out.values.shape[0] == 1
    # grid holds edges (2 bound one cell) even along a curvilinear axis
    assert out.grid[0].shape[0] == 2
  # end

  def test_coordinate_selector_works_on_a_separable_joint_map(self):
    """A conf-space map stored jointly (m*num_basis components) but whose
    physical coordinates each happen to depend on only one computational
    axis -- e.g. Gkeyll's field-aligned ``mc2nu`` remap -- resolves a
    coordinate value exactly, with no need to pin the other axis first."""
    lower, upper, cells = [0.0, 0.0], [1.0, 1.0], [2, 2]
    m0 = _project_2d(lambda z0, z1: 2.0 * z0, lower, upper, cells,
        "serendipity", 1)
    m1 = _project_2d(lambda z0, z1: 3.0 * z1 + 1.0, lower, upper, cells,
        "serendipity", 1)
    mapping = _synthetic_map(np.concatenate([m0, m1], axis=-1),
        lower, upper, cells)
    target = _numpy_target(
        [np.linspace(lower[0], upper[0], 5), np.linspace(lower[1], upper[1], 5)],
        np.zeros((4, 4, 1)))
    mapped = operations.map(target, mapping, space="conf")
    assert mapped.grid[0].ndim == 2  # stored jointly, curvilinear shape

    out = operations.select(mapped, z0=1.0)  # 2.0 * 0.5 == 1.0
    assert out.values.shape[0] == 1
    assert out.grid[0].shape == (2, 5)  # only z0's own axis narrowed
    assert out.grid[1].shape == (2, 5)  # sibling kept in sync
  # end

  def test_selecting_one_axis_narrows_the_sibling_grid_too(self):
    mapped = self._mapped()
    out = operations.select(mapped, z0=1)
    assert out.grid[0].shape[0] == 2  # z0's own axis: sliced 5 -> 2 (edges)
    assert out.grid[1].shape[0] == 2  # z1's array shares the block shape
  # end

  def test_separable_1d_mapped_axis_keeps_coordinate_selection(self):
    """A vel (m=1) mapped axis stays 1-D, so the ordinary coordinate-lookup
    path (unaffected by the curvilinear guard) still applies."""
    lower, upper, cells = -1.0, 1.0, 4
    modal = _project_1d(lambda v: v, lower, upper, cells, "serendipity", 1)
    mapping = _synthetic_map(modal, [lower], [upper], [cells])
    target = _numpy_target([np.linspace(0.0, 1.0, 5), np.linspace(lower, upper, 9)],
        np.zeros((4, 8, 1)))
    mapped = operations.map(target, mapping, space="vel")
    assert mapped.grid[1].ndim == 1
    out = operations.select(mapped, z1=0.0)
    assert out.values.shape[1] == 1
  # end

  def test_2d_vel_map_keeps_ordinary_selection_behind_a_nonzero_offset(self):
    """An m > 1 ``space="vel"`` map sits behind a nonzero ``offset``
    (``num_dims - m``), same as any vel map -- but because each mapped
    dimension stays its own independent 1-D map (unlike a conf-space
    curvilinear map), every mapped axis keeps ordinary coordinate-based
    ``select``, with no ``mapped_axes`` offset translation needed."""
    lower, upper, cells = [-1.0, -1.0], [1.0, 1.0], [4, 3]
    m0 = _project_1d(lambda v: 2.0 * v, lower[0], upper[0], cells[0],
        "serendipity", 1)
    m1 = _project_1d(lambda v: v, lower[1], upper[1], cells[1],
        "serendipity", 1)
    coeffs0 = np.broadcast_to(m0[:, None, :], (cells[0], cells[1], m0.shape[-1]))
    coeffs1 = np.broadcast_to(m1[None, :, :], (cells[0], cells[1], m1.shape[-1]))
    mapping = _synthetic_map(np.concatenate([coeffs0, coeffs1], axis=-1),
        lower, upper, cells)

    x_edges = np.linspace(0.0, 1.0, 3)
    v0_edges = np.linspace(lower[0], upper[0], 6)  # non-square vs. v1
    v1_edges = np.linspace(lower[1], upper[1], 4)
    target = _numpy_target([x_edges, v0_edges, v1_edges],
        np.arange(2 * 5 * 3).reshape(2, 5, 3, 1).astype(float))
    out = operations.map(target, mapping, space="vel")  # offset = 3 - 2 = 1
    assert out.grid[1].ndim == 1 and out.grid[2].ndim == 1

    sel2 = operations.select(out, z2=v1_edges[2])
    assert sel2.values.shape == (2, 5, 1, 1)
    assert sel2.grid[2].shape == (2,)  # v1's own axis sliced 4 -> 2
    assert sel2.grid[1].shape == (6,)  # untouched by this call

    sel1 = operations.select(out, z1=v0_edges[1])
    assert sel1.values.shape == (2, 1, 3, 1)
    assert sel1.grid[1].shape == (2,)  # v0's own axis sliced 6 -> 2
    assert sel1.grid[2].shape == (4,)  # untouched by this call
  # end
# end
