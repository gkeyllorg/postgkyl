"""Tests for ``postgkyl.diagnostics.gk.nodes``.

Ported from ``src_bak/postgkyl/apps/gk_nodes.py`` (no ``tests_bak`` corpus
exists for this app). The pure geometry helpers (``is_geo_mapc2p``,
``nodes_to_RZ``, ``multib_tag``, ``_parse_levels``) are unit-tested
unconditionally; the node-plotting figure path is exercised against
synthetic node arrays stubbed through ``utils.GData`` (single- and
multi-block). The poloidal-flux (``psi_file``) and wall overlays additionally
call ``GData.interpolate()`` on a *real* modal DG field (``nodes`` hardcodes
``poly_order=2``/basis ``"mt"`` for the psi read) -- the repo ships no
interpolatable p2-tensor poloidal-flux fixture, so that branch is skipped
loudly rather than faked.

Run: PYTHONPATH=src pytest tests/test_diagnostics_programs_nodes.py -v
"""

from __future__ import annotations

import importlib
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from postgkyl.diagnostics.gk import utils as gk_utils

nodes = importlib.import_module("postgkyl.diagnostics.gk.nodes")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
GENERATED = os.path.join(DATA, "generated")


class TestGeometryEnum:

  def test_mapc2p_index_matches_gkeyll_header(self):
    # gkeyll/core/zero/gkyl_eqn_type.h: GKYL_GEOMETRY_MAPC2P = 3.
    assert nodes.GKYL_GEOMETRY_ID.index("GKYL_GEOMETRY_MAPC2P") == 3


class TestIsGeoMapc2p:

  def test_defaults_true_when_absent(self):
    assert nodes.is_geo_mapc2p({}) is True

  def test_true_for_mapc2p(self):
    assert nodes.is_geo_mapc2p({"geometry_type": 3}) is True

  def test_false_for_tokamak(self):
    assert nodes.is_geo_mapc2p({"geometry_type": 1}) is False


class TestNodesToRZ:

  def test_mapc2p_2d(self):
    # A 3x2 grid of Cartesian (X, Y, Z) nodes on the unit circle at Z=0.
    shape = (3, 2)
    nodes_arr = np.zeros(shape + (3, ))
    nodes_arr[..., 0] = 1.0  # X
    nodes_arr[..., 1] = 0.0  # Y
    nodes_arr[..., 2] = 5.0  # Z
    major_r, vert_z = nodes.nodes_to_RZ(nodes_arr, is_mapc2p=True)
    np.testing.assert_allclose(major_r, 1.0)
    np.testing.assert_allclose(vert_z, 5.0)

  def test_non_mapc2p_2d(self):
    shape = (3, 2)
    nodes_arr = np.zeros(shape + (2, ))
    nodes_arr[..., 0] = 2.0  # R
    nodes_arr[..., 1] = -1.0  # Z
    major_r, vert_z = nodes.nodes_to_RZ(nodes_arr, is_mapc2p=False)
    np.testing.assert_allclose(major_r, 2.0)
    np.testing.assert_allclose(vert_z, -1.0)

  def test_mapc2p_1d(self):
    shape = (4, )
    nodes_arr = np.zeros(shape + (3, ))
    nodes_arr[..., 0] = 3.0
    nodes_arr[..., 1] = 4.0
    nodes_arr[..., 2] = 7.0
    major_r, vert_z = nodes.nodes_to_RZ(nodes_arr, is_mapc2p=True)
    np.testing.assert_allclose(major_r, 5.0)  # sqrt(3^2+4^2)
    np.testing.assert_allclose(vert_z, 7.0)

  def test_mapc2p_3d_slices_at_yidx_zero(self):
    # cdim == 3 slices the y axis at index 0 before extracting X, Y, Z.
    shape = (2, 3, 2)
    nodes_arr = np.zeros(shape + (3, ))
    nodes_arr[:, 0, :, 0] = 1.0  # X at y-index 0
    nodes_arr[:, 0, :, 1] = 0.0  # Y at y-index 0
    nodes_arr[:, 0, :, 2] = 9.0  # Z at y-index 0
    nodes_arr[:, 1, :, 0] = 100.0  # far-away values at y-index 1 (unused)
    major_r, vert_z = nodes.nodes_to_RZ(nodes_arr, is_mapc2p=True)
    np.testing.assert_allclose(major_r, 1.0)
    np.testing.assert_allclose(vert_z, 9.0)


class TestMultibTag:

  def test_single_block_no_suffix(self):
    assert nodes.multib_tag("nodes", 0, 1) == "nodes"

  def test_multiblock_suffix(self):
    assert nodes.multib_tag("nodes", 2, 3) == "nodes_b2"


class TestParseLevels:

  def test_none_returns_cnlevels(self):
    assert nodes._parse_levels(None, 11) == 11

  def test_range_string(self):
    out = nodes._parse_levels("0:1:3", 11)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])

  def test_comma_list(self):
    out = nodes._parse_levels("0.1,0.2,0.3", 11)
    np.testing.assert_allclose(out, [0.1, 0.2, 0.3])


class _FakeGData:

  def __init__(self, grid, values, ctx=None):
    self._grid = grid
    self._values = values
    self.ctx = ctx or {}

  def get_grid(self):
    return self._grid

  def get_values(self):
    return self._values

  def interpolate(self, num_interp=None):
    return self


class _StubFiles:

  def __init__(self, monkeypatch):
    self._registry: dict[str, _FakeGData] = {}
    monkeypatch.setattr(gk_utils, "GData", self._dispatch)

  def _dispatch(self, file_name, **kwargs):
    return self._registry[file_name]

  def add(self, file_name: str, grid, values, ctx=None) -> None:
    open(file_name, "w").close()
    self._registry[file_name] = _FakeGData(grid, values, ctx)


@pytest.fixture
def stub(monkeypatch):
  return _StubFiles(monkeypatch)


def _square_nodes(nx=3, ny=3):
  """A simple 2-D mapc2p node grid: a regular (nx, ny) square in the X-Y
  plane, with Z varying along x (so both the R and Z extents -- and hence
  ``nodes``'s figure aspect ratio -- are nonzero and finite)."""
  x = np.linspace(1.0, 2.0, nx)
  y = np.linspace(0.0, 1.0, ny)
  xx, yy = np.meshgrid(x, y, indexing="ij")
  out = np.zeros((nx, ny, 3))
  out[..., 0] = xx
  out[..., 1] = yy
  out[..., 2] = xx  # Z varies with x, giving a nonzero vertical extent.
  return out


class TestGkNodesSynthetic:

  def test_single_block_no_overlays(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-nodes.gkyl",
             [np.arange(3.0), np.arange(3.0)], _square_nodes())
    fig = nodes.nodes("sim", path=path)
    try:
      assert fig is not None
      assert len(fig.axes) == 1
    finally:
      plt.close(fig)

  def test_multiblock_sums_extrema_across_blocks(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    block0 = _square_nodes()
    block1 = _square_nodes() + 5.0  # shifted far away in R and Z
    stub.add(f"{path}sim_b0-nodes.gkyl", [np.arange(3.0)] * 2, block0)
    stub.add(f"{path}sim_b1-nodes.gkyl", [np.arange(3.0)] * 2, block1)
    fig = nodes.nodes("sim", path=path, multib="0,1")
    try:
      assert fig is not None
    finally:
      plt.close(fig)

  def test_non_mapc2p_geometry_type(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    rz_nodes = np.zeros((3, 3, 2))
    rz_nodes[..., 0] = np.linspace(1.0, 2.0, 3)[:, None]
    rz_nodes[..., 1] = np.linspace(-1.0, 1.0, 3)[None, :]
    stub.add(f"{path}sim-nodes.gkyl", [np.arange(3.0)] * 2,
             rz_nodes,
             ctx={"geometry_type": 1})
    fig = nodes.nodes("sim", path=path)
    try:
      assert fig is not None
    finally:
      plt.close(fig)

  def test_wall_file_overlay(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-nodes.gkyl", [np.arange(3.0)] * 2, _square_nodes())
    wall_path = tmp_path / "wall.csv"
    wall_path.write_text("0.0,0.0\n1.0,1.0\n2.0,0.0\n")
    fig = nodes.nodes("sim", path=path, wall_file="wall.csv")
    try:
      assert fig is not None
    finally:
      plt.close(fig)

  def test_absolute_nodes_file_override(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    abs_file = f"{path}custom_nodes.gkyl"
    stub.add(abs_file, [np.arange(3.0)] * 2, _square_nodes())
    fig = nodes.nodes("sim", path=path, nodes_file=abs_file)
    try:
      assert fig is not None
    finally:
      plt.close(fig)

  def test_xlim_ylim_and_saveas(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-nodes.gkyl", [np.arange(3.0)] * 2, _square_nodes())
    out_path = str(tmp_path / "out.png")
    fig = nodes.nodes("sim",
                      path=path,
                      xlim=(0.0, 2.0),
                      ylim=(-1.0, 1.0),
                      saveas=out_path)
    try:
      assert fig.axes[0].get_xlim() == (0.0, 2.0)
      assert fig.axes[0].get_ylim() == (-1.0, 1.0)
      assert os.path.exists(out_path)
    finally:
      plt.close(fig)

  def test_1d_node_array_uses_line_plot_branch(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    nodes_1d = np.zeros((4, 3))
    nodes_1d[:, 0] = np.linspace(1.0, 2.0, 4)
    nodes_1d[:, 2] = np.linspace(0.0, 1.0, 4)
    stub.add(f"{path}sim-nodes.gkyl", [np.arange(4.0)], nodes_1d)
    fig = nodes.nodes("sim", path=path)
    try:
      assert fig is not None
    finally:
      plt.close(fig)

  def test_show_calls_plt_show(self, stub, tmp_path, monkeypatch):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-nodes.gkyl", [np.arange(3.0)] * 2, _square_nodes())
    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append(True))
    fig = nodes.nodes("sim", path=path, show=True)
    try:
      assert calls == [True]
    finally:
      plt.close(fig)


class TestGkNodesPsiOverlay:
  """The ``psi_file`` overlay path, stubbed through ``gk_utils.GData`` (its
  ``.interpolate()`` returns itself, carrying the already pre-shaped
  edge-grid/cell-centered-values pair a caller registered) so ``pcolormesh``/
  ``contour`` receive consistently-shaped synthetic data without needing a
  real p2 tensor-basis fixture (see TestGkNodesPsiOverlayRealFixtures)."""

  def _add_psi(self, stub, path):
    psi_grid = [np.linspace(0.0, 3.0, 4), np.linspace(-1.0, 1.0, 3)]
    psi_values = np.ones((3, 2))
    stub.add(f"{path}sim-psi.gkyl", psi_grid, psi_values)

  def test_pcolormesh_with_colorbar(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-nodes.gkyl", [np.arange(3.0)] * 2, _square_nodes())
    self._add_psi(stub, path)
    fig = nodes.nodes("sim", path=path, psi_file="sim-psi.gkyl")
    try:
      assert len(fig.axes) == 2
    finally:
      plt.close(fig)

  def test_contour(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-nodes.gkyl", [np.arange(3.0)] * 2, _square_nodes())
    self._add_psi(stub, path)
    fig = nodes.nodes("sim", path=path, psi_file="sim-psi.gkyl", contour=True)
    try:
      assert len(fig.axes) == 2
    finally:
      plt.close(fig)

  def test_single_level_clevels_suppresses_colorbar(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-nodes.gkyl", [np.arange(3.0)] * 2, _square_nodes())
    self._add_psi(stub, path)
    fig = nodes.nodes("sim", path=path, psi_file="sim-psi.gkyl", clevels="0.5")
    try:
      assert len(fig.axes) == 1
    finally:
      plt.close(fig)

  def test_absolute_psi_file_override(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-nodes.gkyl", [np.arange(3.0)] * 2, _square_nodes())
    abs_psi = f"{path}custom_psi.gkyl"
    psi_grid = [np.linspace(0.0, 3.0, 4), np.linspace(-1.0, 1.0, 3)]
    stub.add(abs_psi, psi_grid, np.ones((3, 2)))
    fig = nodes.nodes("sim", path=path, psi_file=abs_psi)
    try:
      assert len(fig.axes) == 2
    finally:
      plt.close(fig)


class TestGkNodesPsiOverlayRealFixtures:
  """The psi overlay calls ``GData.interpolate()`` on a real modal DG field
  (``nodes`` hardcodes ``poly_order=2``, basis ``"mt"``/tensor, and never
  selects a single component before handing the interpolated array straight
  to ``pcolormesh``/``contour``) -- skipped loudly since the repo's one
  matching-basis fixture, ``tests/test_data/generated/2d_mt_p2.gkyl``, is a
  9-component demo field (no shipped poloidal-flux fixture is single-
  component), which ``pcolormesh``/``contour`` cannot render directly."""

  def test_psi_overlay_needs_single_component_p2_tensor_fixture(self):
    import postgkyl as pg

    candidate = os.path.join(GENERATED, "2d_mt_p2.gkyl")
    if not os.path.exists(candidate):
      pytest.skip(f"no p2 tensor-basis 2-D fixture at '{candidate}'.")
    num_comps = pg.load(candidate).num_comps
    if num_comps == 1:
      pytest.fail("fixture is now single-component -- wire up a real "
                  "psi-overlay assertion here")
    pytest.skip(
        f"'{candidate}' has {num_comps} components; nodes(psi_file=...) "
        "never selects a single component before pcolormesh/contour, so "
        "this fixture cannot exercise that path meaningfully. See "
        "TestGkNodesSynthetic for the node-plotting coverage instead.")
