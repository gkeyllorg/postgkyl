"""Tests for ``postgkyl.diagnostics.gk.nodes``.

Ported from ``src_bak/postgkyl/apps/gk_nodes.py`` (no ``tests_bak`` corpus
exists for this app). The pure geometry helpers (``is_geo_mapc2p``,
``nodes_to_RZ``, ``multib_tag``, ``_parse_levels``) are unit-tested
unconditionally; the node-plotting figure path is exercised against
synthetic node arrays stubbed through ``utils.GData`` (single- and
multi-block). A generated geometry file and the scalar p2-tensor fixture
also exercise the poloidal-flux overlay through real file loading.

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
from click.testing import CliRunner

from postgkyl import gpython
from postgkyl.cli.app import cli
from postgkyl.diagnostics.gk import utils as gk_utils
from postgkyl.io import geometry

nodes = importlib.import_module("postgkyl.diagnostics.gk.nodes")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
GENERATED = os.path.join(DATA, "generated")


class TestGeometryEnum:

  def test_mapc2p_index_matches_gkeyll_header(self):
    # gkeyll/core/zero/gkyl_eqn_type.h: GKYL_GEOMETRY_MAPC2P = 3.
    assert geometry.GKYL_GEOMETRY_ID.index("GKYL_GEOMETRY_MAPC2P") == 3


class TestIsGeoMapc2p:

  def test_defaults_true_when_absent(self):
    assert geometry.is_geo_mapc2p({}) is True

  def test_true_for_mapc2p(self):
    assert geometry.is_geo_mapc2p({"geometry_type": 3}) is True

  def test_false_for_tokamak(self):
    assert geometry.is_geo_mapc2p({"geometry_type": 1}) is False


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

  def test_surface_nodes_draw_curved_edges_2d(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    corner = np.zeros((3, 3, 2))
    corner[..., 0] = np.arange(3)[:, None] + 1.0
    corner[..., 1] = np.arange(3)[None, :]
    surf0 = np.zeros((3, 4, 2))
    surf0[..., 0] = np.arange(3)[:, None] + 1.25
    surf0[..., 1] = np.array([0.25, 0.75, 1.25, 1.75])
    surf1 = np.zeros((4, 3, 2))
    surf1[..., 0] = np.array([1.25, 1.75, 2.25, 2.75])[:, None]
    surf1[..., 1] = np.arange(3)[None, :] + 0.125
    grid = [np.arange(4.0), np.arange(4.0)]
    stub.add(f"{path}sim-geo_corn_nodes.gkyl", grid, corner,
             {"geometry_type": 1})
    stub.add(f"{path}sim-geo_surf0_nodes.gkyl", grid, surf0,
             {"geometry_type": 1})
    stub.add(f"{path}sim-geo_surf1_nodes.gkyl", grid, surf1,
             {"geometry_type": 1})
    interior = np.zeros((4, 4, 2))
    interior[..., 0] = np.arange(4)[:, None] + 1.1
    interior[..., 1] = np.arange(4)[None, :] + 0.2
    stub.add(f"{path}sim-geo_int_nodes.gkyl", grid, interior,
             {"geometry_type": 1})

    fig = nodes.nodes("sim", path=path, int_nodes=True, no_show=True)
    try:
      marker = fig.axes[0].lines[-1]
      assert marker.get_marker() == "x"
      assert marker.get_linestyle() == "None"
      np.testing.assert_allclose(marker.get_xdata(), interior[..., 0].ravel())
    finally:
      plt.close(fig)

    fig = nodes.nodes("sim", path=path, surf_nodes=True, int_nodes=True)
    try:
      segments = fig.axes[0].collections[0].get_segments()
      np.testing.assert_allclose(segments[0],
                                 [[1, 0], [1.25, 0.25], [1.25, 0.75], [1, 1]])
      assert len(segments) == 12
      np.testing.assert_allclose(fig.axes[0].lines[-3].get_xdata(),
                                 surf0[..., 0].ravel())
      assert fig.axes[0].lines[-1].get_marker() == "x"
    finally:
      plt.close(fig)

  def test_3d_sampling_and_missing_surface(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    corner = np.zeros((2, 2, 2, 2))
    corner[..., 0] = 1 + np.arange(2)[:, None, None] + \
        2 * np.arange(2)[None, :, None]
    corner[..., 1] = np.arange(2)[None, None, :]
    grid = [np.arange(3.0)] * 3
    stub.add(f"{path}sim-geo_corn_nodes.gkyl", grid, corner,
             {"geometry_type": 1})
    with pytest.raises(ValueError, match="geo_int_nodes.gkyl"):
      nodes.nodes("sim", path=path, int_nodes=True)
    interior = np.ones((2, 2, 2, 2))
    interior[:, 0, :, 0] = 2.0
    interior[:, 1, :, 0] = 10.0
    stub.add(f"{path}sim-geo_int_nodes.gkyl", grid, interior,
             {"geometry_type": 1})
    fig = nodes.nodes("sim", path=path, int_nodes=True, no_show=True)
    try:
      np.testing.assert_allclose(fig.axes[0].lines[-1].get_xdata(), 2.0)
      assert fig.axes[0].lines[-1].get_linestyle() == "None"
    finally:
      plt.close(fig)
    with pytest.raises(ValueError, match="geo_surf0_nodes.gkyl"):
      nodes.nodes("sim", path=path, surf_nodes=True)

    zero = np.zeros((2, 2, 2, 2))
    stub.add(f"{path}sim-geo_surf0_nodes.gkyl", grid, zero,
             {"geometry_type": 1})
    with pytest.raises(ValueError, match="all zero"):
      nodes.nodes("sim", path=path, surf_nodes=True)

    for direction, shape in ((0, (2, 2, 2)), (2, (2, 2, 2))):
      surface = np.ones(shape + (2, ))
      surface[..., 0] = 1 + direction
      stub.add(f"{path}sim-geo_surf{direction}_nodes.gkyl", grid, surface,
               {"geometry_type": 1})
    for sampling, expected in (("nearest", 4), ("all", 8)):
      fig = nodes.nodes("sim",
                        path=path,
                        surf_nodes=True,
                        toroidal_sampling=sampling)
      try:
        segments = fig.axes[0].collections[0].get_segments()
        assert len(segments) == expected
        assert len(fig.axes[0].lines) == 4  # corners, surf0, surf2
        np.testing.assert_allclose(segments[0][0],
                                   [1 + 2 * (0.5 - 1 / (2 * np.sqrt(3))), 0])
      finally:
        plt.close(fig)
    with pytest.raises(ValueError, match="Unknown toroidal_sampling"):
      nodes.nodes("sim", path=path, toroidal_sampling="invalid")

  def test_single_block_no_overlays(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-geo_corn_nodes.gkyl",
             [np.arange(3.0), np.arange(3.0)], _square_nodes())
    fig = nodes.nodes("sim", path=path)
    try:
      assert fig is not None
      assert len(fig.axes) == 1
      assert fig.axes[0].get_aspect() == 1.0
      assert max(fig.get_size_inches()) <= 9.0
    finally:
      plt.close(fig)

  def test_stretch_and_shift_resize_and_move_displayed_axes(
      self, stub, tmp_path):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-geo_corn_nodes.gkyl", [np.arange(3.0)] * 2,
             _square_nodes())
    figures = [
        nodes.nodes("sim", path=path, no_show=True),
        nodes.nodes("sim", path=path, stretch_axis=(1.1, 0.8), no_show=True),
        nodes.nodes("sim", path=path, shift_axis=(0.03, 0.02), no_show=True),
        nodes.nodes("sim", path=path, shift_axis=(-0.02, -0.03), no_show=True),
    ]
    try:
      for fig in figures:
        fig.canvas.draw()
      base, stretched, shifted, back = [
          fig.axes[0].get_position() for fig in figures
      ]
      assert stretched.width == pytest.approx(1.1 * base.width)
      assert stretched.height == pytest.approx(0.8 * base.height)
      assert figures[1].axes[0].get_aspect() == "auto"
      assert shifted.x0 == pytest.approx(base.x0 + 0.03)
      assert shifted.y0 == pytest.approx(base.y0 + 0.02)
      assert shifted.width == pytest.approx(base.width)
      assert figures[2].axes[0].get_aspect() == 1.0
      assert back.x0 == pytest.approx(base.x0 - 0.02)
      assert back.y0 == pytest.approx(base.y0 - 0.03)
    finally:
      for fig in figures:
        plt.close(fig)

  def test_cli_surf_nodes_is_a_bare_flag(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-geo_corn_nodes.gkyl", [np.arange(3.0)] * 2,
             _square_nodes())
    surface = np.ones((3, 4, 3))
    stub.add(f"{path}sim-geo_surf0_nodes.gkyl", [np.arange(3.0)] * 2, surface)
    stub.add(f"{path}sim-geo_surf1_nodes.gkyl", [np.arange(3.0)] * 2,
             surface.transpose(1, 0, 2))
    result = CliRunner().invoke(
        cli, ["gk_nodes", "-n", "sim", "-p", path, "-s", "--no_show"])
    assert result.exit_code == 0, result.output

  def test_cli_interior_and_two_axis_layout_options(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-geo_corn_nodes.gkyl", [np.arange(3.0)] * 2,
             _square_nodes())
    stub.add(f"{path}sim-geo_int_nodes.gkyl", [np.arange(3.0)] * 2,
             np.ones((4, 4, 3)))
    result = CliRunner().invoke(cli, [
        "gk_nodes", "-n", "sim", "-p", path, "-i", "--shift_axis", "0.02",
        "-0.01", "--stretch_axis", "1.5", "0.8", "--no_show"
    ])
    assert result.exit_code == 0, result.output

  def test_multiblock_sums_extrema_across_blocks(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    block0 = _square_nodes()
    block1 = _square_nodes() + 5.0  # shifted far away in R and Z
    stub.add(f"{path}sim_b0-geo_corn_nodes.gkyl", [np.arange(3.0)] * 2, block0)
    stub.add(f"{path}sim_b1-geo_corn_nodes.gkyl", [np.arange(3.0)] * 2, block1)
    fig = nodes.nodes("sim", path=path, multib=":")
    try:
      assert fig is not None
    finally:
      plt.close(fig)

  def test_non_mapc2p_geometry_type(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    rz_nodes = np.zeros((3, 3, 2))
    rz_nodes[..., 0] = np.linspace(1.0, 2.0, 3)[:, None]
    rz_nodes[..., 1] = np.linspace(-1.0, 1.0, 3)[None, :]
    stub.add(f"{path}sim-geo_corn_nodes.gkyl", [np.arange(3.0)] * 2,
             rz_nodes,
             ctx={"geometry_type": 1})
    fig = nodes.nodes("sim", path=path)
    try:
      assert fig is not None
    finally:
      plt.close(fig)

  def test_wall_file_overlay(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-geo_corn_nodes.gkyl", [np.arange(3.0)] * 2,
             _square_nodes())
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
    stub.add(f"{path}sim-geo_corn_nodes.gkyl", [np.arange(3.0)] * 2,
             _square_nodes())
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
    stub.add(f"{path}sim-geo_corn_nodes.gkyl", [np.arange(4.0)], nodes_1d)
    fig = nodes.nodes("sim", path=path)
    try:
      assert fig is not None
    finally:
      plt.close(fig)

  def test_show_by_default_and_no_show(self, stub, tmp_path, monkeypatch):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-geo_corn_nodes.gkyl", [np.arange(3.0)] * 2,
             _square_nodes())
    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append(True))
    fig = nodes.nodes("sim", path=path)
    try:
      assert calls == [True]
    finally:
      plt.close(fig)
    fig = nodes.nodes("sim", path=path, no_show=True)
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
    stub.add(f"{path}sim-geo_corn_nodes.gkyl", [np.arange(3.0)] * 2,
             _square_nodes())
    self._add_psi(stub, path)
    fig = nodes.nodes("sim", path=path, psi_file="sim-psi.gkyl")
    try:
      assert len(fig.axes) == 2
    finally:
      plt.close(fig)

  def test_contour(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-geo_corn_nodes.gkyl", [np.arange(3.0)] * 2,
             _square_nodes())
    self._add_psi(stub, path)
    fig = nodes.nodes("sim", path=path, psi_file="sim-psi.gkyl", contour=True)
    try:
      assert len(fig.axes) == 2
    finally:
      plt.close(fig)

  def test_single_level_clevels_suppresses_colorbar(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-geo_corn_nodes.gkyl", [np.arange(3.0)] * 2,
             _square_nodes())
    self._add_psi(stub, path)
    fig = nodes.nodes("sim", path=path, psi_file="sim-psi.gkyl", clevels="0.5")
    try:
      assert len(fig.axes) == 1
    finally:
      plt.close(fig)

  def test_absolute_psi_file_override(self, stub, tmp_path):
    path = str(tmp_path) + "/"
    stub.add(f"{path}sim-geo_corn_nodes.gkyl", [np.arange(3.0)] * 2,
             _square_nodes())
    abs_psi = f"{path}custom_psi.gkyl"
    psi_grid = [np.linspace(0.0, 3.0, 4), np.linspace(-1.0, 1.0, 3)]
    stub.add(abs_psi, psi_grid, np.ones((3, 2)))
    fig = nodes.nodes("sim", path=path, psi_file=abs_psi)
    try:
      assert len(fig.axes) == 2
    finally:
      plt.close(fig)


class TestGkNodesPsiOverlayRealFixtures:
  """Nine p2 tensor coefficients interpolate to one scalar field."""

  @pytest.mark.skipif(not gpython.available(), reason="needs compiled Gkeyll")
  @pytest.mark.parametrize("contour", [False, True])
  def test_psi_overlay_real_files(self, contour):
    from matplotlib.collections import QuadMesh

    path = os.path.join(GENERATED, "gk_programs")
    fig = nodes.nodes("sim",
                      path=path,
                      psi_file=os.path.join(GENERATED, "2d_mt_p2.gkyl"),
                      contour=contour,
                      stretch_axis=(1.1, 0.8),
                      no_show=True)
    try:
      assert len(fig.axes) == 2
      assert fig.axes[1].get_position().width > 0
      if contour:
        assert len(fig.axes[0].collections) > 2
      else:
        meshes = [c for c in fig.axes[0].collections if isinstance(c, QuadMesh)]
        assert len(meshes) == 1
        assert meshes[0].get_array().shape == (24, 24)
    finally:
      plt.close(fig)
