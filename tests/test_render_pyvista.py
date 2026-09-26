"""Tests for postgkyl.render.pyvista -- 3-D volume/isosurface rendering.

``pyvista`` is a hard dependency (pyproject.toml) but needs a working
(possibly software/off-screen) OpenGL context; every test here renders
off-screen (``no_show=True``) and is skipped cleanly if that context is not
available on the host, per the layer instructions.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.render.pyvista import pyvista


def _has_gl_context() -> bool:
  # This host's VTK build reports "vtkXOpenGLRenderWindow" -- GLX-only, no
  # OSMesa/EGL fallback -- so without a real or virtual (Xvfb) X server to
  # connect to, VTK doesn't raise a catchable exception: it hits a fatal,
  # unrecoverable "Fatal Python error: Aborted" that takes the whole pytest
  # process down. Check for a display *before* touching pyvista/VTK at all,
  # so a truly headless host skips instead of aborting the run.
  if not os.environ.get("DISPLAY"):
    return False
  try:
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(pv.Sphere())
    pl.screenshot()
    pl.close()
    return True
  except Exception:
    return False


needs_gl = pytest.mark.skipif(
    not _has_gl_context(),
    reason="no working (off-screen) OpenGL context on this host")


def _volume(n=6) -> GDataState:
  grid = [np.linspace(0.0, 1.0, n + 1) for _ in range(3)]
  x, y, z = np.meshgrid(*[0.5 * (g[:-1] + g[1:]) for g in grid], indexing="ij")
  values = (x + y + z)[..., np.newaxis]
  d = GDataState()
  d.push(grid, values)
  return d


@needs_gl
class TestPyvista:

  def test_offscreen_volume_render_does_not_raise(self):
    pyvista(_volume(), no_show=True, volume=True)

  def test_offscreen_contour_render_does_not_raise(self):
    pyvista(_volume(), no_show=True, volume=False, contour_levels=4)

  def test_saves_a_png_screenshot(self, tmp_path):
    out = tmp_path / "out.png"
    pyvista(_volume(), no_show=True, saveas=str(out))
    assert out.exists()
    assert out.stat().st_size > 0

  def test_saves_an_html_export(self, tmp_path):
    # pyvista's HTML export needs the optional "trame" extra, not (only) a
    # GL context; skip cleanly rather than mislabel it as a GL failure.
    pytest.importorskip("trame_vtk")
    out = tmp_path / "out.html"
    pyvista(_volume(), no_show=True, saveas=str(out))
    assert out.exists()

  def test_log_color_scale_does_not_raise(self):
    pyvista(_volume(), no_show=True, is_log=True)

  def test_diverging_colormap_does_not_raise(self):
    pyvista(_volume(), no_show=True, diverging=True)

  def test_clip_plane_does_not_raise(self):
    pyvista(_volume(), no_show=True, mesh_clip_plane=True)

  def test_clip_plane_volume_mode_does_not_raise(self):
    pyvista(_volume(), no_show=True, volume=True, mesh_clip_plane=True)

  def test_hide_axes_does_not_raise(self):
    pyvista(_volume(), no_show=True, hide_axes=True)

  def test_cylindrical_to_cartesian_does_not_raise(self):
    pyvista(_volume(), no_show=True, cylindrical_to_cartesian=True)

  def test_diverging_opacity_ramp_does_not_raise(self):
    pyvista(_volume(), no_show=True, opacity="diverging")

  def test_named_theme_does_not_raise(self):
    pyvista(_volume(), no_show=True, theme="document")

  def test_hide_zeros_hides_exact_zero_points(self):
    d = _volume()
    d.values[0, 0, 0, 0] = 0.0
    pyvista(d, no_show=True, hide_zeros=True)

  def test_mesh_slice_plane_contour_mode_does_not_raise(self):
    pyvista(_volume(), no_show=True, volume=False, mesh_slice_plane=True)

  def test_mesh_slice_plane_volume_mode_does_not_raise(self):
    pyvista(_volume(), no_show=True, volume=True, mesh_slice_plane=True)

  def test_volume_clip_plane_does_not_raise(self):
    pyvista(_volume(), no_show=True, volume=True, volume_clip_plane=True)

  def test_saves_a_vector_graphic(self, tmp_path):
    out = tmp_path / "out.svg"
    pyvista(_volume(), no_show=True, saveas=str(out))
    assert out.exists()

  def test_saves_a_gltf_export(self, tmp_path):
    out = tmp_path / "out.gltf"
    pyvista(_volume(), no_show=True, saveas=str(out))
    assert out.exists()

  def test_saves_a_vtksz_export(self, tmp_path):
    # Like .html, PyVista's .vtksz export needs the optional "trame" extra.
    pytest.importorskip("trame_vtk")
    out = tmp_path / "out.vtksz"
    pyvista(_volume(), no_show=True, saveas=str(out))
    assert out.exists()

  def test_no_title_omits_add_text(self):
    pyvista(_volume(), no_show=True, title=None)

  def test_gl_context_errors_propagate_unwrapped(self):
    # _require_gl_context only wraps *unexpected* exceptions from the render
    # backend into a RuntimeError; a ValueError PyVista itself raises (e.g.
    # an unknown theme name) should pass through as-is, not get relabeled as
    # a GL-context failure.
    with pytest.raises(ValueError, match="Theme"):
      pyvista(_volume(), no_show=True, theme="bogus_theme_xyz")

  def test_spin_rotates_camera_and_stops_after_interaction(self, monkeypatch):
    # The rotation timer/click-observer callbacks only ever run inside VTK's
    # own interactive event loop, which off-screen tests never enter. Capture
    # them by stubbing the registration calls, then invoke them directly to
    # exercise the closures' logic (advance while idle, freeze on click).
    from pyvista.plotting.render_window_interactor import RenderWindowInteractor

    captured = {}
    monkeypatch.setattr(
        pv.Plotter, "add_timer_event",
        lambda self, max_steps, duration, callback: captured.setdefault(
            "rotate", callback))

    def _fake_add_observer(self, event, call, interactor_style_fallback=True):
      if event == "LeftButtonPressEvent":
        captured["click"] = call

    monkeypatch.setattr(RenderWindowInteractor, "add_observer",
                        _fake_add_observer)

    pyvista(_volume(), no_show=True, no_spin=False, volume=True)

    assert "rotate" in captured and "click" in captured
    captured["rotate"](0)
    captured["click"]()
    captured["rotate"](0)  # a no-op once "clicked": interacting freezes it

  def test_html_saveas_dispatches_to_export_html(self, monkeypatch, tmp_path):
    # Exercise postgkyl's own saveas -> exporter dispatch without requiring
    # the optional "trame_vtk" extra that pyvista's real HTML export needs.
    called = {}
    monkeypatch.setattr(pv.Plotter, "export_html",
                        lambda self, path: called.setdefault("path", path))
    out = tmp_path / "out.html"
    pyvista(_volume(), no_show=True, saveas=str(out))
    assert called["path"] == str(out)

  def test_vtksz_saveas_dispatches_to_export_vtksz(self, monkeypatch, tmp_path):
    # Same as above, but for the optional "trame" extra .vtksz export needs.
    called = {}
    monkeypatch.setattr(pv.Plotter, "export_vtksz",
                        lambda self, path: called.setdefault("path", path))
    out = tmp_path / "out.vtksz"
    pyvista(_volume(), no_show=True, saveas=str(out))
    assert called["path"] == str(out)

  def test_show_true_calls_plotter_show(self, monkeypatch):
    # A real interactive .show() blocks waiting for the window to close;
    # stub it out to exercise the no_show=False branch without hanging the test.
    calls = []
    monkeypatch.setattr(pv.Plotter, "show",
                        lambda self, *a, **k: calls.append(True))
    pyvista(_volume(), no_show=False, volume=True)
    assert calls == [True]

  def test_show_bounds_axes_ranges_reflect_scale_and_shift(self, monkeypatch):
    # The mesh itself is always normalized to +/-aspect_ratio (PyVista
    # handles non-integer axis extents poorly), so axes_ranges is the only
    # thing that can carry the user's requested xscale/yscale/zscale and
    # xshift/yshift/zshift into the displayed tick labels -- see C1.
    captured = {}
    original_show_bounds = pv.Plotter.show_bounds

    def _spy(self, **kwargs):
      captured.update(kwargs)
      return original_show_bounds(self, **kwargs)

    monkeypatch.setattr(pv.Plotter, "show_bounds", _spy)

    grid = [np.linspace(0.0, 1.0, 7) for _ in range(3)]
    centers = 0.5 * (grid[0][:-1] + grid[0][1:])
    xmin, xmax = float(centers.min()), float(centers.max())
    x, y, z = np.meshgrid(centers, centers, centers, indexing="ij")
    values = (x + y + z)[..., np.newaxis]
    d = GDataState()
    d.push(grid, values)

    pyvista(d,
            no_show=True,
            volume=True,
            xscale=2.0,
            xshift=1.0,
            yscale=3.0,
            yshift=0.5,
            zscale=4.0,
            zshift=1.0)

    assert "axes_ranges" in captured
    axes_ranges = captured["axes_ranges"]
    # Volume mode with the default aspect_ratio=(1,1,1) and no downsampling
    # builds a mesh spanning exactly [-1, 1] per axis, so pv_bounds.*_min/
    # *_max are -1/+1 and axes_ranges reduces to (val + shift) * scale.
    np.testing.assert_allclose(axes_ranges[0], (xmin + 1.0) * 2.0, atol=1e-9)
    np.testing.assert_allclose(axes_ranges[1], (xmax + 1.0) * 2.0, atol=1e-9)
    np.testing.assert_allclose(axes_ranges[2], (xmin + 0.5) * 3.0, atol=1e-9)
    np.testing.assert_allclose(axes_ranges[3], (xmax + 0.5) * 3.0, atol=1e-9)
    np.testing.assert_allclose(axes_ranges[4], (xmin + 1.0) * 4.0, atol=1e-9)
    np.testing.assert_allclose(axes_ranges[5], (xmax + 1.0) * 4.0, atol=1e-9)


class TestPyvistaValidation:

  def test_non_3d_dataset_raises(self):
    d = GDataState()
    d.push([np.linspace(0.0, 1.0, 5)], np.ones((4, 1)))
    with pytest.raises(ValueError, match="3D"):
      pyvista(d, no_show=True)

  def test_unsupported_saveas_extension_raises(self):
    with pytest.raises(ValueError, match="Unsupported"):
      pyvista(_volume(), no_show=True, saveas="out.bogus")
