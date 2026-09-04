"""Canonical PyVista renderer for 3-D scalar-field volumes and isosurfaces.

``pg.pyvista``, ``GData.pyvista``, ``operations.pyvista``, and the generated
CLI are aliases or lowerings of the one public function in this module.
PyVista needs a working (possibly software/off-screen) OpenGL context; every
entry point re-raises a ``RuntimeError`` naming that requirement instead of
letting a VTK error surface from deep inside the library.
"""

from __future__ import annotations

import os

import numpy as np
import pyvista as pv

from postgkyl.cli_spec import (
    CommandSpec,
    Execution,
    ResultPolicy,
    Section,
    command,
)
from postgkyl.gdatastate import GDataState, materialize_point_values
from postgkyl.numerics import downsample, nodal_to_cell_centered_grid

from ._prep import resolve_axis_labels, squeeze_collapsed_axes
from .labels import latex_to_unicode


def _require_gl_context(action):
  """Run ``action`` (a zero-arg callable), turning a VTK/GL failure into a
  clear ``RuntimeError`` instead of an opaque one from deep inside VTK."""
  try:
    return action()
  except (RuntimeError, ValueError):
    raise
  except Exception as exc:  # pragma: no cover - depends on the host's GL stack
    raise RuntimeError(
        "pyvista rendering requires a working (possibly off-screen) OpenGL "
        f"context; the render backend raised: {exc!r}") from exc


@command(
    CommandSpec(Section.RENDER,
                Execution.TERMINAL_EACH,
                result=ResultPolicy.SILENT))
def pyvista(data: GDataState,
            *,
            show: bool = True,
            spin: bool = True,
            max_points_per_axis: int = -1,
            contour_levels: int = 10,
            is_log: bool = False,
            is_contour: bool = True,
            is_shaded: bool = False,
            hide_axes: bool = False,
            mesh_clip_plane: bool = False,
            mesh_slice_plane: bool = False,
            volume_clip_plane: bool = False,
            cmin: float | None = None,
            cmax: float | None = None,
            aspect_ratio: tuple[float, float, float] = (1, 1, 1),
            camera_azimuth: float = 0.0,
            camera_elevation: float = -30.0,
            opacity: str = "sigmoid_4",
            cmap: str = "inferno",
            xlabel: str | None = None,
            ylabel: str | None = None,
            zlabel: str | None = None,
            clabel: str = "",
            title: str | None = "",
            diverging: bool = False,
            cylindrical_to_cartesian: bool = False,
            theme: str = "default",
            saveas: str = "",
            xscale: float = 1.0,
            yscale: float = 1.0,
            zscale: float = 1.0,
            xshift: float = 0.0,
            yshift: float = 0.0,
            zshift: float = 0.0,
            hide_zeros: bool = False):
  """Render a 3-D scalar field with PyVista.

  Builds a structured grid from the (single-component) scalar values and
  renders it as a volume, contour isosurfaces, or an interactive clip/slice
  plane. The grid is normalized to ``aspect_ratio`` because PyVista handles
  non-integer axis extents poorly. Only the first value component is used.

  Args:
    data: dataset to plot; must be 3-D (after squeezing any size-1 axis).
    show: open an interactive render window; off-screen otherwise (also
      forced off-screen when saving to a raster image format).
    spin: slowly auto-rotate the camera until the user interacts with it
      (interactive windows only).
    max_points_per_axis: downsample to at most this many points per axis;
      ``-1`` disables downsampling.
    contour_levels: number of isosurfaces extracted when ``is_contour``.
    is_log: color by log10 of the scalar (non-positive values masked).
    is_contour: render isosurface contours instead of a volume.
    is_shaded: enable shading on the volume render (volume mode only).
    hide_axes: hide the bounding-box axes and labels.
    mesh_clip_plane: add an interactive clip plane along ``-x``.
    mesh_slice_plane: add an interactive slice plane along ``-x``.
    volume_clip_plane: add an interactive volume clip plane (volume mode).
    cmin: Color-limit lower bound; defaults to the data minimum.
    cmax: Color-limit upper bound; defaults to the data maximum.
    aspect_ratio: per-axis aspect the grid is normalized to.
    camera_azimuth: Initial camera azimuth in degrees.
    camera_elevation: Initial camera elevation in degrees.
    opacity: a PyVista opacity preset string, ``"diverging"`` (opaque at
      both ends, transparent in the middle), or a scalar opacity.
    cmap: colormap name; overridden to ``"RdBu_r"`` when ``diverging``.
    xlabel: Horizontal-axis label; auto-derived when omitted.
    ylabel: Vertical-axis label; auto-derived when omitted.
    zlabel: Third-axis label; auto-derived when omitted.
    clabel: colorbar (scalar bar) title.
    title: text drawn at the top of the render; omitted when ``None``.
    diverging: use the diverging ``"RdBu_r"`` colormap.
    cylindrical_to_cartesian: treat grid coordinates as cylindrical
      ``(R, Z, phi)`` and convert to Cartesian before building the mesh.
    theme: PyVista plot theme name; ``"default"`` leaves it unchanged.
    saveas: output path; extension selects the exporter (``.html``,
      ``.png``/``.jpg``/``.jpeg``, ``.pdf``/``.svg``, ``.gltf``, ``.vtksz``).
      Empty string disables saving.
    xscale: Multiplicative scale recorded for the horizontal axis.
    yscale: Multiplicative scale recorded for the vertical axis.
    zscale: Multiplicative scale recorded for the third axis.
    xshift: Additive shift applied to the horizontal axis.
    yshift: Additive shift applied to the vertical axis.
    zshift: Additive shift applied to the third axis.
    hide_zeros: hide grid points whose scalar value is exactly zero.

  Returns:
    None: the function renders and/or saves the plot for its side effects.

  Raises:
    ValueError: ``data`` is not 3-D, or ``saveas`` has an unsupported
      extension.
    RuntimeError: PyVista could not obtain a working OpenGL context.
  """
  data = materialize_point_values(data)
  _valid_exts = ("", ".html", ".png", ".jpg", ".jpeg", ".pdf", ".svg", ".gltf",
                 ".vtksz")
  if saveas and not os.path.splitext(saveas)[1]:
    saveas += ".png"
  if saveas != "" and not saveas.endswith(_valid_exts[1:]):
    raise ValueError(
        "Unsupported file format for saving. Supported formats are: "
        ".html, .png, .jpg, .jpeg, .pdf, .svg, .gltf, .vtksz")

  grid, values = squeeze_collapsed_axes(list(data.grid), data.values)
  num_dims = len(grid)
  if num_dims != 3:
    raise ValueError(f"pyvista renders 3D scalar fields only, got {num_dims}D")
  xlabel, ylabel, zlabel, clabel = resolve_axis_labels(xlabel=xlabel,
                                                       ylabel=ylabel,
                                                       zlabel=zlabel,
                                                       clabel=clabel,
                                                       num_dims=num_dims,
                                                       xshift=xshift,
                                                       yshift=yshift,
                                                       zshift=zshift,
                                                       xscale=xscale,
                                                       yscale=yscale,
                                                       zscale=zscale)

  scalar = np.asarray(values[..., 0])
  x, y, z = nodal_to_cell_centered_grid(grid, scalar.shape, meshgrid=True)
  if cylindrical_to_cartesian:
    r, z_cyl, theta_ang = x, y, z
    x = r * np.cos(theta_ang)
    y = r * np.sin(theta_ang)
    z = z_cyl

  xmax, xmin = np.max(x), np.min(x)
  ymax, ymin = np.max(y), np.min(y)
  zmax, zmin = np.max(z), np.min(z)
  datamax, datamin = np.max(scalar), np.min(scalar)
  x_range, y_range, z_range = xmax - xmin, ymax - ymin, zmax - zmin

  # Normalize to [-aspect, aspect] per axis -- PyVista struggles with
  # non-integer axis extents.
  x = (x - xmin) / x_range * aspect_ratio[0] * 2 - aspect_ratio[0]
  y = (y - ymin) / y_range * aspect_ratio[1] * 2 - aspect_ratio[1]
  z = (z - zmin) / z_range * aspect_ratio[2] * 2 - aspect_ratio[2]

  x, y, z, scalar = downsample(x,
                               y,
                               z,
                               scalar,
                               maximum_points_per_axis=max_points_per_axis)

  if diverging:
    cmap = "RdBu_r"
  if opacity == "diverging":
    cx = np.linspace(0, 1, num=255)
    opacity = np.abs(cx - 0.5) * 2

  off_screen = saveas.endswith((".png", ".jpg", ".jpeg")) or not show

  def _build_and_render():
    pl = pv.Plotter(window_size=(1400, 900), off_screen=off_screen)
    grid3d = pv.StructuredGrid(x, y, z)

    if theme != "default":
      pv.set_plot_theme(theme)

    if hide_zeros:
      x_ind, y_ind, z_ind = np.where(scalar == 0)
      zero_indices = np.ravel_multi_index((x_ind, y_ind, z_ind),
                                          dims=scalar.shape,
                                          order="F")
      if zero_indices.size:
        grid3d.hide_points(zero_indices)

    grid3d["f_raw"] = scalar.ravel(order="F")
    field = np.asarray(grid3d["f_raw"], dtype=float)

    colorbarformat = "%.2e"
    clim = (cmin if cmin is not None else datamin,
            cmax if cmax is not None else datamax)
    if is_log:
      positive_mask = np.asarray(grid3d["f_raw"]) > 0.0
      field = np.full(field.shape, np.nan, dtype=float)
      field[positive_mask] = np.log10(
          np.asarray(grid3d["f_raw"])[positive_mask])
      finite_field = field[np.isfinite(field)]
      colorbarformat = "10^%.1f"
      clim = (
          np.log10(cmin) if cmin is not None else float(np.min(finite_field)),
          np.log10(cmax) if cmax is not None else float(np.max(finite_field)))
    grid3d["f_plot"] = field

    scalar_bar_args = {"title": latex_to_unicode(clabel), "fmt": colorbarformat}

    if is_contour:
      contours = grid3d.contour(isosurfaces=contour_levels, scalars="f_plot")
      if mesh_clip_plane:
        pl.add_mesh_clip_plane(contours,
                               cmap=cmap,
                               clim=clim,
                               normal="-x",
                               opacity=opacity,
                               scalar_bar_args=scalar_bar_args,
                               factor=1.0)
      elif mesh_slice_plane:
        pl.add_mesh_slice(contours,
                          cmap=cmap,
                          clim=clim,
                          normal="-x",
                          opacity=opacity,
                          scalar_bar_args=scalar_bar_args,
                          factor=1.0)
      else:
        pl.add_mesh(contours,
                    cmap=cmap,
                    clim=clim,
                    opacity=opacity,
                    scalar_bar_args=scalar_bar_args)
    else:
      if mesh_clip_plane:
        pl.add_mesh_clip_plane(grid3d,
                               scalars="f_plot",
                               cmap=cmap,
                               clim=clim,
                               opacity=opacity,
                               normal="-x",
                               scalar_bar_args=scalar_bar_args,
                               factor=1.0)
      elif mesh_slice_plane:
        pl.add_mesh_slice(grid3d,
                          scalars="f_plot",
                          cmap=cmap,
                          clim=clim,
                          opacity=opacity,
                          normal="-x",
                          scalar_bar_args=scalar_bar_args,
                          factor=1.0)
      else:
        vol = pl.add_volume(grid3d,
                            scalars="f_plot",
                            cmap=cmap,
                            clim=clim,
                            opacity=opacity,
                            shade=is_shaded,
                            scalar_bar_args=scalar_bar_args)
        if volume_clip_plane:
          pl.add_volume_clip_plane(vol, normal="-x")

    if title is not None:
      pl.add_text(latex_to_unicode(f"{title}"),
                  position="upper_edge",
                  font_size=12)

    if hide_axes:
      pl.hide_axes()
    else:
      # The mesh itself is normalized to +/-aspect_ratio (see above), so its
      # own bounds carry no physical meaning; axes_ranges relabels the ticks
      # with the true (shift/scale-adjusted) physical extent instead.
      pv_bounds = pl.bounds
      axes_ranges = (-(xmin + xshift) * xscale * pv_bounds.x_min,
                     (xmax + xshift) * xscale * pv_bounds.x_max,
                     -(ymin + yshift) * yscale * pv_bounds.y_min,
                     (ymax + yshift) * yscale * pv_bounds.y_max,
                     -(zmin + zshift) * zscale * pv_bounds.z_min,
                     (zmax + zshift) * zscale * pv_bounds.z_max)
      pl.show_bounds(xtitle=latex_to_unicode(xlabel),
                     ytitle=latex_to_unicode(ylabel),
                     ztitle=latex_to_unicode(zlabel),
                     axes_ranges=axes_ranges,
                     n_xlabels=3,
                     n_ylabels=3,
                     n_zlabels=3,
                     grid="back",
                     location="origin",
                     all_edges=True,
                     use_3d_text=False,
                     fmt="%.2e")

    pl.camera.azimuth = camera_azimuth
    pl.camera.elevation = camera_elevation
    if spin:
      state = {"angle": camera_azimuth, "interacting": False}

      def _rotate(_step):
        if state["interacting"]:
          return
        state["angle"] += 0.5
        pl.camera.azimuth = state["angle"] % 360

      def _on_click(*_args):
        state["interacting"] = True

      pl.add_timer_event(max_steps=99999999, duration=50, callback=_rotate)
      pl.iren.add_observer("LeftButtonPressEvent", _on_click)

    if saveas != "":
      if saveas.endswith(".html"):
        pl.export_html(saveas)
      elif saveas.endswith((".pdf", ".svg")):
        pl.save_graphic(saveas)
      elif saveas.endswith((".png", ".jpg", ".jpeg")):
        pl.screenshot(saveas)
      elif saveas.endswith(".gltf"):
        pl.export_gltf(saveas)
      elif saveas.endswith(".vtksz"):
        pl.export_vtksz(saveas)

    if show:
      pl.show()
    else:
      pl.close()

  _require_gl_context(_build_and_render)


__all__ = ["pyvista"]
