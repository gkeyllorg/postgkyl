"""Plotly rendering backend: interactive 2-D surfaces and 3-D volumes.

Imports only ``gdatastate``/``numerics`` (plus Plotly/Matplotlib themselves),
mirroring ``matplotlib.py``. Plotly cannot render mathtext, so labels go
through ``render.labels.latex_to_html`` instead.

Like ``matplotlib.py``, ``plotly`` and ``plotly_animate`` own their *entire*
save/preview lifecycle here --
``save``/``saveas``/``show`` (plus the rotating-export camera parameters) are
real parameters of both functions, so e.g. ``pg.load(f).interpolate().plotly(show=True)``
opens an auto-rotating browser preview with zero CLI glue, exactly as
the generated CLI (which passes ``show=True`` explicitly to preserve the
terminal's preview-by-default behavior). Both default to inert
(``show=False``, ``save=False``): a bare call just builds and returns the
figure, so a script or unit test never has an unrequested file or browser
side effect. The generated ``plotly`` command lowers the canonical plotting
signature directly; there is no backend-specific plotting command or
operations wrapper.
"""

from __future__ import annotations

import os.path
import tempfile
import time
import webbrowser
from pathlib import Path
from typing import Annotated

import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from postgkyl.command_spec import (
    CliType, CommandSpec, Execution, KeyValue, ResultPolicy, Section, command,
)
from postgkyl.gdatastate import GDataState, materialize_point_values
from postgkyl.numerics import downsample, nodal_to_cell_centered_grid

from ._ffmpeg import require_ffmpeg
from ._prep import resolve_axis_labels, squeeze_collapsed_axes, subplot_grid
from .labels import latex_to_html
from .style import DEFAULT_STYLE, apply_style

def _apply_plot_style(style: str | None, rcParams: dict | None,
    diverging: bool, cmap: str | None, xkcd: bool, *,
    background: str = "dark", invert_cmap: bool = False) -> dict:
  """Apply Matplotlib styling (colormap source) and return Plotly theme colors."""
  import matplotlib.pyplot as plt

  background_name = (background or "dark").strip().lower()
  if style:
    apply_style(style)
  # end
  elif background_name == "light":
    apply_style("default")
  # end
  else:
    apply_style(DEFAULT_STYLE)
  # end

  if background_name == "light":
    mpl.rcParams["figure.facecolor"] = "#ffffff"
    mpl.rcParams["axes.facecolor"] = "#ffffff"
    mpl.rcParams["savefig.facecolor"] = "#ffffff"
    mpl.rcParams["text.color"] = "#111111"
    mpl.rcParams["axes.labelcolor"] = "#111111"
    mpl.rcParams["xtick.color"] = "#111111"
    mpl.rcParams["ytick.color"] = "#111111"
    mpl.rcParams["axes.edgecolor"] = "#222222"
    mpl.rcParams["grid.color"] = "#b8b8b8"
    theme_colors = dict(
        paper_color="#ffffff", scene_color="#ffffff", text_color="#111111",
        grid_color="#b8b8b8", axis_line_color="#222222")
  # end
  else:
    theme_colors = dict(
        paper_color="#000000", scene_color="#000000", text_color="#e6e6e6",
        grid_color="#2a3242", axis_line_color="#9aa3b2")
  # end

  if rcParams:
    for key, value in rcParams.items():
      mpl.rcParams[key] = value
    # end
  # end

  cmap_name = cmap if cmap is not None else ("RdBu_r" if diverging else "inferno")
  mpl.rcParams["image.cmap"] = cmap_name

  if invert_cmap:
    current = mpl.rcParams["image.cmap"]
    mpl.rcParams["image.cmap"] = (current[:-2] if current.endswith("_r")
                                   else f"{current}_r")
  # end

  if xkcd:
    plt.xkcd()
  # end

  return theme_colors
# end


def _plotly_colorscale(cmap_name: str, n: int = 256):
  """Convert a Matplotlib colormap to a Plotly colorscale."""
  cmap = mpl.colormaps.get_cmap(cmap_name).resampled(n)
  xs = np.linspace(0.0, 1.0, n)
  colorscale = []
  for x, rgba in zip(xs, cmap(xs)):
    r, g, b, a = rgba
    colorscale.append([float(x),
        f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {float(a):.3f})"])
  # end
  return colorscale
# end


def _opacity_mapping(colorscale, min_alpha: float, max_alpha: float,
    log_scale: bool = False):
  """Remap a Plotly colorscale's alpha channel over ``[min_alpha, max_alpha]``."""
  min_a = float(np.clip(min_alpha, 0.0, 1.0))
  max_a = float(np.clip(max_alpha, 0.0, 1.0))
  if max_a < min_a:
    min_a, max_a = max_a, min_a
  # end

  out = []
  for stop, color in colorscale:
    stop_value = float(stop)
    mapped_stop = (np.log10(1.0 + 99.0 * stop_value) / np.log10(100.0)
                   if log_scale else stop_value)
    if isinstance(color, str) and color.startswith("rgba(") and color.endswith(")"):
      parts = [part.strip() for part in color[5:-1].split(",")]
      if len(parts) == 4:
        r, g, b = parts[0], parts[1], parts[2]
        alpha = min_a + (max_a - min_a) * mapped_stop
        out.append([stop_value, f"rgba({r}, {g}, {b}, {alpha:.3f})"])
      # end
      else:
        out.append([stop_value, color])
    # end
      # end
    else:
      out.append([stop_value, color])
    # end
  # end
  return out
# end


def _finite_range(values: np.ndarray) -> tuple[float, float]:
  """Finite min/max of an array, ignoring NaN/inf."""
  finite = np.isfinite(values)
  if np.any(finite):
    finite_values = values[finite]
    return float(np.nanmin(finite_values)), float(np.nanmax(finite_values))
  # end
  return float("nan"), float("nan")
# end


def _axis_range(values: np.ndarray, axis_range, log_axis: bool = False):
  """Axis range for a colorbar or scene axis, log10'd when ``log_axis``."""
  lower, upper = _finite_range(values) if axis_range is None else axis_range
  if log_axis:
    lower, upper = np.log10(lower), np.log10(upper)
  # end
  return [lower, upper]
# end


def _log_colorbar_ticks(log_min: float, log_max: float, max_ticks: int = 7):
  """Tick values/text for a logarithmic (decade) colorbar."""
  if not np.isfinite(log_min) or not np.isfinite(log_max):
    return [], []
  # end
  lo, hi = int(np.floor(log_min)), int(np.ceil(log_max))
  hi = max(hi, lo)
  count = hi - lo + 1
  step = max(1, int(np.ceil(count / max_ticks)))
  tick_vals = list(range(lo, hi + 1, step))
  if tick_vals[-1] != hi:
    tick_vals.append(hi)
  # end
  return [float(v) for v in tick_vals], [f"10<sup>{v:d}</sup>" for v in tick_vals]
# end


def _apply_log_colorscale(render_color_value: np.ndarray, cmin_val, cmax_val,
    colorbar_kwargs: dict):
  """Map color values into log10 space; adds decade tick config in place."""
  log_value = np.full(render_color_value.shape, np.nan, dtype=float)
  valid_mask = render_color_value > 0
  log_value[valid_mask] = np.log10(render_color_value[valid_mask])

  if np.any(valid_mask):
    valid_min = float(np.nanmin(log_value[valid_mask]))
    valid_max = float(np.nanmax(log_value[valid_mask]))
  # end
  else:
    valid_min, valid_max = 0.0, 1.0
  # end

  if cmin_val is not None and cmin_val > 0:
    valid_min = float(np.log10(cmin_val))
  # end
  if cmax_val is not None and cmax_val > 0:
    valid_max = float(np.log10(cmax_val))
  # end
  if not np.isfinite(valid_max) or valid_max <= valid_min:
    valid_max = valid_min + 1.0
  # end

  render_color_value = np.nan_to_num(log_value, nan=valid_min,
      posinf=valid_max, neginf=valid_min)

  tick_vals, tick_text = _log_colorbar_ticks(valid_min, valid_max)
  if tick_vals:
    colorbar_kwargs["tickmode"] = "array"
    colorbar_kwargs["tickvals"] = tick_vals
    colorbar_kwargs["ticktext"] = tick_text
  # end
  return render_color_value, valid_min, valid_max
# end


def _resolve_plotly_aspect(aspect: str | float | None):
  """Resolve ``aspectmode``/``aspectratio`` for a Plotly 3-D scene."""
  if aspect is None:
    return "auto", None
  # end
  if isinstance(aspect, str):
    aspect_value = aspect.strip().lower()
    if aspect_value in ("auto", "data", "cube"):
      return aspect_value, None
    # end
    ratio = float(aspect)
    return "manual", dict(x=ratio, y=ratio, z=ratio)
  # end
  ratio = float(aspect)
  return "manual", dict(x=ratio, y=ratio, z=ratio)
# end


def _build_rotation_post_script(scene_name: str, starting_azimuthal_angle: float,
    polar_angle: float, rotation_period: float, radius: float) -> str:
  """Fill in the packaged rotation-controls JS template with camera params."""
  template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
      "rotation_controls.js")
  with open(template_path) as template_file:
    template = template_file.read()
  # end
  replacements = {
      "__PGKYL_SCENE_NAME__": scene_name,
      "__PGKYL_AZIMUTH_DEG__": f"{float(starting_azimuthal_angle):.17g}",
      "__PGKYL_POLAR_DEG__": f"{float(polar_angle):.17g}",
      "__PGKYL_PERIOD_SEC__": f"{float(rotation_period):.17g}",
      "__PGKYL_RADIUS__": f"{float(radius):.17g}",
  }
  for token, value in replacements.items():
    template = template.replace(token, value)
  # end
  return template
# end


def save_rotating_plotly_figure(fig, file_name: str,
    starting_azimuthal_angle: float, fps: int, polar_angle: float,
    rotation_period: float, radius: float = 2.0) -> None:
  """Save a rotating Plotly 3-D figure as a GIF, MP4, or self-rotating HTML.

  Rotates the camera 360 degrees around the vertical axis, starting from
  ``starting_azimuthal_angle`` degrees. ``.gif``/``.mp4`` render frame-by-frame
  through Kaleido and ffmpeg; ``.html`` embeds a small JS animation loop
  instead (no external process).
  """
  import subprocess

  root, ext = os.path.splitext(file_name)
  ext = ext.lower()
  if ext not in (".gif", ".mp4", ".html"):
    raise ValueError(
        "save_rotating_plotly_figure expects an output ending with .gif, "
        ".mp4, or .html")
  # end
  if fps <= 0:
    raise ValueError("fps must be a positive integer")
  # end
  if rotation_period <= 0:
    raise ValueError("rotation_period must be positive")
  # end

  scene_names = [name for name in fig.layout.to_plotly_json().keys()
                 if name == "scene" or name.startswith("scene")]
  if not scene_names:
    raise ValueError("Rotating export requires a Plotly 3D scene figure")
  # end
  scene_name = scene_names[0]

  polar_rad = np.deg2rad(polar_angle)
  xy_radius = radius * np.sin(polar_rad)
  z_eye = radius * np.cos(polar_rad)

  if ext == ".html":
    theta0 = np.deg2rad(starting_azimuthal_angle)
    initial_camera = dict(
        eye=dict(x=float(xy_radius * np.cos(theta0)),
                 y=float(xy_radius * np.sin(theta0)), z=float(z_eye)),
        up=dict(x=0.0, y=0.0, z=1.0), center=dict(x=0.0, y=0.0, z=0.0))
    fig.update_layout(**{scene_name: dict(camera=initial_camera)})

    omega = 2.0 * np.pi / float(rotation_period)
    if omega > 0.0:
      post_script = _build_rotation_post_script(scene_name,
          starting_azimuthal_angle, polar_angle, rotation_period, radius)
      fig.write_html(file_name, include_plotlyjs="cdn", post_script=post_script)
    # end
    else:
      fig.write_html(file_name)
    # end
    return
  # end

  with tempfile.TemporaryDirectory(prefix="pgkyl_rotate_") as tmp_dir:
    frame_pattern = os.path.join(tmp_dir, "frame_%05d.png")
    num_frames = max(2, int(round(float(fps) * float(rotation_period))))
    # Kaleido launches a fresh headless-Chrome process per to_image() call
    # unless a persistent render server is running; for a multi-frame export
    # that means one Chrome startup per frame. Hold the server open for the
    # whole loop so only the first frame pays that cost.
    import kaleido
    kaleido.start_sync_server(silence_warnings=True)
    try:
      for idx in range(num_frames):
        theta = np.deg2rad(starting_azimuthal_angle + 360.0 * idx / num_frames)
        camera = dict(
            eye=dict(x=float(xy_radius * np.cos(theta)),
                     y=float(xy_radius * np.sin(theta)), z=float(z_eye)),
            up=dict(x=0.0, y=0.0, z=1.0), center=dict(x=0.0, y=0.0, z=0.0))
        fig.update_layout(**{name: dict(camera=camera) for name in scene_names})
        png_bytes = fig.to_image(format="png")
        with open(os.path.join(tmp_dir, f"frame_{idx:05d}.png"), "wb") as frame_file:
          frame_file.write(png_bytes)
    # end
        # end
      # end
    finally:
      kaleido.stop_sync_server(silence_warnings=True)
    # end

    ffmpeg_exe = require_ffmpeg("plotly_animate")
    if ext == ".mp4":
      ffmpeg_cmd = [ffmpeg_exe, "-y", "-framerate", str(fps), "-i", frame_pattern,
          "-pix_fmt", "yuv420p", file_name]
    # end
    else:
      ffmpeg_cmd = [ffmpeg_exe, "-y", "-framerate", str(fps), "-i", frame_pattern,
          "-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse", file_name]
    # end
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
# end
  # end


def _default_output_stem(data: "GDataState") -> str:
  """Best-effort output-file stem for a dataset with no explicit ``saveas``."""
  file_name = getattr(data, "_file_name", "") or ""
  if file_name:
    return os.path.basename(file_name).split(".")[0]
  # end
  label = data.get_label() if hasattr(data, "get_label") else ""
  return label or "plotly_output"
# end


def _write_plotly_output(fig, file_name: str, *, starting_azimuthal_angle: float,
    polar_angle: float, rotation_period: float, fps: int) -> str:
  """Save ``fig`` to ``file_name``, returning the (possibly extension-coerced) path.

  ``.mp4``/``.gif``/``.html`` rotate the camera on save (via
  :func:`save_rotating_plotly_figure`); any other extension -- or none --
  is coerced to a plain, non-rotating ``.html``.
  """
  root, ext = os.path.splitext(file_name)
  ext = ext.lower()
  if ext in (".mp4", ".gif", ".html"):
    save_rotating_plotly_figure(fig, file_name,
        starting_azimuthal_angle=starting_azimuthal_angle, polar_angle=polar_angle,
        rotation_period=rotation_period, fps=fps)
    return file_name
  # end
  file_name = f"{root}.html" if root else f"{file_name}.html"
  fig.write_html(file_name)
  return file_name
# end


def _preview_plotly_figure(fig, base_name: str, *, starting_azimuthal_angle: float,
    polar_angle: float, rotation_period: float, fps: int) -> str:
  """Write a temp, auto-rotating HTML preview of ``fig`` and return its path."""
  safe_base = "".join(
      ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base_name).strip("_")
  if not safe_base:
    safe_base = "plotly_preview"
  # end
  file_name = os.path.join(tempfile.gettempdir(), f"{safe_base}_preview.html")
  save_rotating_plotly_figure(fig, file_name,
      starting_azimuthal_angle=starting_azimuthal_angle, polar_angle=polar_angle,
      rotation_period=rotation_period, fps=fps)
  return file_name
# end


def open_preview(path: str) -> None:
  """Open a saved HTML file in the default web browser."""
  webbrowser.open(Path(path).resolve().as_uri())
# end


def _prepare_3d_coordinates(coords, value_shape):
  arrays = tuple(np.asarray(coord) for coord in coords)
  if len(arrays) != 3:
    raise ValueError("Plotly 3D plotting requires exactly three coordinate arrays")
  # end
  if all(array.ndim == 1 for array in arrays):
    mesh = np.meshgrid(*arrays, indexing="ij")
    return mesh[0], mesh[1], mesh[2]
  # end
  return arrays[0], arrays[1], arrays[2]
# end


def _prepare_2d_coordinates(coords, value_shape):
  arrays = tuple(np.asarray(coord) for coord in coords)
  if len(arrays) != 2:
    raise ValueError("Plotly surface plotting requires exactly two coordinate arrays")
  # end
  if all(array.ndim == 1 for array in arrays):
    mesh = np.meshgrid(*arrays, indexing="ij")
    return mesh[0], mesh[1]
  # end
  return arrays[0], arrays[1]
# end


def _scene_axis(label: str | None, log_axis: bool, axis_range, showgrid: bool,
    theme: dict) -> dict:
  """A themed Plotly 3-D scene axis dict, shared by the x/y/z axes."""
  return dict(
      title=dict(text=latex_to_html(label), font=dict(color=theme["text_color"])),
      showgrid=showgrid, type="log" if log_axis else "linear",
      exponentformat="e", range=axis_range, showbackground=True,
      backgroundcolor=theme["scene_color"], gridcolor=theme["grid_color"],
      linecolor=theme["axis_line_color"], tickfont=dict(color=theme["text_color"]),
      zerolinecolor=theme["grid_color"])
# end


def plotly(data: GDataState, *, squeeze: bool = False,
    num_subplot_row: int | None = None, num_subplot_col: int | None = None,
    scatter: bool = False, marker_radius: float = 4.0, markerstyle: str = "circle",
    diverging: bool = False,
    xscale: float = 1.0, xshift: float = 0.0,
    yscale: float = 1.0, yshift: float = 0.0,
    zscale: float = 1.0, zshift: float = 0.0,
    cmin: float | None = None, cmax: float | None = None,
    cscale: float = 1.0, cshift: float = 0.0,
    clim: tuple[float, float] | None = None,
    style: str | None = None,
    rcParams: Annotated[dict[str, object] | None,
        CliType(dict[str, str] | None), KeyValue()] = None,
    background: str = "dark", invert_cmap: bool = False,
    legend: bool = True, label_prefix: str = "", colorbar: bool = True,
    xlabel: str | None = None, ylabel: str | None = None,
    zlabel: str | None = None, clabel: str | None = None, title: str | None = None,
    logx: bool = False, logy: bool = False, logz: bool = False, logc: bool = False,
    aspect: Annotated[str | float | None, CliType(str | None)] = None,
    showgrid: bool = True, hashtag: bool = False, xkcd: bool = False,
    color: str | None = None, opacity: float | None = 1.0,
    scatter_opacity_range: tuple[float, float] | None = None,
    scatter_opacity_log: bool = False,
    maximum_points_per_axis: int = 0, surface_count: int = 32,
    xrange: tuple[float, float] | None = None,
    yrange: tuple[float, float] | None = None,
    zrange: tuple[float, float] | None = None,
    figsize: tuple[int, int] | None = None,
    cylindrical_to_cartesian: bool = False, cmap: str | None = None,
    save: bool = False, saveas: str | None = None, show: bool = False,
    azimuthal_angle: float = 0.0, polar_angle: float = 85.0,
    rotation_period: float = 40.0, fps: int = 1):
  """Render 2-D surface or 3-D volumetric data with Plotly.

  2-D data (``num_dims == 2``, after squeezing any size-1 axis) is drawn as
  a ``go.Surface`` (height map); 3-D data is drawn as a ``go.Volume`` or,
  with ``scatter=True``, a ``go.Scatter3d`` point cloud. Multi-component
  data lays out one scene per component unless ``squeeze`` is set.

  ``save``/``saveas``/``show`` make this call self-sufficient without any CLI
  glue: ``show=True`` opens an auto-rotating HTML preview in the browser;
  ``saveas`` (or ``save=True`` for an auto-derived name from ``data``'s
  source file) writes it instead -- ``.mp4``/``.gif``/``.html`` extensions
  get the rotating camera baked in (via :func:`save_rotating_plotly_figure`),
  any other extension a plain static ``.html``. If both a save and
  ``show=True`` are requested, the just-saved file is what opens (no
  separate preview render). All three default to inert (``show=False``,
  ``save=False``) -- a bare ``pg.load(f).interpolate().plotly()`` just
  builds and returns the figure, no file written and no browser opened;
  the CLI's ``--show/--no-show`` defaults to *on* by passing ``show=True``
  explicitly, since a human running it from a terminal does want to see
  something.

  Args:
    data: Point-value dataset to render.
    squeeze: Draw only the first component and collapse singleton axes.
    num_subplot_row: Forced row count for multi-component subplot layouts.
    num_subplot_col: Forced column count for multi-component subplot layouts.
    scatter: Draw 3-D data as a point cloud instead of a volume.
    marker_radius: Scatter-marker radius.
    markerstyle: Plotly scatter-marker symbol.
    diverging: Center a diverging color range on zero.
    xscale: Horizontal-coordinate scale factor.
    xshift: Horizontal-coordinate shift applied before scaling.
    yscale: Vertical-coordinate scale factor.
    yshift: Vertical-coordinate shift applied before scaling.
    zscale: Third-coordinate scale, or 2-D surface-height scale.
    zshift: Third-coordinate shift, or 2-D surface-height shift.
    cmin: Color-range lower bound.
    cmax: Color-range upper bound.
    cscale: Color-value scale factor.
    cshift: Color-value shift.
    clim: Explicit ``(minimum, maximum)`` color range.
    style: Postgkyl/Matplotlib style used to derive colors.
    rcParams: Matplotlib configuration overrides used while deriving styles.
    background: ``"dark"`` or ``"light"`` scene theme.
    invert_cmap: Reverse the selected colormap.
    legend: Show labeled traces in the legend.
    label_prefix: Prefix for component trace names.
    colorbar: Show the color bar.
    xlabel: Horizontal-axis label override.
    ylabel: Vertical-axis label override.
    zlabel: Third-axis label override.
    clabel: Color-bar label override.
    title: Figure-title override.
    logx: Use a logarithmic horizontal axis.
    logy: Use a logarithmic vertical axis.
    logz: Use a logarithmic third axis.
    logc: Use logarithmic color values.
    aspect: Scene aspect mode (``auto``, ``cube``, ``data``), or numeric ratio.
    showgrid: Draw scene grid lines.
    hashtag: Add a ``#pgkyl`` annotation.
    xkcd: Derive colors from Matplotlib's XKCD style.
    color: Replace the colormap with one fixed trace color.
    opacity: Surface, volume, or marker opacity.
    scatter_opacity_range: Minimum and maximum opacity encoded in scatter colors.
    scatter_opacity_log: Map scatter opacity logarithmically.
    maximum_points_per_axis: Downsample 3-D data to this many points per axis;
      zero disables downsampling.
    surface_count: Number of isosurfaces used by volume rendering.
    xrange: Explicit horizontal-axis range.
    yrange: Explicit vertical-axis range.
    zrange: Explicit third-axis range.
    figsize: Figure width and height in hundreds of pixels.
    cylindrical_to_cartesian: Interpret 3-D coordinates as ``(R, Z, phi)``.
    cmap: Matplotlib colormap name.
    save: Save using a name derived from the input dataset.
    saveas: Explicit output path.
    show: Open the saved or temporary HTML preview in a browser.
    azimuthal_angle: Initial camera azimuth for rotating output.
    polar_angle: Camera polar angle for rotating output.
    rotation_period: Seconds per camera revolution for animated output.
    fps: Frames per second for animated output.

  Returns:
    plotly.graph_objects.Figure: the assembled figure.
  """
  data = materialize_point_values(data)
  theme_colors = _apply_plot_style(style, rcParams, diverging, cmap, xkcd,
      background=background, invert_cmap=invert_cmap)

  grid, values = squeeze_collapsed_axes(list(data.grid), data.values)
  num_dims = len(grid)
  surface_mode = (num_dims == 2)
  if num_dims not in (2, 3):
    raise ValueError("plotly handles only 2D surface data or 3D volumetric data")
  # end
  if surface_mode and scatter:
    raise ValueError("Surface plots do not support scatter mode")
  # end

  # In surface mode the vertical axis is the function value, not a
  # coordinate; default its label to empty unless the caller overrode it.
  if surface_mode and zlabel is None:
    zlabel = " "
  # end
  xlabel, ylabel, zlabel, clabel = resolve_axis_labels(
      xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, clabel=clabel or "",
      num_dims=num_dims, xshift=xshift, yshift=yshift, zshift=zshift,
      xscale=xscale, yscale=yscale, zscale=zscale)

  num_comps = values.shape[-1]
  idx_comps = range(num_comps)

  if squeeze or num_comps == 1:
    fig = go.Figure()
    scene_names = ["scene"]
    grid_shape = (1, 1)
  # end
  else:
    num_rows, num_cols = subplot_grid(num_comps, num_subplot_row, num_subplot_col)
    specs = [[{"type": "scene"} for _ in range(num_cols)] for _ in range(num_rows)]
    fig = make_subplots(rows=num_rows, cols=num_cols, specs=specs)
    scene_names = ["scene" if idx == 0 else f"scene{idx + 1}" for idx in range(num_comps)]
    grid_shape = (num_rows, num_cols)
  # end

  colorscale = _plotly_colorscale(mpl.rcParams["image.cmap"])
  scalar_colorscale = [[0.0, color], [1.0, color]] if bool(color) else colorscale
  paper_color = theme_colors["paper_color"]
  scene_color = theme_colors["scene_color"]
  text_color = theme_colors["text_color"]

  fig.update_layout(paper_bgcolor=paper_color, plot_bgcolor=paper_color,
      font=dict(color=text_color))

  colorbar_kwargs = dict(
      title=dict(text=clabel or "", font=dict(color=text_color)),
      exponentformat="e", showexponent="all",
      tickfont=dict(color=text_color), bgcolor=paper_color)

  for comp_idx, comp in enumerate(idx_comps):
    if comp_idx >= len(scene_names):
      break
    # end
    scene_name = scene_names[comp_idx]
    row = 1 if grid_shape == (1, 1) else int(comp_idx / grid_shape[1]) + 1
    col = 1 if grid_shape == (1, 1) else int(comp_idx % grid_shape[1]) + 1
    label = f"{label_prefix:s}_c{comp:d}".strip("_") if len(idx_comps) > 1 else label_prefix
    cc_grid = nodal_to_cell_centered_grid(grid, values.shape[:num_dims])
    value = np.asarray(values[..., comp]) * zscale + zshift
    color_value = value * cscale + cshift
    render_color_value = np.array(color_value, copy=True)
    value_min, value_max = _finite_range(color_value)

    if surface_mode:
      x_grid, y_grid = _prepare_2d_coordinates(cc_grid, value.shape)
      x = (np.asarray(x_grid) + xshift) * xscale
      y = (np.asarray(y_grid) + yshift) * yscale
      z = np.asarray(value)
    # end
    else:
      x_grid, y_grid, z_grid = _prepare_3d_coordinates(cc_grid, value.shape)
      x_coord, y_coord, z_coord = (np.asarray(x_grid), np.asarray(y_grid),
                                    np.asarray(z_grid))
      if cylindrical_to_cartesian:
        # mapc2p cylindrical ordering is (R, Z, phi)
        r, z_cyl, phi = x_coord, y_coord, z_grid
        x_coord = r * np.cos(phi)
        y_coord = r * np.sin(phi)
        z_coord = z_cyl
      # end
      x = (x_coord + xshift) * xscale
      y = (y_coord + yshift) * yscale
      z = (z_coord + zshift) * zscale
    # end
    x_axis_range = _axis_range(x, xrange, logx)
    y_axis_range = _axis_range(y, yrange, logy)
    z_axis_range = _axis_range(z, zrange, logz)

    scene_aspectmode, scene_aspectratio = _resolve_plotly_aspect(aspect)
    scene = dict(
        xaxis=_scene_axis(xlabel, logx, x_axis_range, showgrid, theme_colors),
        yaxis=_scene_axis(ylabel, logy, y_axis_range, showgrid, theme_colors),
        zaxis=_scene_axis(zlabel, logz, z_axis_range, showgrid, theme_colors),
        bgcolor=scene_color, aspectmode=scene_aspectmode,
        aspectratio=scene_aspectratio)
    fig.update_layout(**{scene_name: scene})

    if diverging:
      cmax_val = float(np.nanmax(np.abs(color_value)))
      cmin_val = -cmax_val
    # end
    else:
      if clim is not None:
        cmin_local, cmax_local = clim
      # end
      else:
        cmin_local, cmax_local = cmin, cmax
      # end
      cmin_val = cmin_local if cmin_local is not None else value_min
      cmax_val = cmax_local if cmax_local is not None else value_max
    # end

    trace_colorscale = scalar_colorscale
    trace_colorbar_kwargs = dict(colorbar_kwargs)
    show_colorbar = colorbar and comp_idx == 0 and not bool(color)
    trace_name = label or f"c{comp}"
    show_trace_legend = legend and bool(label)

    if surface_mode:
      if logc:
        render_color_value, cmin_val, cmax_val = _apply_log_colorscale(
            render_color_value, cmin_val, cmax_val, trace_colorbar_kwargs)
      # end
      trace_list = [go.Surface(
          x=x, y=y, z=z, surfacecolor=render_color_value,
          colorscale=trace_colorscale, cmin=cmin_val, cmax=cmax_val,
          showscale=show_colorbar,
          colorbar=trace_colorbar_kwargs if show_colorbar else None,
          opacity=opacity, name=trace_name, showlegend=show_trace_legend)]
    # end
    else:
      if logz:
        positive = np.where(render_color_value > 0, render_color_value, np.nan)
        render_color_value = np.log10(positive)
        if cmin_val is not None:
          cmin_val = np.log10(max(cmin_val, np.finfo(float).tiny))
        # end
        if cmax_val is not None:
          cmax_val = np.log10(cmax_val)
        # end
      # end
      if logc:
        render_color_value, cmin_val, cmax_val = _apply_log_colorscale(
            render_color_value, cmin_val, cmax_val, trace_colorbar_kwargs)
      # end
      render_x, render_y, render_z, render_color_value = downsample(
          x, y, z, render_color_value,
          maximum_points_per_axis=maximum_points_per_axis)

      if scatter:
        marker_size = max(1.0, 2.0 * float(marker_radius))
        scatter_colorscale = trace_colorscale
        scatter_opacity = opacity
        if not bool(color) and scatter_opacity_range is not None:
          min_alpha, max_alpha = scatter_opacity_range
          scatter_colorscale = _opacity_mapping(trace_colorscale,
              min_alpha=min_alpha, max_alpha=max_alpha,
              log_scale=scatter_opacity_log)
          scatter_opacity = 1.0
        # end
        trace_list = [go.Scatter3d(
            x=render_x.ravel(), y=render_y.ravel(), z=render_z.ravel(),
            mode="markers",
            marker=dict(size=marker_size, symbol=markerstyle,
                color=render_color_value.ravel(), colorscale=scatter_colorscale,
                cmin=cmin_val, cmax=cmax_val, opacity=scatter_opacity,
                showscale=show_colorbar,
                colorbar=trace_colorbar_kwargs if show_colorbar else None),
            name=trace_name, showlegend=show_trace_legend)]
      # end
      else:
        volume_opacity_scale = [[0.0, 0.0], [0.5, 0.2], [1.0, 0.8]]
        trace_list = [go.Volume(
            x=render_x.ravel(), y=render_y.ravel(), z=render_z.ravel(),
            value=render_color_value.ravel(), colorscale=trace_colorscale,
            cmin=cmin_val, cmax=cmax_val, opacity=opacity,
            opacityscale=volume_opacity_scale, surface_count=surface_count,
            showscale=show_colorbar,
            colorbar=trace_colorbar_kwargs if show_colorbar else None,
            name=trace_name, showlegend=show_trace_legend)]
      # end
    # end

    for trace in trace_list:
      if grid_shape == (1, 1):
        fig.add_trace(trace)
      # end
      else:
        fig.add_trace(trace, row=row, col=col)
      # end
    # end
  # end

  if bool(title):
    fig.update_layout(title=title)
  # end
  if bool(hashtag):
    fig.add_annotation(text="#pgkyl", x=0.99, y=0.01, xref="paper", yref="paper",
        showarrow=False, xanchor="right", yanchor="bottom")
  # end
  if bool(figsize):
    fig.update_layout(width=figsize[0] * 100, height=figsize[1] * 100)
  # end
  fig.update_layout(margin=dict(l=10, r=10, t=40 if title else 10, b=10))

  output_path = None
  if save or saveas:
    output_path = _write_plotly_output(fig, saveas or _default_output_stem(data),
        starting_azimuthal_angle=azimuthal_angle, polar_angle=polar_angle,
        rotation_period=rotation_period, fps=fps)
  # end
  if show:
    if output_path is None:
      output_path = _preview_plotly_figure(fig, _default_output_stem(data),
          starting_azimuthal_angle=azimuthal_angle, polar_angle=polar_angle,
          rotation_period=rotation_period, fps=fps)
    # end
    open_preview(output_path)
  # end
  return fig
# end


command(CommandSpec(Section.RENDER, Execution.TERMINAL_EACH,
    result=ResultPolicy.SILENT))(plotly)


def plotly_animate(data_sequence: list["GDataState"],
    frame_labels: list[str] | None = None, frame_duration: int = 50,
    transition_duration: int = 0, fromcurrent: bool = True,
    redraw: bool = True, save: bool = False, saveas: str | None = None,
    show: bool = False, **plot_kwargs):
  """Build a Plotly animation figure from a sequence of datasets.

  Renders the first dataset with :func:`plotly` to create the base figure,
  then renders every subsequent dataset as an animation frame, wiring up
  Play/Pause buttons and a frame slider. All datasets must produce the same
  number of traces.

  Like :func:`plotly`, ``save``/``saveas``/``show`` are self-sufficient here
  with zero CLI glue: ``show=True`` opens the animation (a plain HTML preview
  -- the frame slider is its own scrubber, so unlike :func:`plotly` this
  never rotates the camera on save) in the browser; both default to inert,
  so a bare call just builds and returns the figure (see :func:`plotly`'s
  docstring for why). The per-frame :func:`plotly` calls below always render
  with ``save=False, show=False`` regardless of ``plot_kwargs``: only *this*
  function's own save/preview, on the assembled animation, should ever hit
  disk or a browser tab.
  """
  if not data_sequence:
    raise ValueError("plotly_animate requires at least one dataset")
  # end

  frame_plot_kwargs = dict(plot_kwargs, save=False, saveas=None, show=False)
  base_fig = plotly(data_sequence[0], **frame_plot_kwargs)
  num_traces = len(base_fig.data)

  if frame_labels is None:
    frame_labels = [str(idx) for idx in range(len(data_sequence))]
  # end
  if len(frame_labels) != len(data_sequence):
    raise ValueError("frame_labels length must match data_sequence length")
  # end

  frames = []
  for idx, dat in enumerate(data_sequence):
    if idx == 0:
      continue
    # end
    frame_fig = plotly(dat, **frame_plot_kwargs)
    if len(frame_fig.data) != num_traces:
      raise ValueError(
          "All animation frames must produce the same number of traces; "
          f"frame 0 has {num_traces:d}, frame {idx:d} has {len(frame_fig.data):d}.")
    # end
    frames.append(go.Frame(name=str(frame_labels[idx]), data=list(frame_fig.data),
        traces=list(range(num_traces))))
  # end

  base_fig.frames = frames

  animation_args = {"frame": {"duration": int(frame_duration), "redraw": bool(redraw)},
      "transition": {"duration": int(transition_duration)},
      "fromcurrent": bool(fromcurrent)}
  pause_args = {"frame": {"duration": 0, "redraw": bool(redraw)},
      "transition": {"duration": 0}, "mode": "immediate"}

  slider_steps = [{
      "label": str(label), "method": "animate",
      "args": [[str(label)], {"mode": "immediate",
          "frame": {"duration": int(frame_duration), "redraw": bool(redraw)},
          "transition": {"duration": int(transition_duration)}}],
  } for label in frame_labels]

  base_fig.update_layout(
      updatemenus=[{
          "type": "buttons", "showactive": False,
          "buttons": [
              {"label": "Play", "method": "animate", "args": [None, animation_args]},
              {"label": "Pause", "method": "animate", "args": [[None], pause_args]},
          ],
          "x": 0.02, "y": 0.0, "xanchor": "left", "yanchor": "bottom",
      }],
      sliders=[{"active": 0, "currentvalue": {"prefix": "Frame: "},
          "pad": {"t": 24}, "steps": slider_steps}],
  )

  output_path = None
  if save or saveas:
    out_name = saveas or "plotly_animate.html"
    if not str(out_name).lower().endswith(".html"):
      out_name = f"{out_name}.html"
    # end
    base_fig.write_html(out_name)
    output_path = out_name
  # end
  if show:
    if output_path is None:
      output_path = os.path.join(tempfile.gettempdir(), "plotly-animate_preview.html")
      base_fig.write_html(output_path)
    # end
    open_preview(output_path)
  # end
  return base_fig
# end


__all__ = ["plotly", "plotly_animate", "save_rotating_plotly_figure", "open_preview"]
