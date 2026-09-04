"""The canonical Matplotlib plot function and its private drawing helpers.

``pg.plot``, ``GData.plot``, ``operations.plot``, and the generated CLI are
aliases or lowerings of the one public function in this module. It owns every
plot option, dataset grouping, figure construction, saving, and display.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Annotated

import matplotlib as mpl
import matplotlib.font_manager as fm
import mpl_toolkits.mplot3d  # noqa: F401  (registers the '3d' projection)
import numpy as np
from matplotlib import cm, colors, patches
from matplotlib.figure import Figure
from matplotlib.typing import ColorType

from postgkyl.cli_spec import (
    CliType,
    CommandSpec,
    Execution,
    KeyValue,
    ResultPolicy,
    Section,
    command,
)
from postgkyl.gdatastate import (
    GDataState,
    GDataStateGroup,
    flatten_datasets,
    group_blocks,
    materialize_point_values,
)

from ._prep import subplot_grid
from .style import apply_style

_AXES_LABELS = [rf"$z_{i}$" for i in range(6)]
_INDEX_AXES_LABELS = [rf"$i_{i}$" for i in range(6)]
_OUTPUT_EXTENSIONS = (".png", ".pdf")
_AxisLimits = (tuple[float, float] | list[tuple[float, float]]
               | dict[int, tuple[float, float]])


def _indexed_saveas(saveas, index: int, indexed: bool):
  """Add a family index to every requested output path when needed."""
  if saveas is None or not indexed:
    return saveas
  if isinstance(saveas, (str, os.PathLike)):
    path = os.fspath(saveas)
    stem, extension = os.path.splitext(path)
    return f"{stem}_{index}{extension}"
  return tuple(_indexed_saveas(path, index, True) for path in saveas)


def _default_output_stem(states) -> str:
  """Best-effort output stem when ``save=True`` has no explicit path."""
  stems = []
  for i, data in enumerate(states):
    file_name = getattr(data, "_file_name", "") or ""
    if file_name:
      stem = os.path.basename(file_name).split(".")[0]
    else:
      label = data.get_label() if hasattr(data, "get_label") else ""
      stem = label.replace(" ", "_") if label else f"dataset_{i}"
    stems.append(stem)
  return "_".join(stems) or "matplotlib_output"


def _output_paths(save, saveas, states) -> tuple[str, ...]:
  """Normalize and validate Matplotlib output paths.

  A sequence is accepted so the CLI can preserve combinations such as
  ``--saveas plot.pdf --saveframes frame`` without saving outside the render
  backend. Extension-less names retain the historical PNG default.
  """
  empty_path = (isinstance(saveas, (str, os.PathLike))
                and not os.fspath(saveas))
  if saveas is None or empty_path:
    if not save:
      return ()
    paths = [_default_output_stem(states)]
  elif isinstance(saveas, (str, os.PathLike)):
    paths = [saveas]
  else:
    try:
      paths = list(saveas)
    except TypeError as err:
      raise TypeError(
          "'saveas' must be a path or an iterable of paths") from err

  normalized = []
  for path in paths:
    try:
      path = os.fspath(path)
    except TypeError as err:
      raise TypeError("every 'saveas' entry must be path-like") from err
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if not ext:
      path = f"{path}.png"
    elif ext not in _OUTPUT_EXTENSIONS:
      raise ValueError("Unsupported file format for saving. Supported formats "
                       "are: .png, .pdf")
    normalized.append(path)
  return tuple(normalized)


def _normalize_line_colors(color):
  """Return ``color`` as a per-line tuple, or ``None`` for a scalar color.

  Matplotlib color specifications such as RGB/RGBA tuples are sequences too,
  so test the complete value before interpreting it as a sequence of colors.
  """
  if color is None or colors.is_color_like(color):
    return None
  if isinstance(color, str):
    return None  # let Matplotlib report its usual error for an invalid color
  try:
    line_colors = tuple(color)
  except TypeError:
    return None
  if not line_colors:
    raise ValueError("'color' must not be an empty sequence")
  if not all(colors.is_color_like(line_color) for line_color in line_colors):
    raise ValueError(
        "every entry in a 'color' sequence must be a valid Matplotlib color")
  return line_colors


def _normalize_linestyles(linestyle, num_datasets: int):
  """Return one linestyle per dataset, or ``None`` for a scalar style."""
  if linestyle is None or isinstance(linestyle, str):
    return None
  # A Matplotlib custom dash pattern, e.g. ``(0, (5, 2))``, is one style
  # despite being a sequence itself.
  try:
    is_dash_pattern = (len(linestyle) == 2 and np.isscalar(linestyle[0])
                       and not isinstance(linestyle[1], str)
                       and all(np.isscalar(value) for value in linestyle[1]))
  except (TypeError, IndexError):
    is_dash_pattern = False
  if is_dash_pattern:
    return None
  try:
    linestyles = tuple(linestyle)
  except TypeError:
    return None
  if not linestyles:
    raise ValueError("'linestyle' must not be an empty sequence")
  if len(linestyles) == 1:
    return linestyles * num_datasets
  if len(linestyles) != num_datasets:
    raise ValueError(
        f"'linestyle' contains {len(linestyles)} entries; expected either 1 "
        f"(applied to every dataset) or {num_datasets} (one per dataset)")
  return linestyles


def _pgkyl_colorbar(im, fig, ax, *, label: str = "", extend: str | None = None):
  """The Postgkyl colorbar: appended beside ``ax`` (not shrinking it) via
  ``make_axes_locatable``, instead of stealing width from the panel."""
  from mpl_toolkits.axes_grid1 import make_axes_locatable

  divider = make_axes_locatable(ax)
  cax2 = divider.append_axes("right", size="3%", pad=0.05)
  return fig.colorbar(im, cax=cax2, label=label or "", extend=extend)


def get_xkcd_safely():
  """An xkcd context manager + rc override that degrades gracefully when no
  xkcd-style font is installed, instead of silently drawing with whatever
  default font Matplotlib falls back to."""
  import warnings

  import matplotlib.pyplot as plt

  required_fonts = {"xkcd", "xkcd Script", "Comic Neue", "Comic Sans MS"}
  available_fonts = {f.name for f in fm.fontManager.ttflist}
  if required_fonts.isdisjoint(available_fonts):
    warnings.warn(
        "No xkcd-style font found (xkcd/xkcd Script/Comic Neue/Comic Sans "
        "MS); falling back to the default sans-serif font.",
        stacklevel=2)
    font_rc = {"font.family": "sans-serif"}
  else:
    font_rc = {"font.family": "Comic Sans MS"}
  return plt.xkcd, font_rc


def _nodal_grid(grid: list, cells: np.ndarray) -> list:
  """Cell-center coordinates from nodal (edge) coordinate arrays.

  Handles both flat per-axis edge arrays and curvilinear (multi-dimensional,
  ``.map()``-produced) coordinate arrays, where every coordinate array spans
  all dimensions jointly.
  """
  num_dims = len(grid)
  if num_dims != len(cells):
    raise ValueError("Number dimensions for 'grid' and 'values' doesn't match")
  out = []
  for d in range(num_dims):
    g = grid[d]
    if g.ndim == 1:
      if g.shape[0] == cells[d]:
        out.append(g)
      elif g.shape[0] == cells[d] + 1:
        out.append(0.5 * (g[:-1] + g[1:]))
      else:
        raise ValueError("Something is terribly wrong...")
    else:
      if g.shape[d] == cells[d]:
        out.append(g)
      elif g.shape[d] == cells[d] + 1:
        if num_dims == 1:
          out.append(0.5 * (g[:-1] + g[1:]))
        else:
          out.append(0.5 * (g[:-1, :-1] + g[1:, 1:]))
      else:
        raise ValueError("Something is terribly wrong...")
  return out


def _shared_component_range(states, zshift: float, zscale: float) -> list:
  """Per-component ``(vmin, vmax)`` across *every* dataset drawn on one figure.

  When several 2-D datasets share one set of axes -- overlaid frames, or the
  blocks of a multiblock field, each covering its own patch of the domain --
  each ``pcolormesh`` would otherwise normalize against only its own values,
  so identical colors would mean different numbers in different patches and
  the single shared colorbar would be a lie. Computing the range up front
  makes one color scale describe the whole picture.

  Ranges are computed on the *plotted* values (``(v + zshift) * zscale``).
  A component that is all-NaN (or absent from a dataset with fewer
  components) yields ``(None, None)`` -- i.e. defer to Matplotlib.
  """
  ranges = []
  num_comps = max(int(state.values.shape[-1]) for state in states)
  for comp in range(num_comps):
    low, high = np.inf, -np.inf
    for state in states:
      values = state.values
      if comp >= values.shape[-1]:
        continue
      z = (values[..., comp] + zshift) * zscale
      if not np.any(np.isfinite(z)):
        continue
      low = min(low, float(np.nanmin(z)))
      high = max(high, float(np.nanmax(z)))
    ranges.append((low, high) if np.isfinite(low) and low < high else (None,
                                                                       None))
  return ranges


def _split_ylim_for_component(limits, comp: int):
  """Resolve a split-axis y-limit specification for one component.

  ``limits`` may be one ``(min, max)`` pair shared by every component, a
  sequence of pairs (one per component), or a mapping keyed by component
  index.  Missing sequence/mapping entries leave that component automatic.
  """
  if limits is None:
    return None
  if isinstance(limits, dict):
    limits = limits.get(comp)
    if limits is None:
      return None
  else:
    try:
      is_shared_pair = (len(limits) == 2
                        and all(value is None or np.isscalar(value)
                                for value in limits))
    except TypeError as err:
      raise TypeError("split y-limits must be a (min, max) pair, a sequence "
                      "of pairs, or a component-indexed mapping") from err
    if not is_shared_pair:
      if comp >= len(limits):
        return None
      limits = limits[comp]
      if limits is None:
        return None
  try:
    if len(limits) != 2:
      raise ValueError
  except (TypeError, ValueError) as err:
    raise ValueError("each split y-limit must be a (min, max) pair") from err
  return tuple(limits)


def plot(
    *datasets: GDataState | Iterable[GDataState],
    multiblock: bool = False,
    args: list[str] | None = None,
    figure: Annotated[int | str | Figure | None,
                      CliType(int | None)] = None,
    squeeze: bool = False,
    transpose: bool = False,
    grid_indices: bool = False,
    num_axes: int | None = None,
    start_axes: int = 0,
    overlay_axes: bool = False,
    num_subplot_row: int | None = None,
    num_subplot_col: int | None = None,
    streamline: bool = False,
    sdensity: int = 1,
    quiver: bool = False,
    contour: bool = False,
    clevels: str | None = None,
    cnlevels: int | None = None,
    cont_label: bool = False,
    surface: bool = False,
    comparison: bool = False,
    alpha: float | None = None,
    diverging: bool = False,
    lineouts: int | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
    xscale: float = 1.0,
    xshift: float = 0.0,
    ymin: float | None = None,
    ymax: float | None = None,
    yscale: float = 1.0,
    yshift: float = 0.0,
    zmin: float | None = None,
    zmax: float | None = None,
    zscale: float = 1.0,
    zshift: float = 0.0,
    relax: bool = False,
    style: str | None = None,
    rcParams: Annotated[dict[str, object] | None,
                        CliType(dict[str, str] | None),
                        KeyValue()] = None,
    no_legend: bool = False,
    legend_labels: list[str] | None = None,
    legend_subplot: int | None = None,
    legend_loc: Annotated[str | int | tuple[float, float],
                          CliType(str)] = "best",
    forcelegend: bool = False,
    no_colorbar: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    title: str | None = None,
    subplot_titles: str | None = None,
    subplot_xlabels: str | None = None,
    subplot_ylabels: str | None = None,
    logx: bool = False,
    logy: bool = False,
    logz: bool = False,
    split_linear_log: bool = False,
    split_point: float = 0.0,
    split_log_side: str = "right",
    split_width_ratios: tuple[float, float] = (1.0, 1.0),
    split_gap: float = 0.0,
    split_linear_ylim: Annotated[_AxisLimits | None,
                                 CliType(tuple[float, float] | None)] = None,
    split_log_ylim: Annotated[_AxisLimits | None,
                              CliType(tuple[float, float] | None)] = None,
    no_split_right_ticks: bool = False,
    split_legend_side: str = "log",
    split_log_base: float = 10.0,
    split_log_nonpositive: str = "clip",
    split_seam_ticklabels: str = "left",
    fixaspect: bool = False,
    aspect: float | None = None,
    edgecolors: str | None = None,
    no_showgrid: bool = False,
    hashtag: bool = False,
    xkcd: bool = False,
    color: Annotated[ColorType | Iterable[ColorType] | None,
                     CliType(str | None)] = None,
    markersize: float | None = None,
    linewidth: float | None = None,
    linestyle: Annotated[str | Iterable[str] | None,
                         CliType(str | None)] = None,
    figsize: Annotated[tuple[float, float] | str | None,
                       CliType(tuple[float, float] | None)] = None,
    jet: bool = False,
    cmap: str | None = None,
    cval: float | None = None,
    cval_min: float | None = None,
    cval_max: float | None = None,
    save: bool = False,
    saveas: Annotated[str | os.PathLike | Iterable[str | os.PathLike] | None,
                      CliType(str | None)] = None,
    dpi: int = 200,
    no_show: bool = False,
    clear: bool = False):
  """Plot one or more datasets onto a shared figure and return it.

  Accepts ``plot(a)`` or ``plot(a, b, multiblock=True)``. The first dataset sets
  the layout (dimensionality and panel count, after squeezing any size-1 axis
  left by a coordinate ``select()``); every dataset (including the first) is
  then drawn -- overlaid onto the same panels for 1-D, or onto the next
  ``start_axes``-offset block of panels when ``num_axes`` spreads multiple
  datasets' components across one grid (the old ``--subplots`` behaviour).
  Set ``overlay_axes=True`` to keep every dataset in the *same*
  ``start_axes`` block instead of advancing per dataset -- what the blocks of
  one multiblock field want, since they are one field and belong in one panel.

  When more than one 2-D dataset is drawn as a ``pcolormesh`` and no explicit
  ``zmin``/``zmax`` is given, all of them share one per-component color scale
  (computed across the whole call) and one colorbar per panel, so the
  colorbar describes every dataset on the axes rather than whichever was
  drawn last.

  Most of the keyword arguments mirror main's ``output.plot``/CLI ``plot``
  1:1 (contour/quiver/streamline/lineouts, shifts/scales, limits, labels,
  legend, colorbar, aspect, log axes, xkcd/hashtag/jet, style). ``transpose``
  swaps the horizontal and vertical axes: in 1-D the coordinate moves to the
  vertical axis; in 2-D the data, grid, and default labels are swapped before
  drawing (shifts/scales keep their screen-axis meaning). ``grid_indices``
  replaces each plotted coordinate with its zero-based sample index without
  changing the dataset itself. ``save``/``saveas``/
  ``no_show`` make the render call self-sufficient: ``saveas`` writes a PNG or
  PDF according to its extension (an extension-less name defaults to PNG),
  while ``save=True`` derives a PNG name from the input dataset. ``clear`` lets
  ``render.animate`` redraw onto a persistent figure across frames.
  A sequence of ``saveas`` paths writes the same figure to each path; this is
  primarily useful to CLI callers that request both a named output and frame
  output.

  For 1-D data, passing ``cmap`` together with ``cval`` colors the line by
  mapping ``cval`` onto the colormap; ``cval_min``/``cval_max`` set the
  normalization range (typically the min/max of the ``cval`` values across
  all curves), so several curves drawn into the same axes share one scale.
  ``color`` accepts either one Matplotlib color, applied to every line, or a
  sequence containing one color per dataset (reused for all its components).
  A sequence with one color per individual line is also accepted, in
  dataset/component order. ``linestyle`` similarly accepts one style applied
  to every dataset, or a sequence with one entry per dataset; a one-entry
  sequence is broadcast to every dataset.

  For 2-D data, ``surface`` draws a 3D surface instead of a ``pcolormesh``.
  When several 2-D datasets are overlaid onto the same axes for comparison,
  set ``comparison`` so each surface/contour gets a distinct color and a
  legend entry instead of overlapping and hiding each other. ``alpha``
  controls the surface transparency. Set ``legend_subplot`` to a zero-based
  subplot index to draw the legend only there; ``legend_loc`` accepts any
  Matplotlib legend location and defaults to ``"best"``. Explicit
  ``legend_labels`` are used verbatim on every component, without an added
  ``_cN`` suffix. ``xkcd`` no longer leaks into Matplotlib's global rcParams
  past this call -- it is scoped to the figure drawn here.

  ``split_linear_log=True`` turns every 1-D component panel into a joined
  pair split at ``split_point``: coordinates below the point are drawn on the
  left and coordinates at/above it on the right.  The right side is
  logarithmic in y by default; ``split_log_side='left'`` reverses which half
  is logarithmic.  ``split_width_ratios`` and ``split_gap`` control the pair's
  geometry.  ``split_linear_ylim`` and ``split_log_ylim`` accept either one
  ``(min, max)`` pair, a sequence of pairs (one per component), or a mapping
  from component index to pair.  ``split_legend_side`` is ``'linear'``,
  ``'log'``, ``'left'``, or ``'right'``.  This mode is intentionally limited
  to 1-D, non-transposed plots.  ``split_seam_ticklabels`` chooses which side
  owns the label at the joined boundary: ``'left'`` (the default),
  ``'right'``, ``'both'``, or ``'none'``.

  Args:
    datasets: Datasets to draw.
    multiblock: Force every dataset onto one figure instead of grouping fields.
    args: Positional Matplotlib plot arguments.
    figure: Existing or numbered figure to target.
    squeeze: Remove singleton spatial axes before laying out panels.
    transpose: Swap the horizontal and vertical display axes.
    grid_indices: Plot zero-based sample indices instead of grid values.
    num_axes: Number of logical component axes.
    start_axes: First logical component axis to use.
    overlay_axes: Draw every dataset in the same component-axis block.
    num_subplot_row: Forced subplot row count.
    num_subplot_col: Forced subplot column count.
    streamline: Draw two-component fields as streamlines.
    sdensity: Streamline density.
    quiver: Draw two-component fields as arrows.
    contour: Draw two-dimensional values as contours.
    clevels: Explicit contour-level specification.
    cnlevels: Number of contour levels.
    cont_label: Label contour lines.
    surface: Draw two-dimensional values as a three-dimensional surface.
    comparison: Distinguish overlaid two-dimensional datasets.
    alpha: Surface or comparison transparency.
    diverging: Use a diverging colormap.
    lineouts: Draw this many lineouts from two-dimensional data.
    xmin: Lower horizontal-axis bound.
    xmax: Upper horizontal-axis bound.
    xscale: Horizontal-coordinate scale factor.
    xshift: Horizontal-coordinate shift.
    ymin: Lower vertical-axis bound.
    ymax: Upper vertical-axis bound.
    yscale: Vertical-coordinate scale factor.
    yshift: Vertical-coordinate shift.
    zmin: Lower value or color bound.
    zmax: Upper value or color bound.
    zscale: Value scale factor.
    zshift: Value shift.
    relax: Allow relaxed layout behavior for reused figures.
    style: Matplotlib style name or style-file path.
    rcParams: Matplotlib configuration overrides.
    no_legend: Suppress legends for line plots.
    legend_labels: Explicit dataset legend labels.
    legend_subplot: Zero-based subplot receiving the legend.
    legend_loc: Matplotlib legend location.
    forcelegend: Draw a legend even for one unlabeled curve.
    no_colorbar: Suppress color bars for field plots.
    xlabel: Horizontal-axis label override.
    ylabel: Vertical-axis label override.
    clabel: Color-bar label override.
    title: Figure-title override.
    subplot_titles: Per-subplot title specification.
    subplot_xlabels: Per-subplot horizontal-label specification.
    subplot_ylabels: Per-subplot vertical-label specification.
    logx: Use logarithmic horizontal coordinates.
    logy: Use a logarithmic vertical axis.
    logz: Use logarithmic values or colors.
    split_linear_log: Split each one-dimensional panel into linear and log halves.
    split_point: Coordinate joining the split halves.
    split_log_side: Half using logarithmic scaling.
    split_width_ratios: Relative widths of the split halves.
    split_gap: Gap between split halves.
    split_linear_ylim: Limits for the linear half.
    split_log_ylim: Limits for the logarithmic half.
    no_split_right_ticks: Suppress ticks on the right edge of split panels.
    split_legend_side: Split half receiving the legend.
    split_log_base: Logarithm base for the logarithmic half.
    split_log_nonpositive: Handling of nonpositive logarithmic values.
    split_seam_ticklabels: Half owning labels at the split seam.
    fixaspect: Use equal physical scaling on coordinate axes.
    aspect: Explicit axes aspect ratio.
    edgecolors: Mesh edge color.
    no_showgrid: Suppress plot grid lines.
    hashtag: Prefix labels with a hash marker.
    xkcd: Draw using Matplotlib's XKCD context.
    color: Line color or per-line colors.
    markersize: Line-marker size.
    linewidth: Line width.
    linestyle: Line style or per-dataset styles.
    figsize: Figure width and height in inches.
    jet: Use the legacy jet colormap.
    cmap: Matplotlib colormap name.
    cval: Scalar used to color a one-dimensional curve.
    cval_min: Lower normalization bound for curve coloring.
    cval_max: Upper normalization bound for curve coloring.
    save: Save to an automatically derived output name.
    saveas: Explicit image output path or paths.
    dpi: Saved-image resolution.
    no_show: Do not display the figures interactively.
    clear: Clear a reused figure before drawing.

  Returns:
    One Matplotlib ``Figure``, or one figure per distinct field/frame family.

  Raises:
    ValueError: nothing to plot, a dataset has no values, a dataset has more
      than two (squeezed) dimensions, or (without ``squeeze``) a reused
      figure does not have enough axes for the panel count.
  """
  import matplotlib.pyplot as plt

  states = flatten_datasets(datasets)
  if not states:
    raise ValueError("nothing to plot")
  group_call = len(datasets) == 1 and isinstance(datasets[0], GDataStateGroup)
  families = ([states] if multiblock or group_call or figure is not None else
              group_blocks(states))
  indexed = len(families) > 1 or (save and saveas is not None and isinstance(
      saveas,
      (str, os.PathLike)) and not os.path.splitext(os.fspath(saveas))[1])
  figures = []
  plot_args = () if args is None else args
  for family_index, states in enumerate(families):
    states = [materialize_point_values(state) for state in states]
    family_saveas = _indexed_saveas(saveas, family_index, indexed)
    for st in states:
      if st.values is None:
        raise ValueError("dataset has no values to plot")

    line_colors = _normalize_line_colors(color)
    line_styles = _normalize_linestyles(linestyle, len(states))

    # ---- Style / global rcParams novelties ----
    apply_style(style) if style else apply_style("postgkyl")
    if rcParams:
      for key, value in rcParams.items():
        mpl.rcParams[key] = value
    if cmap:
      mpl.rcParams["image.cmap"] = cmap
    elif diverging:
      mpl.rcParams["image.cmap"] = "RdBu_r"
    if jet:  # not for general use -- only for comparing against literature
      mpl.rcParams["image.cmap"] = "jet"
    if xkcd:
      xkcd_cm, xkcd_rc = get_xkcd_safely()
    else:
      xkcd_cm, xkcd_rc = nullcontext, {}
    if color is not None and line_colors is None:
      mpl.rcParams["lines.color"] = color
    if linewidth:
      mpl.rcParams["lines.linewidth"] = linewidth
    if linestyle is not None and line_styles is None:
      mpl.rcParams["lines.linestyle"] = linestyle

    with xkcd_cm(), mpl.rc_context(rc=xkcd_rc):

      if not aspect:
        aspect = 1.0

      # ---- Phase 1: figure/axes layout, from the first dataset ----
      ref = states[0]
      ref_cells = ref.num_cells
      ref_num_dims = len(ref_cells) - int(np.sum(ref_cells <= 1))
      if ref_num_dims > 2:
        raise ValueError("Only 1D and 2D plots are currently supported")
      if line_colors is not None and ref_num_dims != 1:
        raise ValueError("a 'color' sequence is only supported for 1D plots")
      line_colors_by_dataset = False
      if line_colors is not None:
        component_step = 2 if (streamline or quiver) else 1
        expected_colors = sum(st.values.shape[-1] // component_step
                              for st in states)
        if len(line_colors) == len(states):
          line_colors_by_dataset = True
        elif len(line_colors) != expected_colors:
          raise ValueError(
              f"'color' contains {len(line_colors)} entries; expected either "
              f"{len(states)} (one per dataset) or {expected_colors} "
              "(one per line)")
      if split_linear_log:
        if ref_num_dims != 1:
          raise ValueError("'split_linear_log' is only supported for 1D plots")
        if transpose:
          raise ValueError(
              "'split_linear_log' cannot be combined with 'transpose'")
        if logy:
          raise ValueError("'logy' is redundant with 'split_linear_log'; use "
                           "'split_log_side' to choose the logarithmic half")
        if split_log_side not in {"left", "right"}:
          raise ValueError("'split_log_side' must be 'left' or 'right'")
        if split_legend_side not in {"linear", "log", "left", "right"}:
          raise ValueError("'split_legend_side' must be 'linear', 'log', "
                           "'left', or 'right'")
        if split_log_nonpositive not in {"clip", "mask"}:
          raise ValueError("'split_log_nonpositive' must be 'clip' or 'mask'")
        if split_seam_ticklabels not in {"left", "right", "both", "none"}:
          raise ValueError("'split_seam_ticklabels' must be 'left', 'right', "
                           "'both', or 'none'")
        try:
          split_point = float(split_point)
        except (TypeError, ValueError) as err:
          raise TypeError("'split_point' must be a finite number") from err
        if not np.isfinite(split_point):
          raise ValueError("'split_point' must be a finite number")
        try:
          split_log_base = float(split_log_base)
        except (TypeError, ValueError) as err:
          raise TypeError("'split_log_base' must be a number") from err
        if not np.isfinite(
            split_log_base) or split_log_base <= 0 or split_log_base == 1:
          raise ValueError(
              "'split_log_base' must be positive and not equal to 1")
        try:
          split_width_ratios = tuple(float(v) for v in split_width_ratios)
          if (len(split_width_ratios) != 2
              or any(not np.isfinite(v) or v <= 0 for v in split_width_ratios)):
            raise ValueError
        except (TypeError, ValueError) as err:
          raise ValueError(
              "'split_width_ratios' must contain two positive values") from err
        try:
          split_gap = float(split_gap)
        except (TypeError, ValueError) as err:
          raise TypeError("'split_gap' must be a number") from err
        if not np.isfinite(split_gap) or split_gap < 0:
          raise ValueError("'split_gap' must be non-negative")

      # Surface plots need 3D axes; only meaningful for 2D data.
      use_3d = bool(surface) and ref_num_dims == 2
      subplot_kw = {"projection": "3d"} if use_3d else {}

      coordinate_labels = _INDEX_AXES_LABELS if grid_indices else _AXES_LABELS
      default_xlabel, default_ylabel = coordinate_labels[0], coordinate_labels[
          1]
      if transpose and ref_num_dims == 2:
        # The data axes are swapped before drawing, so the default label base
        # names swap too; the shift/scale annotations below keep their
        # screen-axis meaning (xshift still shifts the horizontal axis).
        default_xlabel, default_ylabel = default_ylabel, default_xlabel
      layout_xlabel = xlabel
      layout_ylabel = ylabel
      layout_clabel = clabel
      if layout_xlabel is None:
        layout_xlabel = default_xlabel if lineouts != 1 else coordinate_labels[1]
        if xshift != 0.0 and xscale != 1.0:
          layout_xlabel = rf"({layout_xlabel:s} + {xshift:.2e}) $\times$ {xscale:.2e}"
        elif xshift != 0.0:
          layout_xlabel = rf"{layout_xlabel:s} + {xshift:.2e}"
        elif xscale != 1.0:
          layout_xlabel = rf"{layout_xlabel:s} $\times$ {xscale:.2e}"
      if layout_ylabel is None and ref_num_dims == 2 and lineouts is None:
        layout_ylabel = default_ylabel
        # NB: these elif conditions check xshift/xscale, not yshift/yscale --
        # a literal main bug (commands.plot's ylabel branch), kept for fidelity.
        if yshift != 0.0 and yscale != 1.0:
          layout_ylabel = rf"({layout_ylabel:s} + {yshift:.2e}) $\times$ {yscale:.2e}"
        elif xshift != 0.0:
          layout_ylabel = rf"{layout_ylabel:s} + {yshift:.2e}"
        elif xscale != 1.0:
          layout_ylabel = rf"{layout_ylabel:s} $\times$ {yscale:.2e}"
      if zscale != 1.0:
        layout_clabel = (rf"{layout_clabel:s} $\times$ {zscale:.3e}"
                         if layout_clabel else rf"$\times$ {zscale:.3e}")
      if transpose and ref_num_dims == 1:
        # The coordinate moves to the vertical axis, so the (resolved) labels
        # follow it -- including the shift/scale annotation, which travels with
        # the data it describes.
        layout_xlabel, layout_ylabel = layout_ylabel, layout_xlabel

      if isinstance(figsize, str):
        parts = figsize.split(",")
        figsize = (float(parts[0]), float(parts[1]))

      if figure is None:
        mpl_fig = plt.figure(figsize=figsize)
      elif isinstance(figure, int):
        mpl_fig = plt.figure(figure, figsize=figsize)
      elif isinstance(figure, mpl.figure.Figure):
        mpl_fig = figure
      elif isinstance(figure, str):
        mpl_fig = plt.figure(int(figure), figsize=figsize)
      else:
        raise TypeError(
            "'figure' keyword needs to be one of None (default), int, str, "
            "or a Matplotlib Figure")
      if clear:
        mpl_fig.clf()

      step = 2 if (streamline or quiver) else 1
      ref_idx_comps = range(int(np.floor(ref.num_comps / step)))
      layout_num_comps = num_axes if num_axes else len(ref_idx_comps)

      physical_num_axes = (2 * (1 if squeeze else layout_num_comps)
                           if split_linear_log else
                           (1 if squeeze else layout_num_comps))
      if mpl_fig.axes:
        ax = mpl_fig.axes
        if physical_num_axes > len(ax):
          raise ValueError("Trying to plot into figure with not enough axes")
      else:
        if split_linear_log:
          # Each logical component owns a nested 1x2 GridSpec.  Nesting keeps
          # the within-pair gap independent from spacing between components.
          logical_num_axes = 1 if squeeze else layout_num_comps
          num_rows, num_cols = ((1, 1) if squeeze else subplot_grid(
              logical_num_axes, num_subplot_row, num_subplot_col))
          outer = mpl_fig.add_gridspec(num_rows, num_cols)
          ax = []
          shared_left = None
          shared_right = None
          for logical_idx in range(logical_num_axes):
            row, col = divmod(logical_idx, num_cols)
            inner = outer[row, col].subgridspec(1,
                                                2,
                                                width_ratios=split_width_ratios,
                                                wspace=split_gap)
            left_ax = mpl_fig.add_subplot(inner[0], sharex=shared_left)
            right_ax = mpl_fig.add_subplot(inner[1], sharex=shared_right)
            if shared_left is None:
              shared_left, shared_right = left_ax, right_ax
            ax.extend((left_ax, right_ax))

          if title:
            mpl_fig.suptitle(title)
          if layout_xlabel:
            mpl_fig.supxlabel(layout_xlabel)
          if layout_ylabel:
            mpl_fig.supylabel(layout_ylabel)
          sub_titles = subplot_titles.split(",") if subplot_titles else []
          sub_xlabels = subplot_xlabels.split(",") if subplot_xlabels else []
          sub_ylabels = subplot_ylabels.split(",") if subplot_ylabels else []
          pair_center = (split_width_ratios[0] +
                         split_width_ratios[1]) / (2.0 * split_width_ratios[0])
          for logical_idx in range(logical_num_axes):
            left_ax, right_ax = ax[2 * logical_idx:2 * logical_idx + 2]
            sub_title = (sub_titles[logical_idx]
                         if logical_idx < len(sub_titles) else "")
            sub_xlabel = (sub_xlabels[logical_idx]
                          if logical_idx < len(sub_xlabels) else "")
            sub_ylabel = (sub_ylabels[logical_idx]
                          if logical_idx < len(sub_ylabels) else "")
            left_ax.set_ylabel(sub_ylabel)
            if sub_xlabel:
              left_ax.set_xlabel(sub_xlabel)
              left_ax.xaxis.set_label_coords(pair_center, -0.1)
            if sub_title:
              left_ax.set_title(sub_title, x=pair_center, y=1.08)
            if not no_split_right_ticks:
              right_ax.yaxis.tick_right()
              right_ax.yaxis.set_label_position("right")
        elif squeeze:  # Plotting into 1 panel
          mpl_fig.subplots(1, 1, subplot_kw=subplot_kw)
          ax = mpl_fig.axes
          ax[0].set_xlabel(layout_xlabel)
          ax[0].set_ylabel(layout_ylabel)
          if title is not None:
            ax[0].set_title(title, y=1.08)
        else:  # Plotting each component into its own subplot
          num_rows, num_cols = subplot_grid(layout_num_comps, num_subplot_row,
                                            num_subplot_col)
          if ref_num_dims == 1 or lineouts is not None:
            mpl_fig.subplots(num_rows,
                             num_cols,
                             sharex=True,
                             subplot_kw=subplot_kw)
          elif use_3d:  # 3D axes cannot share x/y with each other
            mpl_fig.subplots(num_rows, num_cols, subplot_kw=subplot_kw)
          else:  # In 2D, share y-axis as well
            mpl_fig.subplots(num_rows, num_cols, sharex=True, sharey=True)
          ax = mpl_fig.axes
          for extra in ax[layout_num_comps:]:
            extra.axis("off")
          if title:
            mpl_fig.suptitle(title)
          if layout_xlabel:
            mpl_fig.supxlabel(layout_xlabel)
          if layout_ylabel:
            mpl_fig.supylabel(layout_ylabel)

          for ax_idx in range(len(ax)):
            sub_titles = subplot_titles.split(",") if subplot_titles else []
            sub_xlabels = subplot_xlabels.split(",") if subplot_xlabels else []
            sub_ylabels = subplot_ylabels.split(",") if subplot_ylabels else []
            sub_title = sub_titles[ax_idx] if ax_idx < len(sub_titles) else ""
            sub_xlabel = sub_xlabels[ax_idx] if ax_idx < len(
                sub_xlabels) else ""
            sub_ylabel = sub_ylabels[ax_idx] if ax_idx < len(
                sub_ylabels) else ""
            ax[ax_idx].set_xlabel(sub_xlabel)
            ax[ax_idx].set_ylabel(sub_ylabel)
            if sub_title:
              ax[ax_idx].set_title(sub_title, y=1.08)

      # One color scale for every dataset drawn here (see
      # _shared_component_range). Only the plain 2-D pcolormesh path consumes
      # it: surface/contour/quiver/streamline/lineouts each own their own
      # normalization, and an explicit zmin/zmax always wins.
      shared_z = None
      if (len(states) > 1 and ref_num_dims == 2 and zmin is None
          and zmax is None
          and not (surface or contour or quiver or streamline or diverging)
          and lineouts is None):
        shared_z = _shared_component_range(states, zshift, zscale)

      if legend_subplot is not None:
        num_legend_subplots = 1 if squeeze else layout_num_comps
        if not isinstance(legend_subplot, int):
          raise TypeError("'legend_subplot' must be an integer or None")
        if legend_subplot < 0 or legend_subplot >= num_legend_subplots:
          raise ValueError(
              f"'legend_subplot' must be between 0 and {num_legend_subplots - 1}"
          )

      # ---- Phase 2: draw each dataset ----
      im = None
      cur_start_axes = start_axes
      line_color_idx = 0
      for ds_i, data in enumerate(states):
        if legend_labels is not None and ds_i < len(legend_labels):
          label_prefix = legend_labels[ds_i]
          explicit_legend_label = True
        elif len(states) > 1 or forcelegend:
          label_prefix = data.get_label()
          explicit_legend_label = False
        else:
          label_prefix = ""
          explicit_legend_label = False

        cells = data.num_cells
        grid = list(data.grid)
        values = data.values
        num_dims = len(cells) - int(np.sum(cells <= 1))
        if num_dims > 2:
          raise ValueError("Only 1D and 2D plots are currently supported")
        if split_linear_log and num_dims != 1:
          raise ValueError(
              "every dataset must be 1D when 'split_linear_log' is set")

        axes_labels = list(coordinate_labels)
        if len(grid) > num_dims:
          idx = [d for d in range(len(grid)) if cells[d] <= 1]
          grid = [g.squeeze() for g in grid]
          if idx:
            for d in reversed(idx):
              grid.pop(d)
            cells = np.delete(cells, idx)
            axes_labels = list(np.delete(np.array(axes_labels), idx))
            values = np.squeeze(values, tuple(idx))
            if grid and grid[0].ndim > 1:  # curvilinear (mapped) coordinates
              for d in range(num_dims):
                for i in reversed(idx):
                  grid[d] = np.mean(grid[d], axis=i)

        if transpose and num_dims == 2:  # swap the horizontal and vertical axes
          values = np.swapaxes(values, 0, 1)
          g0, g1 = grid[1], grid[0]
          if g0.ndim > 1:  # curvilinear coordinate arrays span both axes jointly
            g0, g1 = g0.transpose(), g1.transpose()
          grid[0], grid[1] = g0, g1
          cells = cells[[1,
                         0]]  # fancy indexing: num_cells may alias ctx["cells"]
          axes_labels[0], axes_labels[1] = axes_labels[1], axes_labels[0]

        if grid_indices:
          grid = [np.arange(int(num_cells)) for num_cells in cells]

        num_comps = values.shape[-1]
        idx_comps = range(int(np.floor(num_comps / step)))

        for comp in idx_comps:
          logical_ax_idx = 0 if squeeze else comp + cur_start_axes
          if split_linear_log:
            component_axes = ax[2 * logical_ax_idx:2 * logical_ax_idx + 2]
            cax = component_axes[0]
          else:
            cax = ax[logical_ax_idx]
            component_axes = [cax]
          comp_label = (label_prefix if explicit_legend_label else
                        (f"{label_prefix:s}_c{comp:d}".strip("_")
                         if len(idx_comps) > 1 else label_prefix))
          comp_legend = (not no_legend and
                         (legend_subplot is None or
                          (logical_ax_idx == legend_subplot
                           if split_linear_log else cax is ax[legend_subplot])))
          comp_colorbar = not no_colorbar

          if num_dims == 1:
            nodal_grid = _nodal_grid(grid, cells)
            x = (nodal_grid[0] + xshift) * xscale
            y = (values[..., comp] + yshift) * yscale
            if transpose:  # put the coordinate on the vertical axis
              x, y = y, x
            # Color the line from the colormap when a 'cval' is given (1D only).
            if line_colors is None:
              line_color = color
            elif line_colors_by_dataset:
              line_color = line_colors[ds_i]
            else:
              line_color = line_colors[line_color_idx]
            line_color_idx += 1
            if cmap and cval is not None:
              if cval_max is not None and cval_min is not None and cval_max != cval_min:
                t = (cval - cval_min) / (cval_max - cval_min)
              else:
                t = 0.5
              line_color = plt.get_cmap(cmap)(t)
            line_style = line_styles[ds_i] if line_styles is not None else None
            line_kwargs = dict(color=line_color,
                               label=comp_label,
                               markersize=markersize)
            if line_style is not None:
              line_kwargs["linestyle"] = line_style
            if split_linear_log:
              left_mask = x < split_point
              split_masks = (left_mask, ~left_mask)
              im = []
              for split_ax, mask in zip(component_axes, split_masks):
                im.extend(
                    split_ax.plot(x[mask], y[mask], *plot_args, **line_kwargs))
            else:
              im = cax.plot(x, y, *plot_args, **line_kwargs)
            # Add a colorbar describing the cval-to-color mapping once per axes.
            if (cmap and cval is not None and comp_colorbar
                and cval_max is not None and cval_min is not None
                and cval_max != cval_min
                and not getattr(cax, "_pgkyl_cval_cbar", False)):
              mappable = cm.ScalarMappable(norm=colors.Normalize(vmin=cval_min,
                                                                 vmax=cval_max),
                                           cmap=plt.get_cmap(cmap))
              _pgkyl_colorbar(mappable, mpl_fig, cax, label=layout_clabel)
              cax._pgkyl_cval_cbar = True

          elif num_dims == 2:
            extend = None

            if surface:  # ------------------------------------------------------
              nodal_grid = _nodal_grid(grid, cells)
              xg = (nodal_grid[0] + xshift) * xscale
              yg = (nodal_grid[1] + yshift) * yscale
              z = (values[..., comp].transpose() + zshift) * zscale
              if xg.ndim == 1:
                xg, yg = np.meshgrid(xg, yg)
              else:
                xg, yg = xg.transpose(), yg.transpose()
              # Count how many overlays already live on these axes so each gets
              # a distinct color (used for both surface and contour comparisons).
              overlay_count = getattr(cax, "_pgkyl_overlay_count", 0)
              cax._pgkyl_overlay_count = overlay_count + 1
              if comparison or bool(color):
                surf_color = color if bool(color) else f"C{overlay_count:d}"
                im = cax.plot_surface(xg,
                                      yg,
                                      z,
                                      color=surf_color,
                                      alpha=alpha if alpha is not None else 0.6,
                                      linewidth=0,
                                      antialiased=True,
                                      shade=True)
                if comp_label:
                  handles = getattr(cax, "_pgkyl_handles", [])
                  handles.append(
                      patches.Patch(color=surf_color, label=comp_label))
                  cax._pgkyl_handles = handles
              else:
                im = cax.plot_surface(xg,
                                      yg,
                                      z,
                                      cmap=mpl.rcParams["image.cmap"],
                                      alpha=alpha if alpha is not None else 1.0,
                                      linewidth=0,
                                      antialiased=True)
                if comp_colorbar:
                  mpl_fig.colorbar(im,
                                   ax=cax,
                                   label=layout_clabel or "",
                                   shrink=0.6,
                                   pad=0.1)
              if layout_clabel:
                cax.set_zlabel(layout_clabel)
              if zmin is not None or zmax is not None:
                cax.set_zlim(zmin, zmax)
              comp_colorbar = False

            elif contour:  # ------------------------------------------------------
              levels = 10
              if cnlevels:
                levels = int(cnlevels) - 1
              elif clevels:
                if ":" in clevels:
                  s = clevels.split(":")
                  levels = np.linspace(float(s[0]), float(s[1]), int(s[2]))
                else:
                  levels = np.array(clevels.split(","))
                  levels = np.array(list(filter(None, levels)))
              if isinstance(levels, np.ndarray) and len(levels) == 1:
                comp_colorbar = False
              nodal_grid = _nodal_grid(grid, cells)
              x = (nodal_grid[0] + xshift) * xscale
              y = (nodal_grid[1] + yshift) * yscale
              z = (values[..., comp].transpose() + zshift) * zscale
              cont_colors = color
              if comparison and not bool(color):
                # Give each overlaid dataset a distinct, single color + legend entry.
                overlay_count = getattr(cax, "_pgkyl_overlay_count", 0)
                cax._pgkyl_overlay_count = overlay_count + 1
                cont_colors = f"C{overlay_count:d}"
                if comp_label:
                  handles = getattr(cax, "_pgkyl_handles", [])
                  handles.append(
                      patches.Patch(color=cont_colors, label=comp_label))
                  cax._pgkyl_handles = handles
                comp_colorbar = False
              im = cax.contour(x,
                               y,
                               z,
                               levels,
                               *plot_args,
                               origin="lower",
                               colors=cont_colors,
                               linewidths=linewidth)
              if cont_label:
                cax.clabel(im, inline=1)

            elif quiver:  # -----------------------------------------------------
              skip = int(np.max((len(grid[0]), len(grid[1]))) // 15)
              skip2 = int(skip // 2)
              nodal_grid = _nodal_grid(grid, cells)
              if nodal_grid[0].ndim == 1:
                x = (nodal_grid[0][skip2::skip] + xshift) * xscale
                y = (nodal_grid[1][skip2::skip] + yshift) * yscale
              else:
                x = (nodal_grid[0][skip2::skip, skip2::skip] + xshift) * xscale
                y = (nodal_grid[1][skip2::skip, skip2::skip] + yshift) * yscale
              z1 = (values[skip2::skip, skip2::skip, 2 * comp].transpose() +
                    zshift) * zscale
              z2 = (values[skip2::skip, skip2::skip, 2 * comp + 1].transpose() +
                    zshift) * zscale
              im = cax.quiver(x, y, z1, z2)

            elif streamline:  # -------------------------------------------------
              if color:
                cl = color
              else:
                cl = np.sqrt(values[..., 2 * comp]**2 +
                             values[..., 2 * comp + 1]**2).transpose()
              nodal_grid = _nodal_grid(grid, cells)
              x = (nodal_grid[0] + xshift) * xscale
              y = (nodal_grid[1] + yshift) * yscale
              z1 = (values[..., 2 * comp].transpose() + zshift) * zscale
              z2 = (values[..., 2 * comp + 1].transpose() + zshift) * zscale
              im = cax.streamplot(x,
                                  y,
                                  z1,
                                  z2,
                                  *plot_args,
                                  density=sdensity,
                                  broken_streamlines=False,
                                  color=cl,
                                  linewidth=linewidth)

            elif lineouts is not None:  # ---------------------------------------
              num_lines = values.shape[1] if lineouts == 0 else values.shape[0]
              nodal_grid = _nodal_grid(grid, cells)

              if lineouts == 0:
                x = (nodal_grid[0] + xshift) * xscale
                line_vmin = (nodal_grid[1][0] + yshift) * yscale
                line_vmax = (nodal_grid[1][-1] + yshift) * yscale
                cbar_label = clabel or axes_labels[1]
              else:
                x = (nodal_grid[1] + xshift) * xscale
                line_vmin = (nodal_grid[0][0] + yshift) * yscale
                line_vmax = (nodal_grid[0][-1] + yshift) * yscale
                cbar_label = clabel or axes_labels[0]
              line_idx = [slice(0, u) for u in values.shape]
              line_idx[-1] = comp
              for line in range(num_lines):
                line_color = cm.inferno(line / (num_lines - 1))
                if lineouts == 0:
                  line_idx[1] = line
                else:
                  line_idx[0] = line
                y = (values[tuple(line_idx)] + yshift) * yscale
                im = cax.plot(x, y, *plot_args, color=line_color)
              mappable = cm.ScalarMappable(norm=colors.Normalize(vmin=line_vmin,
                                                                 vmax=line_vmax,
                                                                 clip=False),
                                           cmap=cm.inferno)
              _pgkyl_colorbar(mappable, mpl_fig, cax, label=cbar_label)
              comp_colorbar = False
              comp_legend = False

            else:  # ------------------------------------------------------------
              if zmin is not None and zmax is not None:
                extend = "both"
              elif zmax is not None:
                extend = "max"
              elif zmin is not None:
                extend = "min"
              x = (grid[0] + xshift) * xscale
              y = (grid[1] + yshift) * yscale
              z = (values[..., comp].transpose() + zshift) * zscale
              if len(x) == z.shape[1] or len(y) == z.shape[0]:
                nodal_grid = _nodal_grid(grid, cells)
                x = (nodal_grid[0] + xshift) * xscale
                y = (nodal_grid[1] + yshift) * yscale
              if x.ndim > 1:
                x, y = x.transpose(), y.transpose()
              comp_zmin, comp_zmax = zmin, zmax
              if diverging:
                comp_zmax = np.abs(z).max()
                comp_zmin = -comp_zmax
              elif shared_z is not None and comp < len(shared_z):
                comp_zmin, comp_zmax = shared_z[comp]
              vmax, vmin = comp_zmax, comp_zmin
              norm = None
              if logz:
                if diverging:
                  tmp = vmax / 1000
                  norm = colors.SymLogNorm(linthresh=tmp,
                                           linscale=tmp,
                                           vmin=vmin,
                                           vmax=vmax,
                                           base=10)
                else:
                  norm = colors.LogNorm(vmin=vmin, vmax=vmax)
                vmin, vmax = None, None
              im = cax.pcolormesh(x,
                                  y,
                                  z,
                                  norm=norm,
                                  vmin=vmin,
                                  vmax=vmax,
                                  edgecolors=edgecolors,
                                  linewidth=0.1,
                                  shading="auto",
                                  *plot_args)
            # One colorbar per (panel, component), not one per dataset drawn
            # into it: with several datasets on shared axes (multiblock blocks,
            # overlays) the per-dataset call used to stack an identical
            # colorbar per dataset, each shrinking the figure further. They
            # share one scale now, so the first describes them all. Keying on
            # the component too keeps ``squeeze``'s several components in one
            # panel getting their own (genuinely differently scaled) colorbars.
            drawn = getattr(cax, "_pgkyl_cbar_comps", None)
            if drawn is None:
              drawn = cax._pgkyl_cbar_comps = set()
            if not color and comp_colorbar and not streamline and comp not in drawn:
              _pgkyl_colorbar(im,
                              mpl_fig,
                              cax,
                              extend=extend,
                              label=layout_clabel)
              drawn.add(comp)
          else:
            raise ValueError(f"{num_dims:d}D data not supported")

          legend_ax = cax
          if split_linear_log:
            if split_legend_side == "left":
              legend_ax = component_axes[0]
            elif split_legend_side == "right":
              legend_ax = component_axes[1]
            elif split_legend_side == "linear":
              legend_ax = component_axes[0 if split_log_side == "right" else 1]
            else:  # log
              legend_ax = component_axes[0 if split_log_side == "left" else 1]
          if comp_legend:
            if getattr(legend_ax, "_pgkyl_handles", None):
              # Overlaid 2D datasets (surface/contour comparison): real legend.
              legend_ax.legend(handles=legend_ax._pgkyl_handles, loc=legend_loc)
            elif num_dims == 1 and comp_label != "":
              legend_ax.legend(loc=legend_loc)
            elif not (surface and num_dims == 2):
              legend_ax.text(0.03,
                             0.96,
                             comp_label,
                             bbox={
                                 "facecolor": "w",
                                 "edgecolor": "w",
                                 "alpha": 0.8,
                                 "boxstyle": "round"
                             },
                             verticalalignment="top",
                             horizontalalignment="left",
                             transform=legend_ax.transAxes)
          for side_idx, side_ax in enumerate(component_axes):
            side_ax.grid(not no_showgrid)
            if hashtag and (not split_linear_log or side_ax is legend_ax):
              side_ax.text(0.97,
                           0.03,
                           "#pgkyl",
                           bbox={
                               "facecolor": "w",
                               "edgecolor": "w",
                               "alpha": 0.8,
                               "boxstyle": "round"
                           },
                           verticalalignment="bottom",
                           horizontalalignment="right",
                           transform=side_ax.transAxes)
            if logx:
              side_ax.set_xscale("log")
            if logy:
              side_ax.set_yscale("log")
            if split_linear_log:
              is_log_side = ((side_idx == 0 and split_log_side == "left")
                             or (side_idx == 1 and split_log_side == "right"))
              if is_log_side:
                side_ax.set_yscale("log",
                                   base=split_log_base,
                                   nonpositive=split_log_nonpositive)
                side_ylim = _split_ylim_for_component(split_log_ylim, comp)
              else:
                side_ax.set_yscale("linear")
                side_ylim = _split_ylim_for_component(split_linear_ylim, comp)
            else:
              side_ylim = None
            if num_dims == 1 and not relax:  # this causes troubles with contours
              side_ax.autoscale(enable=True, axis="x", tight=True)
              side_ax.autoscale(enable=True, axis="y")
            if split_linear_log:
              if side_idx == 0:
                side_ax.set_xlim(xmin, split_point)
              else:
                side_ax.set_xlim(split_point, xmax)
              if not logx:
                prune = None
                if ((split_seam_ticklabels == "left" and side_idx == 1)
                    or (split_seam_ticklabels == "right" and side_idx == 0)
                    or split_seam_ticklabels == "none"):
                  prune = "upper" if side_idx == 0 else "lower"
                if prune is not None:
                  side_ax.xaxis.get_major_locator().set_params(prune=prune)
            elif xmin is not None or xmax is not None:
              side_ax.set_xlim(xmin, xmax)
            if ymin is not None or ymax is not None:
              side_ax.set_ylim(ymin, ymax)
            if side_ylim is not None:
              side_ax.set_ylim(*side_ylim)
            if fixaspect and not (surface and num_dims == 2):
              plt.setp(side_ax, aspect=aspect)

        if num_axes and not overlay_axes:
          cur_start_axes += num_comps

      mpl_fig.tight_layout()
      for output_path in _output_paths(save, family_saveas, states):
        mpl_fig.savefig(output_path, dpi=dpi)
    figures.append(mpl_fig)
  if not no_show:
    plt.show()
  return figures[0] if len(figures) == 1 else figures


command(
    CommandSpec(Section.RENDER,
                Execution.TERMINAL_ALL,
                result=ResultPolicy.SILENT))(plot)
