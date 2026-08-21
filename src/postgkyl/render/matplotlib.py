"""Matplotlib rendering backend -- a faithful port of main's plotting engine.

This is main's ``commands/plot.py`` + ``output/plot.py`` (the CLI's per-dataset
render call) ported onto ``GDataState`` in place of the old ``GData``/tuple
dispatch (``utils.input_parser``). Supports everything the old engine did: 1-D
lines, 2-D pcolormesh/contour/quiver/streamline/lineouts, one sub-panel per
component (near-square grid, or forced rows/cols), log axes, the Postgkyl
colorbar, xkcd/hashtag/jet novelties, and per-axis shift/scale/limits.

``plot(*datasets, ...)`` generalizes the old single-dataset call to one-or-more
datasets sharing one figure: the layout (dimensionality, panel count, default
labels) comes from the first dataset -- exactly what main's CLI achieved by
repeating its single-dataset call onto a figure whose axes already exist (see
``cli/commands/plot.py``, which calls this once per active dataset, targeting
a shared or fresh figure exactly as main's loop did). ``show``/``fig`` are the
only render-time conveniences this layer still owns; save/saveframes/batch
file-naming stay a CLI concern, as they were in main.
"""

from __future__ import annotations

from contextlib import nullcontext

import matplotlib as mpl
import matplotlib.font_manager as fm
import mpl_toolkits.mplot3d  # noqa: F401  (registers the '3d' projection)
import numpy as np
from matplotlib import cm, colors, patches

from postgkyl.gdatastate import flatten_datasets

from ._prep import subplot_grid
from .style import apply_style

_AXES_LABELS = [rf"$z_{i}$" for i in range(6)]


def _pgkyl_colorbar(im, fig, ax, *, label: str = "", extend: str | None = None):
  """The Postgkyl colorbar: appended beside ``ax`` (not shrinking it) via
  ``make_axes_locatable``, instead of stealing width from the panel."""
  from mpl_toolkits.axes_grid1 import make_axes_locatable

  divider = make_axes_locatable(ax)
  cax2 = divider.append_axes("right", size="3%", pad=0.05)
  return fig.colorbar(im, cax=cax2, label=label or "", extend=extend)
# end


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
        "MS); falling back to the default sans-serif font.", stacklevel=2)
    font_rc = {"font.family": "sans-serif"}
  # end
  else:
    font_rc = {"font.family": "Comic Sans MS"}
  # end
  return plt.xkcd, font_rc
# end


def _nodal_grid(grid: list, cells: np.ndarray) -> list:
  """Cell-center coordinates from nodal (edge) coordinate arrays.

  Handles both flat per-axis edge arrays and curvilinear (multi-dimensional,
  ``.map()``-produced) coordinate arrays, where every coordinate array spans
  all dimensions jointly.
  """
  num_dims = len(grid)
  if num_dims != len(cells):
    raise ValueError("Number dimensions for 'grid' and 'values' doesn't match")
  # end
  out = []
  for d in range(num_dims):
    g = grid[d]
    if g.ndim == 1:
      if g.shape[0] == cells[d]:
        out.append(g)
      # end
      elif g.shape[0] == cells[d] + 1:
        out.append(0.5 * (g[:-1] + g[1:]))
      # end
      else:
        raise ValueError("Something is terribly wrong...")
    # end
      # end
    else:
      if g.shape[d] == cells[d]:
        out.append(g)
      # end
      elif g.shape[d] == cells[d] + 1:
        if num_dims == 1:
          out.append(0.5 * (g[:-1] + g[1:]))
        # end
        else:
          out.append(0.5 * (g[:-1, :-1] + g[1:, 1:]))
      # end
        # end
      else:
        raise ValueError("Something is terribly wrong...")
      # end
    # end
  # end
  return out
# end


def _split_ylim_for_component(limits, comp: int):
  """Resolve a split-axis y-limit specification for one component.

  ``limits`` may be one ``(min, max)`` pair shared by every component, a
  sequence of pairs (one per component), or a mapping keyed by component
  index.  Missing sequence/mapping entries leave that component automatic.
  """
  if limits is None:
    return None
  # end
  if isinstance(limits, dict):
    limits = limits.get(comp)
    if limits is None:
      return None
    # end
  # end
  else:
    try:
      is_shared_pair = (len(limits) == 2
          and all(value is None or np.isscalar(value) for value in limits))
    except TypeError as err:
      raise TypeError("split y-limits must be a (min, max) pair, a sequence "
          "of pairs, or a component-indexed mapping") from err
    # end
    if not is_shared_pair:
      if comp >= len(limits):
        return None
      # end
      limits = limits[comp]
      if limits is None:
        return None
      # end
  # end
  try:
    if len(limits) != 2:
      raise ValueError
    # end
  except (TypeError, ValueError) as err:
    raise ValueError("each split y-limit must be a (min, max) pair") from err
  # end
  return tuple(limits)
# end


def plot(*datasets, args: str = "", figure=None, squeeze: bool = False,
    transpose: bool = False, num_axes: int | None = None, start_axes: int = 0,
    num_subplot_row: int | None = None, num_subplot_col: int | None = None,
    streamline: bool = False, sdensity: int = 1, quiver: bool = False,
    contour: bool = False, clevels: str | None = None,
    cnlevels: int | None = None, cont_label: bool = False,
    surface: bool = False, comparison: bool = False, alpha: float | None = None,
    diverging: bool = False, lineouts: int | None = None,
    xmin: float | None = None, xmax: float | None = None,
    xscale: float = 1.0, xshift: float = 0.0,
    ymin: float | None = None, ymax: float | None = None,
    yscale: float = 1.0, yshift: float = 0.0,
    zmin: float | None = None, zmax: float | None = None,
    zscale: float = 1.0, zshift: float = 0.0,
    relax: bool = False, style: str | None = None, rcParams: dict | None = None,
    legend: bool = True, legend_labels: list | None = None,
    legend_subplot: int | None = None,
    legend_loc: str | int | tuple = "best",
    forcelegend: bool = False,
    colorbar: bool = True,
    xlabel: str | None = None, ylabel: str | None = None,
    clabel: str | None = None, title: str | None = None,
    subplot_titles: str | None = None, subplot_xlabels: str | None = None,
    subplot_ylabels: str | None = None,
    logx: bool = False, logy: bool = False, logz: bool = False,
    split_linear_log: bool = False, split_point: float = 0.0,
    split_log_side: str = "right", split_width_ratios=(1.0, 1.0),
    split_gap: float = 0.0, split_linear_ylim=None, split_log_ylim=None,
    split_right_ticks: bool = True, split_legend_side: str = "log",
    split_log_base: float = 10.0, split_log_nonpositive: str = "clip",
    split_seam_ticklabels: str = "left",
    fixaspect: bool = False, aspect=None,
    edgecolors: str | None = None, showgrid: bool = True,
    hashtag: bool = False, xkcd: bool = False,
    color: str | None = None, markersize: float | None = None,
    linewidth: float | None = None, linestyle: str | None = None,
    figsize=None, jet: bool = False, cmap: str | None = None,
    cval: float | None = None, cval_min: float | None = None,
    cval_max: float | None = None,
    show: bool = True, fig=None):
  """Plot one or more datasets onto a shared figure and return it.

  Accepts ``plot(a)`` (main's single-dataset call) or ``plot(a, b)`` (what
  main's CLI achieved by repeating the single-dataset call onto a figure whose
  axes already exist -- see ``cli/commands/plot.py``). The first dataset sets
  the layout (dimensionality and panel count, after squeezing any size-1 axis
  left by a coordinate ``select()``); every dataset (including the first) is
  then drawn -- overlaid onto the same panels for 1-D, or onto the next
  ``start_axes``-offset block of panels when ``num_axes`` spreads multiple
  datasets' components across one grid (the old ``--subplots`` behaviour).

  Most of the keyword arguments mirror main's ``output.plot``/CLI ``plot``
  1:1 (contour/quiver/streamline/lineouts, shifts/scales, limits, labels,
  legend, colorbar, aspect, log axes, xkcd/hashtag/jet, style). ``transpose``
  swaps the horizontal and vertical axes: in 1-D the coordinate moves to the
  vertical axis; in 2-D the data, grid, and default labels are swapped before
  drawing (shifts/scales keep their screen-axis meaning). ``show``
  and ``fig`` are new-era conveniences: ``fig`` lets ``render.animate``
  redraw onto a persistent (cleared) figure across frames; save/saveframes/
  batch file-naming remain a CLI-layer concern, matching main.

  For 1-D data, passing ``cmap`` together with ``cval`` colors the line by
  mapping ``cval`` onto the colormap; ``cval_min``/``cval_max`` set the
  normalization range (typically the min/max of the ``cval`` values across
  all curves), so several curves drawn into the same axes share one scale.

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

  Returns:
    The Matplotlib ``Figure``.

  Raises:
    ValueError: nothing to plot, a dataset has no values, a dataset has more
      than two (squeezed) dimensions, or (without ``squeeze``) a reused
      figure does not have enough axes for the panel count.
  """
  import matplotlib.pyplot as plt

  states = flatten_datasets(datasets)
  if not states:
    raise ValueError("nothing to plot")
  # end
  for st in states:
    if st.values is None:
      raise ValueError("dataset has no values to plot")
    # end
  # end

  # ---- Style / global rcParams novelties ----
  apply_style(style) if style else apply_style("postgkyl")
  if rcParams:
    for key, value in rcParams.items():
      mpl.rcParams[key] = value
    # end
  # end
  if cmap:
    mpl.rcParams["image.cmap"] = cmap
  # end
  elif diverging:
    mpl.rcParams["image.cmap"] = "RdBu_r"
  # end
  if jet:  # not for general use -- only for comparing against literature
    mpl.rcParams["image.cmap"] = "jet"
  # end
  if xkcd:
    xkcd_cm, xkcd_rc = get_xkcd_safely()
  # end
  else:
    xkcd_cm, xkcd_rc = nullcontext, {}
  # end
  if color:
    mpl.rcParams["lines.color"] = color
  # end
  if linewidth:
    mpl.rcParams["lines.linewidth"] = linewidth
  # end
  if linestyle:
    mpl.rcParams["lines.linestyle"] = linestyle
  # end

  with xkcd_cm(), mpl.rc_context(rc=xkcd_rc):

    if not aspect:
      aspect = 1.0
    # end

    # ---- Phase 1: figure/axes layout, from the first dataset ----
    ref = states[0]
    ref_cells = ref.num_cells
    ref_num_dims = len(ref_cells) - int(np.sum(ref_cells <= 1))
    if ref_num_dims > 2:
      raise ValueError("Only 1D and 2D plots are currently supported")
    # end
    if split_linear_log:
      if ref_num_dims != 1:
        raise ValueError("'split_linear_log' is only supported for 1D plots")
      # end
      if transpose:
        raise ValueError("'split_linear_log' cannot be combined with 'transpose'")
      # end
      if logy:
        raise ValueError("'logy' is redundant with 'split_linear_log'; use "
            "'split_log_side' to choose the logarithmic half")
      # end
      if split_log_side not in {"left", "right"}:
        raise ValueError("'split_log_side' must be 'left' or 'right'")
      # end
      if split_legend_side not in {"linear", "log", "left", "right"}:
        raise ValueError("'split_legend_side' must be 'linear', 'log', "
            "'left', or 'right'")
      # end
      if split_log_nonpositive not in {"clip", "mask"}:
        raise ValueError("'split_log_nonpositive' must be 'clip' or 'mask'")
      # end
      if split_seam_ticklabels not in {"left", "right", "both", "none"}:
        raise ValueError("'split_seam_ticklabels' must be 'left', 'right', "
            "'both', or 'none'")
      # end
      try:
        split_point = float(split_point)
      except (TypeError, ValueError) as err:
        raise TypeError("'split_point' must be a finite number") from err
      # end
      if not np.isfinite(split_point):
        raise ValueError("'split_point' must be a finite number")
      # end
      try:
        split_log_base = float(split_log_base)
      except (TypeError, ValueError) as err:
        raise TypeError("'split_log_base' must be a number") from err
      # end
      if not np.isfinite(split_log_base) or split_log_base <= 0 or split_log_base == 1:
        raise ValueError("'split_log_base' must be positive and not equal to 1")
      # end
      try:
        split_width_ratios = tuple(float(v) for v in split_width_ratios)
        if (len(split_width_ratios) != 2 or any(not np.isfinite(v) or v <= 0
            for v in split_width_ratios)):
          raise ValueError
        # end
      except (TypeError, ValueError) as err:
        raise ValueError("'split_width_ratios' must contain two positive values") from err
      # end
      try:
        split_gap = float(split_gap)
      except (TypeError, ValueError) as err:
        raise TypeError("'split_gap' must be a number") from err
      # end
      if not np.isfinite(split_gap) or split_gap < 0:
        raise ValueError("'split_gap' must be non-negative")
      # end
    # end

    # Surface plots need 3D axes; only meaningful for 2D data.
    use_3d = bool(surface) and ref_num_dims == 2
    subplot_kw = {"projection": "3d"} if use_3d else {}

    default_xlabel, default_ylabel = _AXES_LABELS[0], _AXES_LABELS[1]
    if transpose and ref_num_dims == 2:
      # The data axes are swapped before drawing, so the default label base
      # names swap too; the shift/scale annotations below keep their
      # screen-axis meaning (xshift still shifts the horizontal axis).
      default_xlabel, default_ylabel = default_ylabel, default_xlabel
    # end
    layout_xlabel = xlabel
    layout_ylabel = ylabel
    layout_clabel = clabel
    if layout_xlabel is None:
      layout_xlabel = default_xlabel if lineouts != 1 else _AXES_LABELS[1]
      if xshift != 0.0 and xscale != 1.0:
        layout_xlabel = rf"({layout_xlabel:s} + {xshift:.2e}) $\times$ {xscale:.2e}"
      # end
      elif xshift != 0.0:
        layout_xlabel = rf"{layout_xlabel:s} + {xshift:.2e}"
      # end
      elif xscale != 1.0:
        layout_xlabel = rf"{layout_xlabel:s} $\times$ {xscale:.2e}"
      # end
    # end
    if layout_ylabel is None and ref_num_dims == 2 and lineouts is None:
      layout_ylabel = default_ylabel
      # NB: these elif conditions check xshift/xscale, not yshift/yscale --
      # a literal main bug (commands.plot's ylabel branch), kept for fidelity.
      if yshift != 0.0 and yscale != 1.0:
        layout_ylabel = rf"({layout_ylabel:s} + {yshift:.2e}) $\times$ {yscale:.2e}"
      # end
      elif xshift != 0.0:
        layout_ylabel = rf"{layout_ylabel:s} + {yshift:.2e}"
      # end
      elif xscale != 1.0:
        layout_ylabel = rf"{layout_ylabel:s} $\times$ {yscale:.2e}"
      # end
    # end
    if zscale != 1.0:
      layout_clabel = (rf"{layout_clabel:s} $\times$ {zscale:.3e}" if layout_clabel
                        else rf"$\times$ {zscale:.3e}")
    # end
    if transpose and ref_num_dims == 1:
      # The coordinate moves to the vertical axis, so the (resolved) labels
      # follow it -- including the shift/scale annotation, which travels with
      # the data it describes.
      layout_xlabel, layout_ylabel = layout_ylabel, layout_xlabel
    # end

    if isinstance(figsize, str):
      parts = figsize.split(",")
      figsize = (float(parts[0]), float(parts[1]))
    # end

    if fig is not None:
      mpl_fig = fig
      mpl_fig.clf()
    # end
    elif figure is None:
      mpl_fig = plt.figure(figsize=figsize)
    # end
    elif isinstance(figure, int):
      mpl_fig = plt.figure(figure, figsize=figsize)
    # end
    elif isinstance(figure, mpl.figure.Figure):
      mpl_fig = figure
    # end
    elif isinstance(figure, str):
      mpl_fig = plt.figure(int(figure), figsize=figsize)
    # end
    else:
      raise TypeError(
          "'figure' keyword needs to be one of None (default), int, str, "
          "or a Matplotlib Figure")
    # end

    step = 2 if (streamline or quiver) else 1
    ref_idx_comps = range(int(np.floor(ref.num_comps / step)))
    layout_num_comps = num_axes if num_axes else len(ref_idx_comps)

    physical_num_axes = (2 * (1 if squeeze else layout_num_comps)
        if split_linear_log else (1 if squeeze else layout_num_comps))
    if mpl_fig.axes:
      ax = mpl_fig.axes
      if physical_num_axes > len(ax):
        raise ValueError("Trying to plot into figure with not enough axes")
      # end
      # end
    else:
      if split_linear_log:
        # Each logical component owns a nested 1x2 GridSpec.  Nesting keeps
        # the within-pair gap independent from spacing between components.
        logical_num_axes = 1 if squeeze else layout_num_comps
        num_rows, num_cols = ((1, 1) if squeeze else
            subplot_grid(logical_num_axes, num_subplot_row, num_subplot_col))
        outer = mpl_fig.add_gridspec(num_rows, num_cols)
        ax = []
        shared_left = None
        shared_right = None
        for logical_idx in range(logical_num_axes):
          row, col = divmod(logical_idx, num_cols)
          inner = outer[row, col].subgridspec(1, 2,
              width_ratios=split_width_ratios, wspace=split_gap)
          left_ax = mpl_fig.add_subplot(inner[0], sharex=shared_left)
          right_ax = mpl_fig.add_subplot(inner[1], sharex=shared_right)
          if shared_left is None:
            shared_left, shared_right = left_ax, right_ax
          # end
          ax.extend((left_ax, right_ax))
        # end

        if title:
          mpl_fig.suptitle(title)
        # end
        if layout_xlabel:
          mpl_fig.supxlabel(layout_xlabel)
        # end
        if layout_ylabel:
          mpl_fig.supylabel(layout_ylabel)
        # end
        sub_titles = subplot_titles.split(",") if subplot_titles else []
        sub_xlabels = subplot_xlabels.split(",") if subplot_xlabels else []
        sub_ylabels = subplot_ylabels.split(",") if subplot_ylabels else []
        pair_center = (split_width_ratios[0] + split_width_ratios[1]) / (
            2.0 * split_width_ratios[0])
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
          # end
          if sub_title:
            left_ax.set_title(sub_title, x=pair_center, y=1.08)
          # end
          if split_right_ticks:
            right_ax.yaxis.tick_right()
            right_ax.yaxis.set_label_position("right")
          # end
        # end
      # end
      elif squeeze:  # Plotting into 1 panel
        mpl_fig.subplots(1, 1, subplot_kw=subplot_kw)
        ax = mpl_fig.axes
        ax[0].set_xlabel(layout_xlabel)
        ax[0].set_ylabel(layout_ylabel)
        if title is not None:
          ax[0].set_title(title, y=1.08)
      # end
        # end
      else:  # Plotting each component into its own subplot
        num_rows, num_cols = subplot_grid(layout_num_comps, num_subplot_row,
            num_subplot_col)
        if ref_num_dims == 1 or lineouts is not None:
          mpl_fig.subplots(num_rows, num_cols, sharex=True, subplot_kw=subplot_kw)
        # end
        elif use_3d:  # 3D axes cannot share x/y with each other
          mpl_fig.subplots(num_rows, num_cols, subplot_kw=subplot_kw)
        # end
        else:  # In 2D, share y-axis as well
          mpl_fig.subplots(num_rows, num_cols, sharex=True, sharey=True)
        # end
        ax = mpl_fig.axes
        for extra in ax[layout_num_comps:]:
          extra.axis("off")
        # end
        if title:
          mpl_fig.suptitle(title)
        # end
        if layout_xlabel:
          mpl_fig.supxlabel(layout_xlabel)
        # end
        if layout_ylabel:
          mpl_fig.supylabel(layout_ylabel)
        # end

        for ax_idx in range(len(ax)):
          sub_titles = subplot_titles.split(",") if subplot_titles else []
          sub_xlabels = subplot_xlabels.split(",") if subplot_xlabels else []
          sub_ylabels = subplot_ylabels.split(",") if subplot_ylabels else []
          sub_title = sub_titles[ax_idx] if ax_idx < len(sub_titles) else ""
          sub_xlabel = sub_xlabels[ax_idx] if ax_idx < len(sub_xlabels) else ""
          sub_ylabel = sub_ylabels[ax_idx] if ax_idx < len(sub_ylabels) else ""
          ax[ax_idx].set_xlabel(sub_xlabel)
          ax[ax_idx].set_ylabel(sub_ylabel)
          if sub_title:
            ax[ax_idx].set_title(sub_title, y=1.08)
          # end
        # end
      # end
    # end

    if legend_subplot is not None:
      num_legend_subplots = 1 if squeeze else layout_num_comps
      if not isinstance(legend_subplot, int):
        raise TypeError("'legend_subplot' must be an integer or None")
      # end
      if legend_subplot < 0 or legend_subplot >= num_legend_subplots:
        raise ValueError(
            f"'legend_subplot' must be between 0 and {num_legend_subplots - 1}")
      # end
    # end

    # ---- Phase 2: draw each dataset ----
    im = None
    cur_start_axes = start_axes
    for ds_i, data in enumerate(states):
      if legend_labels is not None and ds_i < len(legend_labels):
        label_prefix = legend_labels[ds_i]
        explicit_legend_label = True
      # end
      elif len(states) > 1 or forcelegend:
        label_prefix = data.get_label()
        explicit_legend_label = False
      # end
      else:
        label_prefix = ""
        explicit_legend_label = False
      # end

      cells = data.num_cells
      grid = list(data.grid)
      values = data.values
      num_dims = len(cells) - int(np.sum(cells <= 1))
      if num_dims > 2:
        raise ValueError("Only 1D and 2D plots are currently supported")
      # end
      if split_linear_log and num_dims != 1:
        raise ValueError("every dataset must be 1D when 'split_linear_log' is set")
      # end

      axes_labels = list(_AXES_LABELS)
      if len(grid) > num_dims:
        idx = [d for d in range(len(grid)) if cells[d] <= 1]
        grid = [g.squeeze() for g in grid]
        if idx:
          for d in reversed(idx):
            grid.pop(d)
          # end
          cells = np.delete(cells, idx)
          axes_labels = list(np.delete(np.array(axes_labels), idx))
          values = np.squeeze(values, tuple(idx))
          if grid and grid[0].ndim > 1:  # curvilinear (mapped) coordinates
            for d in range(num_dims):
              for i in reversed(idx):
                grid[d] = np.mean(grid[d], axis=i)
              # end
            # end
          # end
        # end
      # end

      if transpose and num_dims == 2:  # swap the horizontal and vertical axes
        values = np.swapaxes(values, 0, 1)
        g0, g1 = grid[1], grid[0]
        if g0.ndim > 1:  # curvilinear coordinate arrays span both axes jointly
          g0, g1 = g0.transpose(), g1.transpose()
        # end
        grid[0], grid[1] = g0, g1
        cells = cells[[1, 0]]  # fancy indexing: num_cells may alias ctx["cells"]
        axes_labels[0], axes_labels[1] = axes_labels[1], axes_labels[0]
      # end

      num_comps = values.shape[-1]
      idx_comps = range(int(np.floor(num_comps / step)))

      for comp in idx_comps:
        logical_ax_idx = 0 if squeeze else comp + cur_start_axes
        if split_linear_log:
          component_axes = ax[2 * logical_ax_idx:2 * logical_ax_idx + 2]
          cax = component_axes[0]
        # end
        else:
          cax = ax[logical_ax_idx]
          component_axes = [cax]
        # end
        comp_label = (label_prefix if explicit_legend_label else
            (f"{label_prefix:s}_c{comp:d}".strip("_")
             if len(idx_comps) > 1 else label_prefix))
        comp_legend = (legend and (legend_subplot is None
            or (logical_ax_idx == legend_subplot if split_linear_log
                else cax is ax[legend_subplot])))
        comp_colorbar = colorbar

        if num_dims == 1:
          nodal_grid = _nodal_grid(grid, cells)
          x = (nodal_grid[0] + xshift) * xscale
          y = (values[..., comp] + yshift) * yscale
          if transpose:  # put the coordinate on the vertical axis
            x, y = y, x
          # end
          # Color the line from the colormap when a 'cval' is given (1D only).
          line_color = color
          if cmap and cval is not None:
            if cval_max is not None and cval_min is not None and cval_max != cval_min:
              t = (cval - cval_min) / (cval_max - cval_min)
            # end
            else:
              t = 0.5
            # end
            line_color = plt.get_cmap(cmap)(t)
          # end
          if split_linear_log:
            left_mask = x < split_point
            split_masks = (left_mask, ~left_mask)
            im = []
            for split_ax, mask in zip(component_axes, split_masks):
              im.extend(split_ax.plot(x[mask], y[mask], *args,
                  color=line_color, label=comp_label, markersize=markersize))
            # end
          # end
          else:
            im = cax.plot(x, y, *args, color=line_color, label=comp_label,
                markersize=markersize)
          # end
          # Add a colorbar describing the cval-to-color mapping once per axes.
          if (cmap and cval is not None and comp_colorbar
              and cval_max is not None and cval_min is not None and cval_max != cval_min
              and not getattr(cax, "_pgkyl_cval_cbar", False)):
            mappable = cm.ScalarMappable(
                norm=colors.Normalize(vmin=cval_min, vmax=cval_max), cmap=plt.get_cmap(cmap))
            _pgkyl_colorbar(mappable, mpl_fig, cax, label=layout_clabel)
            cax._pgkyl_cval_cbar = True
          # end
        # end

        elif num_dims == 2:
          extend = None

          if surface:  # ------------------------------------------------------
            nodal_grid = _nodal_grid(grid, cells)
            xg = (nodal_grid[0] + xshift) * xscale
            yg = (nodal_grid[1] + yshift) * yscale
            z = (values[..., comp].transpose() + zshift) * zscale
            if xg.ndim == 1:
              xg, yg = np.meshgrid(xg, yg)
            # end
            else:
              xg, yg = xg.transpose(), yg.transpose()
            # end
            # Count how many overlays already live on these axes so each gets
            # a distinct color (used for both surface and contour comparisons).
            overlay_count = getattr(cax, "_pgkyl_overlay_count", 0)
            cax._pgkyl_overlay_count = overlay_count + 1
            if comparison or bool(color):
              surf_color = color if bool(color) else f"C{overlay_count:d}"
              im = cax.plot_surface(xg, yg, z, color=surf_color,
                  alpha=alpha if alpha is not None else 0.6,
                  linewidth=0, antialiased=True, shade=True)
              if comp_label:
                handles = getattr(cax, "_pgkyl_handles", [])
                handles.append(patches.Patch(color=surf_color, label=comp_label))
                cax._pgkyl_handles = handles
              # end
            # end
            else:
              im = cax.plot_surface(xg, yg, z, cmap=mpl.rcParams["image.cmap"],
                  alpha=alpha if alpha is not None else 1.0,
                  linewidth=0, antialiased=True)
              if comp_colorbar:
                mpl_fig.colorbar(im, ax=cax, label=layout_clabel or "",
                    shrink=0.6, pad=0.1)
              # end
            # end
            if layout_clabel:
              cax.set_zlabel(layout_clabel)
            # end
            if zmin is not None or zmax is not None:
              cax.set_zlim(zmin, zmax)
            # end
            comp_colorbar = False
          # end

          elif contour:  # ------------------------------------------------------
            levels = 10
            if cnlevels:
              levels = int(cnlevels) - 1
            # end
            elif clevels:
              if ":" in clevels:
                s = clevels.split(":")
                levels = np.linspace(float(s[0]), float(s[1]), int(s[2]))
              # end
              else:
                levels = np.array(clevels.split(","))
                levels = np.array(list(filter(None, levels)))
              # end
            # end
            if isinstance(levels, np.ndarray) and len(levels) == 1:
              comp_colorbar = False
            # end
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
                handles.append(patches.Patch(color=cont_colors, label=comp_label))
                cax._pgkyl_handles = handles
              # end
              comp_colorbar = False
            # end
            im = cax.contour(x, y, z, levels, *args, origin="lower",
                colors=cont_colors, linewidths=linewidth)
            if cont_label:
              cax.clabel(im, inline=1)
          # end
            # end

          elif quiver:  # -----------------------------------------------------
            skip = int(np.max((len(grid[0]), len(grid[1]))) // 15)
            skip2 = int(skip // 2)
            nodal_grid = _nodal_grid(grid, cells)
            if nodal_grid[0].ndim == 1:
              x = (nodal_grid[0][skip2::skip] + xshift) * xscale
              y = (nodal_grid[1][skip2::skip] + yshift) * yscale
            # end
            else:
              x = (nodal_grid[0][skip2::skip, skip2::skip] + xshift) * xscale
              y = (nodal_grid[1][skip2::skip, skip2::skip] + yshift) * yscale
            # end
            z1 = (values[skip2::skip, skip2::skip, 2 * comp].transpose()
                  + zshift) * zscale
            z2 = (values[skip2::skip, skip2::skip, 2 * comp + 1].transpose()
                  + zshift) * zscale
            im = cax.quiver(x, y, z1, z2)
          # end

          elif streamline:  # -------------------------------------------------
            if color:
              cl = color
            # end
            else:
              cl = np.sqrt(values[..., 2 * comp] ** 2
                  + values[..., 2 * comp + 1] ** 2).transpose()
            # end
            nodal_grid = _nodal_grid(grid, cells)
            x = (nodal_grid[0] + xshift) * xscale
            y = (nodal_grid[1] + yshift) * yscale
            z1 = (values[..., 2 * comp].transpose() + zshift) * zscale
            z2 = (values[..., 2 * comp + 1].transpose() + zshift) * zscale
            im = cax.streamplot(x, y, z1, z2, *args, density=sdensity,
                broken_streamlines=False, color=cl, linewidth=linewidth)
          # end

          elif lineouts is not None:  # ---------------------------------------
            num_lines = values.shape[1] if lineouts == 0 else values.shape[0]
            nodal_grid = _nodal_grid(grid, cells)

            if lineouts == 0:
              x = (nodal_grid[0] + xshift) * xscale
              line_vmin = (nodal_grid[1][0] + yshift) * yscale
              line_vmax = (nodal_grid[1][-1] + yshift) * yscale
              cbar_label = clabel or axes_labels[1]
            # end
            else:
              x = (nodal_grid[1] + xshift) * xscale
              line_vmin = (nodal_grid[0][0] + yshift) * yscale
              line_vmax = (nodal_grid[0][-1] + yshift) * yscale
              cbar_label = clabel or axes_labels[0]
            # end
            line_idx = [slice(0, u) for u in values.shape]
            line_idx[-1] = comp
            for line in range(num_lines):
              line_color = cm.inferno(line / (num_lines - 1))
              if lineouts == 0:
                line_idx[1] = line
              # end
              else:
                line_idx[0] = line
              # end
              y = (values[tuple(line_idx)] + yshift) * yscale
              im = cax.plot(x, y, *args, color=line_color)
            # end
            mappable = cm.ScalarMappable(
                norm=colors.Normalize(vmin=line_vmin, vmax=line_vmax, clip=False),
                cmap=cm.inferno)
            _pgkyl_colorbar(mappable, mpl_fig, cax, label=cbar_label)
            comp_colorbar = False
            comp_legend = False
          # end

          else:  # ------------------------------------------------------------
            if zmin is not None and zmax is not None:
              extend = "both"
            # end
            elif zmax is not None:
              extend = "max"
            # end
            elif zmin is not None:
              extend = "min"
            # end
            x = (grid[0] + xshift) * xscale
            y = (grid[1] + yshift) * yscale
            z = (values[..., comp].transpose() + zshift) * zscale
            if len(x) == z.shape[1] or len(y) == z.shape[0]:
              nodal_grid = _nodal_grid(grid, cells)
              x = (nodal_grid[0] + xshift) * xscale
              y = (nodal_grid[1] + yshift) * yscale
            # end
            if x.ndim > 1:
              x, y = x.transpose(), y.transpose()
            # end
            comp_zmin, comp_zmax = zmin, zmax
            if diverging:
              comp_zmax = np.abs(z).max()
              comp_zmin = -comp_zmax
            # end
            vmax, vmin = comp_zmax, comp_zmin
            norm = None
            if logz:
              if diverging:
                tmp = vmax / 1000
                norm = colors.SymLogNorm(linthresh=tmp, linscale=tmp,
                    vmin=vmin, vmax=vmax, base=10)
              # end
              else:
                norm = colors.LogNorm(vmin=vmin, vmax=vmax)
              # end
              vmin, vmax = None, None
            # end
            im = cax.pcolormesh(x, y, z, norm=norm, vmin=vmin, vmax=vmax,
                edgecolors=edgecolors, linewidth=0.1, shading="auto", *args)
          # end
          if not color and comp_colorbar and not streamline:
            _pgkyl_colorbar(im, mpl_fig, cax, extend=extend, label=layout_clabel)
        # end
          # end
        else:
          raise ValueError(f"{num_dims:d}D data not supported")
        # end

        legend_ax = cax
        if split_linear_log:
          if split_legend_side == "left":
            legend_ax = component_axes[0]
          # end
          elif split_legend_side == "right":
            legend_ax = component_axes[1]
          # end
          elif split_legend_side == "linear":
            legend_ax = component_axes[0 if split_log_side == "right" else 1]
          # end
          else:  # log
            legend_ax = component_axes[0 if split_log_side == "left" else 1]
          # end
        # end
        if comp_legend:
          if getattr(legend_ax, "_pgkyl_handles", None):
            # Overlaid 2D datasets (surface/contour comparison): real legend.
            legend_ax.legend(handles=legend_ax._pgkyl_handles, loc=legend_loc)
          # end
          elif num_dims == 1 and comp_label != "":
            legend_ax.legend(loc=legend_loc)
          # end
          elif not (surface and num_dims == 2):
            legend_ax.text(0.03, 0.96, comp_label,
                bbox={"facecolor": "w", "edgecolor": "w", "alpha": 0.8,
                      "boxstyle": "round"},
                verticalalignment="top", horizontalalignment="left",
                transform=legend_ax.transAxes)
          # end
        # end
        for side_idx, side_ax in enumerate(component_axes):
          side_ax.grid(showgrid)
          if hashtag and (not split_linear_log or side_ax is legend_ax):
            side_ax.text(0.97, 0.03, "#pgkyl",
                bbox={"facecolor": "w", "edgecolor": "w", "alpha": 0.8,
                      "boxstyle": "round"},
                verticalalignment="bottom", horizontalalignment="right",
                transform=side_ax.transAxes)
          # end
          if logx:
            side_ax.set_xscale("log")
          # end
          if logy:
            side_ax.set_yscale("log")
          # end
          if split_linear_log:
            is_log_side = ((side_idx == 0 and split_log_side == "left")
                or (side_idx == 1 and split_log_side == "right"))
            if is_log_side:
              side_ax.set_yscale("log", base=split_log_base,
                  nonpositive=split_log_nonpositive)
              side_ylim = _split_ylim_for_component(split_log_ylim, comp)
            # end
            else:
              side_ax.set_yscale("linear")
              side_ylim = _split_ylim_for_component(split_linear_ylim, comp)
            # end
          # end
          else:
            side_ylim = None
          # end
          if num_dims == 1 and not relax:  # this causes troubles with contours
            side_ax.autoscale(enable=True, axis="x", tight=True)
            side_ax.autoscale(enable=True, axis="y")
          # end
          if split_linear_log:
            if side_idx == 0:
              side_ax.set_xlim(xmin, split_point)
            # end
            else:
              side_ax.set_xlim(split_point, xmax)
            # end
            if not logx:
              prune = None
              if ((split_seam_ticklabels == "left" and side_idx == 1)
                  or (split_seam_ticklabels == "right" and side_idx == 0)
                  or split_seam_ticklabels == "none"):
                prune = "upper" if side_idx == 0 else "lower"
              # end
              if prune is not None:
                side_ax.xaxis.get_major_locator().set_params(prune=prune)
              # end
            # end
          elif xmin is not None or xmax is not None:
            side_ax.set_xlim(xmin, xmax)
          # end
          if ymin is not None or ymax is not None:
            side_ax.set_ylim(ymin, ymax)
          # end
          if side_ylim is not None:
            side_ax.set_ylim(*side_ylim)
          # end
          if fixaspect and not (surface and num_dims == 2):
            plt.setp(side_ax, aspect=aspect)
          # end
        # end
        # end
      # end component loop

      if num_axes:
        cur_start_axes += num_comps
    # end
      # end
    # end dataset loop

    mpl_fig.tight_layout()
  if show:
    plt.show()
  # end
  return mpl_fig
# end
