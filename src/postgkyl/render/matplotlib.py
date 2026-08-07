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
      # end
      z = (values[..., comp] + zshift) * zscale
      if not np.any(np.isfinite(z)):
        continue
      # end
      low = min(low, float(np.nanmin(z)))
      high = max(high, float(np.nanmax(z)))
    # end
    ranges.append((low, high) if np.isfinite(low) and low < high else (None, None))
  # end
  return ranges
# end


def plot(*datasets, args: str = "", figure=None, squeeze: bool = False,
    transpose: bool = False, num_axes: int | None = None, start_axes: int = 0,
    spread_axes: bool = True,
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
    legend: bool = True, labels: list | None = None, forcelegend: bool = False,
    colorbar: bool = True,
    xlabel: str | None = None, ylabel: str | None = None,
    clabel: str | None = None, title: str | None = None,
    subplot_titles: str | None = None, subplot_xlabels: str | None = None,
    subplot_ylabels: str | None = None,
    logx: bool = False, logy: bool = False, logz: bool = False,
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
  Set ``spread_axes=False`` to keep every dataset in the *same*
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
  controls the surface transparency. ``xkcd`` no longer leaks into
  Matplotlib's global rcParams past this call -- it is scoped to the figure
  drawn here.

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

    if mpl_fig.axes:
      ax = mpl_fig.axes
      if not squeeze and layout_num_comps > len(ax):
        raise ValueError("Trying to plot into figure with not enough axes")
    # end
      # end
    else:
      if squeeze:  # Plotting into 1 panel
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

    # One color scale for every dataset drawn here (see
    # _shared_component_range). Only the plain 2-D pcolormesh path consumes
    # it: surface/contour/quiver/streamline/lineouts each own their own
    # normalization, and an explicit zmin/zmax always wins.
    shared_z = None
    if (len(states) > 1 and ref_num_dims == 2 and zmin is None and zmax is None
        and not (surface or contour or quiver or streamline or diverging)
        and lineouts is None):
      shared_z = _shared_component_range(states, zshift, zscale)
    # end

    # ---- Phase 2: draw each dataset ----
    im = None
    cur_start_axes = start_axes
    for ds_i, data in enumerate(states):
      if labels is not None and ds_i < len(labels):
        label_prefix = labels[ds_i]
      # end
      elif len(states) > 1 or forcelegend:
        label_prefix = data.get_label()
      # end
      else:
        label_prefix = ""
      # end

      cells = data.num_cells
      grid = list(data.grid)
      values = data.values
      num_dims = len(cells) - int(np.sum(cells <= 1))
      if num_dims > 2:
        raise ValueError("Only 1D and 2D plots are currently supported")
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
        cax = ax[0] if squeeze else ax[comp + cur_start_axes]
        comp_label = (f"{label_prefix:s}_c{comp:d}".strip("_")
            if len(idx_comps) > 1 else label_prefix)
        comp_legend = legend
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
          im = cax.plot(x, y, *args, color=line_color, label=comp_label,
              markersize=markersize)
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
            elif shared_z is not None and comp < len(shared_z):
              comp_zmin, comp_zmax = shared_z[comp]
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
          # end
          if not color and comp_colorbar and not streamline and comp not in drawn:
            _pgkyl_colorbar(im, mpl_fig, cax, extend=extend, label=layout_clabel)
            drawn.add(comp)
        # end
          # end
        else:
          raise ValueError(f"{num_dims:d}D data not supported")
        # end

        cax.grid(showgrid)
        if comp_legend:
          if getattr(cax, "_pgkyl_handles", None):
            # Overlaid 2D datasets (surface/contour comparison): real legend.
            cax.legend(handles=cax._pgkyl_handles, loc=0)
          # end
          elif num_dims == 1 and comp_label != "":
            cax.legend(loc=0)
          # end
          elif not (surface and num_dims == 2):
            cax.text(0.03, 0.96, comp_label,
                bbox={"facecolor": "w", "edgecolor": "w", "alpha": 0.8,
                      "boxstyle": "round"},
                verticalalignment="top", horizontalalignment="left",
                transform=cax.transAxes)
          # end
        # end
        if hashtag:
          cax.text(0.97, 0.03, "#pgkyl",
              bbox={"facecolor": "w", "edgecolor": "w", "alpha": 0.8,
                    "boxstyle": "round"},
              verticalalignment="bottom", horizontalalignment="right",
              transform=cax.transAxes)
        # end
        if logx:
          cax.set_xscale("log")
        # end
        if logy:
          cax.set_yscale("log")
        # end
        if num_dims == 1 and not relax:  # this causes troubles with contours
          plt.autoscale(enable=True, axis="x", tight=True)
          plt.autoscale(enable=True, axis="y")
        # end
        if xmin is not None or xmax is not None:
          cax.set_xlim(xmin, xmax)
        # end
        if ymin is not None or ymax is not None:
          cax.set_ylim(ymin, ymax)
        # end
        if fixaspect and not (surface and num_dims == 2):
          plt.setp(cax, aspect=aspect)
      # end
        # end
      # end component loop

      if num_axes and spread_axes:
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
