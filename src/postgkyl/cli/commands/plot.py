"""``plot`` -- terminal verb; render the active datasets.

A faithful port of main's ``commands/plot.py``: every option, and the exact
per-dataset loop (default: a fresh figure per dataset; ``--figure``/
``--subplots``/``--multiblock``/``--figure dataset`` target a shared one,
exactly as before) is preserved. ``render.plot`` (the ``output.plot`` engine)
is called once per dataset, mirroring main's loop 1:1; this module owns
figure targeting, the legend-label decision, save/saveframes/batch-mode file
naming, and the final (single) ``plt.show()`` -- all of which were CLI-layer
concerns in main too.
"""

from __future__ import annotations

import os

import click
import numpy as np

import postgkyl as pg

from .._apply import active_datasets
from .._options import show_option, use_option


@click.command("plot")
@use_option
@click.option("--figure", "-f", default=None,
    help="Specify figure to plot in; either number or 'dataset'.")
@click.option("--squeeze", is_flag=True, default=False,
    help="Squeeze the components into one panel.")
@click.option("--subplots", "-b", is_flag=True, default=False,
    help="Make subplots from multiple datasets.")
@click.option("--nsubplotrow", "num_subplot_row", type=int, default=None,
    help="Manually set the number of rows for subplots.")
@click.option("--nsubplotcol", "num_subplot_col", type=int, default=None,
    help="Manually set the number of columns for subplots.")
@click.option("--transpose", is_flag=True, default=False, help="Transpose axes.")
# --arrowstyle: accepted for CLI compatibility but unused -- main never wired
# it into its render engine (a dead option there too).
@click.option("-c", "--contour", is_flag=True, default=False, help="Make contour plot.")
@click.option("--surface", "--surf", "surface", is_flag=True, default=False,
    help="Make a 3D surface plot for 2D data (auto-enabled when overlaying "
    "multiple 2D datasets).")
@click.option("--alpha", type=float, default=None,
    help="Surface transparency (0-1); useful when overlaying surfaces.")
@click.option("--clevels", default=None,
    help="Specify levels for contours: comma-separated level values or start:end:nlevels.")
@click.option("--cnlevels", type=int, default=None,
    help="Specify the number of levels for contours.")
@click.option("--contlabel", "cont_label", is_flag=True, default=False,
    help="Add labels to contours")
@click.option("-q", "--quiver", is_flag=True, default=False, help="Make quiver plot.")
@click.option("-l", "--streamline", is_flag=True, default=False, help="Make streamline plot.")
@click.option("--sdensity", type=int, default=1, help="Control density of the streamlines.")
@click.option("--arrowstyle", default=None, help="Set the style for streamline arrows.")
@click.option("--lineouts", type=click.Choice(["0", "1"]), default=None,
    help="Switch to lineouts mode.")
@click.option("-s", "--scatter", is_flag=True, default=False, help="Make scatter plot.")
@click.option("--markersize", type=float, default=None,
    help="Set marker size for scatter plots.")
@click.option("--linewidth", type=float, default=None, help="Set the linewidth.")
@click.option("--linestyle", type=click.Choice(["solid", "dashed", "dotted", "dashdot"]),
    default=None, help="Set the linestyle.")
@click.option("--style", default=None, help="Specify Matplotlib style file (default: Postgkyl).")
@click.option("-d", "--diverging", is_flag=True, default=False,
    help="Switch to diverging color map.")
@click.option("--arg", default="", help="Additional plotting arguments, e.g., '*--'.")
@click.option("--fix-aspect", "-a", "fixaspect", is_flag=True, default=False,
    help="Enforce the same scaling on both axes.")
@click.option("--aspect", default=None, help="Specify the scaling ratio.")
@click.option("--logx", is_flag=True, default=False, help="Set x-axis to log scale.")
@click.option("--logy", is_flag=True, default=False, help="Set y-axis to log scale.")
@click.option("--logz", is_flag=True, default=False, help="Set values of 2D plot to log scale.")
@click.option("--xshift", default=0.0, type=float, show_default=True,
    help="Value to shift the x-axis.")
@click.option("--yshift", default=0.0, type=float, show_default=True,
    help="Value to shift the y-axis.")
@click.option("--zshift", default=0.0, type=float, show_default=True,
    help="Value to shift the z-axis.")
@click.option("--xscale", default=1.0, type=float, show_default=True,
    help="Value to scale the x-axis.")
@click.option("--yscale", default=1.0, type=float, show_default=True,
    help="Value to scale the y-axis.")
@click.option("--zscale", default=1.0, type=float, show_default=True,
    help="Value to scale the z-axis (default: 1.0).")
@click.option("--xmax", default=None, type=float, help="Set maximal x-value.")
@click.option("--xmin", default=None, type=float, help="Set minimal x-values.")
@click.option("--ymax", default=None, type=float, help="Set maximal y-value.")
@click.option("--ymin", default=None, type=float, help="Set minimal y-values.")
@click.option("--zmax", default=None, type=float, help="Set maximal z-value.")
@click.option("--zmin", default=None, type=float, help="Set minimal z-values.")
@click.option("--xlim", default=None, help="Set limits for the x-coordinate (lower,upper)")
@click.option("--ylim", default=None, help="Set limits for the y-coordinate (lower,upper).")
@click.option("--zlim", default=None, help="Set limits for the z-coordinate (lower,upper).")
@click.option("--relax", is_flag=True, default=False,
    help="Relax the stringent x axis limits for 1D plots.")
@click.option("--globalrange", "-r", is_flag=True, default=False,
    help="Make uniform extends across datasets.")
@click.option("--cutoffglobalrange", "-cogr", default=None, type=float,
    help="Set custom limit for uniform across datasets")
@click.option("--legend", default=None,
    help="If specified, comma-separated legend labels (e.g., 'a,b,c').")
@click.option("--no-legend", is_flag=True, default=False, help="Hide legend.")
@click.option("--force-legend", "forcelegend", is_flag=True, default=False,
    help="Force legend even when plotting a single dataset.")
@click.option("--color", default=None, help="Set color when available.")
@click.option("-x", "--xlabel", default=None, help="Specify a x-axis label.")
@click.option("-y", "--ylabel", default=None, help="Specify a y-axis label.")
@click.option("--clabel", default=None, help="Specify a label for colorbar.")
@click.option("--title", default=None, help="Specify a title.")
@click.option("--subplot-titles", default=None,
    help="Comma-separated titles for each subplot. e.g. --subplot-titles 'Title1,Title2,Title3'")
@click.option("--subplot-xlabels", default=None,
    help="Comma-separated x-axis labels for each subplot. e.g. --subplot-xlabels 'X1,X2,X3'")
@click.option("--subplot-ylabels", default=None,
    help="Comma-separated y-axis labels for each subplot. e.g. --subplot-ylabels 'Y1,Y2,Y3'")
@click.option("--save", is_flag=True, default=False, help="Save figure as PNG file.")
@click.option("--saveas", default=None, help="Name of figure file.")
@click.option("--dpi", type=int, default=200, help="DPI (resolution) for output.")
@click.option("-e", "--edgecolors", default=None,
    help="Set color for cell edges to show grid outline.")
@click.option("--showgrid/--no-showgrid", default=True, help="Show grid-lines.")
@click.option("--xkcd", is_flag=True, default=False, help="Turns on the xkcd style!")
@click.option("--hashtag", is_flag=True, default=False, help="Turns on the pgkyl hashtag!")
@show_option("Turn showing of the plot ON and OFF.")
@click.option("--figsize", default=None, help="Comma-separated values for x and y size.")
@click.option("--saveframes", default=None,
    help="Save individual frames as PNGS instead of an opening them")
@click.option("--jet", is_flag=True, default=False,
    help="Turn colormap to jet for comparison with literature.")
@click.option("--cmap", "--colormap", "cmap", default=None,
    help="Override default colormap with a valid matplotlib cmap.")
@click.option("--cval", default=None,
    help="For 1D plots, comma-separated values mapping each curve onto the "
    "colormap (e.g. '1e-6,2e-6'). Requires --cmap; defaults to the dataset "
    "index if omitted.")
@click.option("-m", "--multiblock", is_flag=True, default=False,
    help="Put all blocks (datasets) on the same figure.")
@click.pass_context
def command(ctx, use, figure, squeeze, subplots, num_subplot_row, num_subplot_col,
    transpose, contour, surface, alpha, clevels, cnlevels, cont_label, quiver, streamline,
    sdensity, arrowstyle, lineouts, scatter, markersize, linewidth, linestyle,
    style, diverging, arg, fixaspect, aspect, logx, logy, logz, xshift, yshift,
    zshift, xscale, yscale, zscale, xmax, xmin, ymax, ymin, zmax, zmin,
    xlim, ylim, zlim, relax, globalrange, cutoffglobalrange, legend, no_legend,
    forcelegend, color, xlabel, ylabel, clabel, title, subplot_titles,
    subplot_xlabels, subplot_ylabels, save, saveas, dpi, edgecolors, showgrid,
    xkcd, hashtag, show, figsize, saveframes, jet, cmap, cval, multiblock) -> None:
  """Plot active datasets, optionally displaying the plot and/or saving it to PNG files.

  Plot labels can use a sub-set of LaTeX math commands placed between dollar ($) signs.
  """
  import matplotlib.pyplot as plt

  ds = ctx.obj

  if scatter:
    arg = arg + "."
  # end

  if jet:
    click.echo(click.style(
        "WARNING: The 'jet' colormap has been selected. This colormap is "
        "not perceptually uniform and seemingly creates features which do "
        "not exist in the data!", fg="yellow"))
  # end

  if aspect:
    fixaspect = True
  # end

  if lineouts is not None:
    lineouts = int(lineouts)
  # end

  if xlim:
    xmin, xmax = (float(v) for v in xlim.split(","))
  # end
  if ylim:
    ymin, ymax = (float(v) for v in ylim.split(","))
  # end
  if zlim:
    zmin, zmax = (float(v) for v in zlim.split(","))
  # end

  dataset_fignum = figure in ("dataset", "set", "s")

  # Automatically sets correct scale for multiblock cases.
  if multiblock and cutoffglobalrange is None:
    globalrange = True
  # end

  all_active = active_datasets(ctx)
  pool = all_active
  if use is not None:
    pool = [d for d in pool if d.tag == use]
  # end
  if not pool:
    raise click.UsageError("plot: no datasets to plot; load a file first")
  # end

  # When several 2D datasets are drawn into the same figure we switch to
  # contour mode by default, and give each overlay its own color + legend
  # entry (rather than letting them obscure one another).
  first_cells = pool[0].num_cells
  is_2d = (len(first_cells) - int(np.sum(first_cells <= 1))) == 2
  overlay_2d = (
      is_2d and len(pool) > 1 and not dataset_fignum
      and figure is not None
      and not subplots and lineouts is None
      and not quiver and not streamline
  )
  if overlay_2d and not surface and not contour:
    contour = True
  # end
  comparison = overlay_2d and (surface or contour)

  if globalrange or cutoffglobalrange:
    vmin, vmax = float("inf"), float("-inf")
    v_extrema = np.array([])
    for d in pool:
      val = d.values * zscale
      vmin = min(vmin, float(np.nanmin(val)))
      vmax = max(vmax, float(np.nanmax(val)))
      v_extrema = np.append(v_extrema, np.nanmin(val))
      v_extrema = np.append(v_extrema, np.nanmax(val))
    # end
    v_extrema = np.sort(v_extrema)
    if cutoffglobalrange:
      boundary = 100 * (1 - cutoffglobalrange) / 2
      vmax = float(np.percentile(v_extrema, 100 - boundary))
      vmin = float(np.percentile(v_extrema, boundary))
    # end
    if zmin is None:
      zmin = vmin
    # end
    if zmax is None:
      zmax = vmax
    # end
  # end

  # Prevents scale errors for multiblock contour plots.
  if multiblock and contour and clevels is None:
    clevels = f"{zmin}:{zmax}:10"
  # end

  legend_labels = None
  if legend:
    legend_labels = [s.strip() for s in legend.split(",")]
  # end
  show_legend = not no_legend

  # Colormap-based line coloring for 1D plots.
  cval_list = None
  if cval:
    cval_list = [float(v) for v in cval.split(",")]
  # end
  elif cmap:
    cval_list = list(range(len(pool)))
  # end
  cval_min = min(cval_list) if cval_list else None
  cval_max = max(cval_list) if cval_list else None

  num_axes = None
  start_axes = 0
  if subplots:
    num_axes = int(sum(d.num_comps for d in pool))
    if figure is None:
      figure = 0
    # end
  # end

  parsed_figsize = None
  if figsize:
    parts = figsize.split(",")
    if len(parts) != 2:
      raise click.UsageError(
          f"--figsize expects 'w,h' (e.g. '8,6'), got '{figsize}'")
    # end
    try:
      parsed_figsize = (float(parts[0]), float(parts[1]))
    # end
    except ValueError:
      raise click.UsageError(
          f"--figsize expects 'w,h' (e.g. '8,6'), got '{figsize}'")
    # end
  # end

  file_name = ""
  for i, d in enumerate(pool):
    fig_target = figure
    if dataset_fignum:
      fig_target = int(i)
    # end
    if multiblock:  # puts all blocks on the same figure
      fig_target = 0
    # end

    if legend_labels is not None and i < len(legend_labels):
      label = legend_labels[i]
    # end
    elif len(all_active) > 1 or forcelegend:
      label = d.get_label()
    # end
    else:
      label = ""
    # end

    cval_value = cval_list[i] if cval_list is not None and i < len(cval_list) else None

    pg.plot(d, args=arg, figure=fig_target, squeeze=squeeze,
        transpose=transpose, num_axes=num_axes, start_axes=start_axes,
        num_subplot_row=num_subplot_row, num_subplot_col=num_subplot_col,
        streamline=streamline, sdensity=sdensity, quiver=quiver,
        contour=contour, clevels=clevels, cnlevels=cnlevels,
        cont_label=cont_label, surface=surface, comparison=comparison,
        alpha=alpha, diverging=diverging, lineouts=lineouts,
        xmin=xmin, xmax=xmax, xscale=xscale, xshift=xshift,
        ymin=ymin, ymax=ymax, yscale=yscale, yshift=yshift,
        zmin=zmin, zmax=zmax, zscale=zscale, zshift=zshift,
        relax=relax, style=style, legend=show_legend, legend_labels=[label],
        colorbar=True,
        xlabel=xlabel, ylabel=ylabel, clabel=clabel, title=title,
        subplot_titles=subplot_titles, subplot_xlabels=subplot_xlabels,
        subplot_ylabels=subplot_ylabels, logx=logx, logy=logy, logz=logz,
        fixaspect=fixaspect, aspect=aspect, edgecolors=edgecolors,
        showgrid=showgrid, hashtag=hashtag, xkcd=xkcd, color=color,
        markersize=markersize, linewidth=linewidth, linestyle=linestyle,
        figsize=parsed_figsize, jet=jet, cmap=cmap,
        cval=cval_value, cval_min=cval_min, cval_max=cval_max, show=False)

    if subplots:
      start_axes += d.num_comps
    # end

    if save or saveas:
      if saveas:
        file_name = saveas
      # end
      else:
        if file_name != "":
          file_name = file_name + "_"
        # end
        src = getattr(d, "_file_name", "") or ""
        if src:
          file_name = file_name + os.path.basename(src).split(".")[0]
        # end
        else:
          file_name = file_name + "ev_" + (d.get_label() or f"dataset_{i}").replace(" ", "_")
        # end
      # end
    # end
    if (save or saveas) and fig_target is None:
      plt.savefig(str(file_name), dpi=dpi)
      file_name = ""
    # end

    if saveframes:
      plt.savefig(f"{saveframes}_{i}.png", dpi=dpi)
    # end

    if ds.batch and not (save or saveas or saveframes):
      plt.savefig(f"{ds.prefix}_{i}.png", dpi=dpi)
    # end
  # end

  if (save or saveas) and file_name:
    plt.savefig(str(file_name), dpi=dpi)
  # end

  if show and not (saveframes or ds.batch):
    plt.show()
  # end
  else:
    # Nothing will ever display this figure (saved-only/batch/saveframes
    # runs) -- close it so headless pipelines that call `plot` repeatedly
    # (e.g. a batch-mode loop over frames) don't accumulate open figures.
    plt.close("all")
  # end
# end
  # end
