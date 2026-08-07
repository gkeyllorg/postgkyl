"""``animate`` -- animate the active datasets, one frame per dataset.

A thin shell, mirroring ``plot.py``: option parsing plus the pool-level
bookkeeping a single call to ``render.animate`` cannot know on its own (which
datasets are in the pool, how they group into frames, per-group output
naming). Every plot-styling option is forwarded straight through -- the same
vocabulary ``plot`` uses, since both funnel through ``render.matplotlib.plot``
per frame. The animate-specific options (``--use``, ``--grouptags``,
``--multiblock``, ``--float``, ``--nproc``/``--tmpdir``) decide *which*
datasets become frames and how the sequence is built; building the actual
animation and its save/saveframes/show lifecycle lives in
``render.animate.animate`` (via ``pg.animate``), so the same options work
identically from a script.

Two deliberate deviations from main's ``commands/animate.py``:

- ``--grouptags``/explicit ``--saveas``/``--saveframes`` no longer silently
  overwrite one tag's output with the next: multiple resolved tags get their
  filenames suffixed with the tag automatically.
- An explicit ``--title`` is no longer clobbered by the per-frame frame/time
  auto-title every frame (see ``render.animate._draw_frame``) -- it was a
  dead option in main (declared, but always overwritten before use).

``--collected``, ``--group``/``-g`` and ``--arrowstyle`` were never wired to
any behavior in main either; the first two are dropped, ``--arrowstyle`` is
kept accepted-but-unused for CLI compatibility, exactly as ``plot.py`` does.
"""

from __future__ import annotations

import os

import click

import postgkyl as pg

from .._apply import active_datasets
from .._options import show_option, use_option


def _group_by_frame(datasets: list) -> list[list]:
  """Group ``datasets`` into per-frame lists, sorted by ascending frame index.

  ``ctx["frame"]`` is authoritative: readers stamp it from the file header,
  and ``GDataState`` falls back to the ``_<digits>`` suffix of the file name
  (``io.naming``). This replaces main's ``utils.set_frame``, which recovered
  a frame index by diffing the loaded file names character by character --
  a second, weaker home for the same naming convention.

  Datasets with no resolvable frame are all placed in one trailing group
  rather than crashing, so a mixed pool still animates.
  """
  groups: dict = {}
  for d in datasets:
    frame = d.ctx.get("frame")
    # int(): a reader-native ctx["frame"] may come back as a numpy scalar,
    # which isn't hashable the same way a plain int is.
    groups.setdefault(int(frame) if frame is not None else None, []).append(d)
  # end
  known = sorted(f for f in groups if f is not None)
  return [groups[f] for f in known] + ([groups[None]] if None in groups else [])
# end


def _suffixed(path: str | None, suffix: str) -> str | None:
  """``path`` with ``_<suffix>`` inserted before its extension (or appended,
  for an extension-less prefix like ``--saveframes``); ``None`` stays ``None``."""
  if path is None:
    return None
  # end
  base, ext = os.path.splitext(path)
  return f"{base}_{suffix}{ext}"
# end


@click.command("animate")
@use_option
@click.option("--grouptags", is_flag=True, default=False,
    help="Animate each tag separately instead of mixing tags into one sequence.")
@click.option("-m", "--multiblock", is_flag=True, default=False,
    help="Group datasets sharing a frame index into one multi-block frame. "
    "Automatic when the file names carry a '<sim>_b<N>-' block index; this "
    "flag forces the grouping for pools that do not.")
@click.option("--squeeze", is_flag=True, default=False,
    help="Squeeze the components into one panel.")
@click.option("--nsubplotrow", "num_subplot_row", type=int, default=None,
    help="Manually set the number of rows for the component subplot grid.")
@click.option("--nsubplotcol", "num_subplot_col", type=int, default=None,
    help="Manually set the number of columns for the component subplot grid.")
@click.option("--transpose", is_flag=True, default=False, help="Transpose axes.")
# --arrowstyle: accepted for CLI compatibility but unused -- main never wired
# it into its render engine (a dead option there too).
@click.option("-c", "--contour", is_flag=True, default=False, help="Make contour plot.")
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
    help="Value to scale the z-axis.")
@click.option("--float", "use_float_range", is_flag=True, default=False,
    help="Scale each frame to its own min/max instead of a fixed range "
    "shared across the whole animation.")
@click.option("--xmax", default=None, type=float, help="Set maximal x-value.")
@click.option("--xmin", default=None, type=float, help="Set minimal x-values.")
@click.option("--ymax", default=None, type=float, help="Set maximal y-value.")
@click.option("--ymin", default=None, type=float, help="Set minimal y-values.")
@click.option("--zmax", default=None, type=float, help="Set maximal z-value.")
@click.option("--zmin", default=None, type=float, help="Set minimal z-values.")
@click.option("--xlim", default=None, help="Set limits for the x-coordinate (lower,upper).")
@click.option("--ylim", default=None, help="Set limits for the y-coordinate (lower,upper).")
@click.option("--zlim", default=None, help="Set limits for the z-coordinate (lower,upper).")
@click.option("--relax", is_flag=True, default=False,
    help="Relax the stringent x axis limits for 1D plots.")
@click.option("--cutoffglobalrange", "-cogr", default=None, type=float,
    help="Specify the middle percentile of data extrema the fixed range covers.")
@click.option("--legend/--no-legend", default=True, help="Show legend.")
@click.option("--force-legend", "forcelegend", is_flag=True, default=False,
    help="Force legend even when a frame holds a single dataset.")
@click.option("--colorbar/--no-colorbar", default=True,
    help="Show colorbar (2D animations); disabling it improves animation performance.")
@click.option("--color", default=None, help="Set color when available.")
@click.option("-x", "--xlabel", default=None, help="Specify a x-axis label.")
@click.option("-y", "--ylabel", default=None, help="Specify a y-axis label.")
@click.option("--clabel", default=None, help="Specify a label for colorbar.")
@click.option("--title", default=None, help="Specify a title (shown on every frame).")
@click.option("--notitle", is_flag=True, default=False,
    help="Suppress the automatic per-frame frame/time title.")
@click.option("--subplot-titles", default=None,
    help="Comma-separated titles for each subplot.")
@click.option("--subplot-xlabels", default=None,
    help="Comma-separated x-axis labels for each subplot.")
@click.option("--subplot-ylabels", default=None,
    help="Comma-separated y-axis labels for each subplot.")
@click.option("-i", "--interval", type=int, default=100,
    help="Live-animation delay between frames, in milliseconds.")
@click.option("--save", is_flag=True, default=False, help="Save the animation.")
@click.option("--saveas", default=None,
    help="Save path (.gif/.webp/.apng, or .mp4/.mov/.avi/.mkv via ffmpeg).")
@click.option("--fps", type=int, default=None, help="Frames per second for a saved movie.")
@click.option("--dpi", type=int, default=None, help="Resolution for saved frames/movies.")
@click.option("-e", "--edgecolors", default=None, help="Set color for cell edges.")
@click.option("--showgrid/--no-showgrid", default=True, help="Show grid-lines.")
@click.option("--hashtag", is_flag=True, default=False, help="Turns on the pgkyl hashtag!")
@click.option("--xkcd", is_flag=True, default=False, help="Turns on the xkcd style!")
@click.option("--jet", is_flag=True, default=False,
    help="Turn colormap to jet for comparison with literature.")
@click.option("--cmap", default=None,
    help="Override default colormap with a valid matplotlib cmap.")
@show_option("Turn showing of the plot ON and OFF.")
@click.option("--saveframes", default=None,
    help="Write '<prefix>_<i>.png' per frame instead of a live/saved animation.")
@click.option("--nproc", default=1, type=int, show_default=True,
    help="Number of parallel processes for frame generation.")
@click.option("--tmpdir", default=None,
    help="Directory for the temporary frame directory used by --nproc without --saveframes.")
@click.option("--figsize", default=None, help="Comma-separated values for x and y size.")
@click.pass_context
def command(ctx, use, grouptags, multiblock, squeeze, num_subplot_row, num_subplot_col,
    transpose, contour, clevels, cnlevels, cont_label, quiver, streamline, sdensity,
    arrowstyle, lineouts, scatter, markersize, linewidth, linestyle, style, diverging,
    arg, fixaspect, aspect, logx, logy, logz, xshift, yshift, zshift, xscale, yscale,
    zscale, use_float_range, xmax, xmin, ymax, ymin, zmax, zmin, xlim, ylim, zlim, relax,
    cutoffglobalrange, legend, forcelegend, colorbar, color, xlabel, ylabel, clabel,
    title, notitle, subplot_titles, subplot_xlabels, subplot_ylabels, interval, save,
    saveas, fps, dpi, edgecolors, showgrid, hashtag, xkcd, jet, cmap, show, saveframes,
    nproc, tmpdir, figsize) -> None:
  """Animate the active datasets, one frame per dataset.

  Typically the datasets are loaded using the wildcard/regex feature of a shell
  glob or the ``load`` command, so each active dataset becomes one frame.
  """
  ds = ctx.obj

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

  parsed_figsize = None
  if figsize:
    parts = figsize.split(",")
    if len(parts) != 2:
      raise click.UsageError(f"--figsize expects 'w,h' (e.g. '8,6'), got '{figsize}'")
    # end
    try:
      parsed_figsize = (float(parts[0]), float(parts[1]))
    except ValueError:
      raise click.UsageError(f"--figsize expects 'w,h' (e.g. '8,6'), got '{figsize}'")
    # end
  # end

  pool = active_datasets(ctx)
  if use is not None:
    pool = [d for d in pool if d.tag == use]
  # end
  if not pool:
    raise click.UsageError("animate: no datasets to animate; load files first")
  # end

  plot_kwargs = dict(
      squeeze=squeeze, transpose=transpose, num_subplot_row=num_subplot_row,
      num_subplot_col=num_subplot_col, streamline=streamline, sdensity=sdensity,
      quiver=quiver, contour=contour, clevels=clevels, cnlevels=cnlevels,
      cont_label=cont_label, diverging=diverging, lineouts=lineouts,
      xmin=xmin, xmax=xmax, xscale=xscale, xshift=xshift,
      ymin=ymin, ymax=ymax, yscale=yscale, yshift=yshift,
      zmin=zmin, zmax=zmax, zscale=zscale, zshift=zshift,
      relax=relax, style=style, legend=legend, forcelegend=forcelegend,
      colorbar=colorbar, xlabel=xlabel, ylabel=ylabel, clabel=clabel, title=title,
      subplot_titles=subplot_titles, subplot_xlabels=subplot_xlabels,
      subplot_ylabels=subplot_ylabels, logx=logx, logy=logy, logz=logz,
      fixaspect=fixaspect, aspect=aspect, edgecolors=edgecolors, showgrid=showgrid,
      hashtag=hashtag, xkcd=xkcd, color=color, markersize=markersize,
      linewidth=linewidth, linestyle=linestyle, args=arg, jet=jet, cmap=cmap)

  animate_kwargs = dict(
      interval=interval, fixed_range=not use_float_range,
      cutoffglobalrange=cutoffglobalrange, notitle=notitle, fps=fps, dpi=dpi,
      figsize=parsed_figsize, nproc=nproc, tmpdir=tmpdir, **plot_kwargs)

  resolved_show = show and not (saveframes or ds.batch)

  # Blocks of one field belong in one frame, drawn together. The block index
  # is recognized from the '<sim>_b<N>-...' file names and stamped into ctx at
  # load time (io.naming), so this needs no flag; --multiblock remains the
  # explicit override for pools whose names carry no block index.
  if multiblock or any(d.ctx.get("block") is not None for d in pool):
    groups = _group_by_frame(pool)
    if not color and groups[0][0].num_dims == 1:
      # Keep stitched blocks looking like one continuous curve/mesh instead
      # of a rainbow of per-block colors.
      animate_kwargs["color"] = "tab:blue"
    # end
    save_path = saveas
    if ds.batch and not save_path and not saveframes:
      save_path = f"{ds.prefix}.gif"
    # end
    pg.animate(*groups, save=save or bool(save_path), saveas=save_path,
        saveframes=saveframes, show=resolved_show, **animate_kwargs)
    return
  # end

  if grouptags:
    tags = list(dict.fromkeys(d.tag for d in pool))
    min_size = min(sum(1 for d in pool if d.tag == tag) for tag in tags)
    anims = []
    for tag in tags:
      tag_pool = [d for d in pool if d.tag == tag][:min_size]
      save_path = saveas
      tag_saveframes = saveframes
      if len(tags) > 1:
        save_path = _suffixed(save_path, tag) if save_path else None
        tag_saveframes = f"{saveframes}_{tag}" if saveframes else None
      # end
      if ds.batch and not save_path and not tag_saveframes:
        save_path = f"{ds.prefix}_{tag}.gif" if len(tags) > 1 else f"{ds.prefix}.gif"
      # end
      # Each tag gets its own figure; defer the final plt.show() (below)
      # until every tag's animation has been built, so live windows open
      # together rather than blocking one at a time.
      anims.append(pg.animate(*tag_pool, save=save or bool(save_path),
          saveas=save_path, saveframes=tag_saveframes, show=False, **animate_kwargs))
    # end
    if resolved_show:
      import matplotlib.pyplot as plt
      plt.show()
    # end
    return
  # end

  save_path = saveas
  if ds.batch and not save_path and not saveframes:
    save_path = f"{ds.prefix}.gif"
  # end
  pg.animate(*pool, save=save or bool(save_path), saveas=save_path,
      saveframes=saveframes, show=resolved_show, **animate_kwargs)
# end
