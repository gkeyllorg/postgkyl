"""The canonical Matplotlib animation callable and its private helpers.

``pg.animate``, ``operations.animate``, and the generated CLI all resolve to
the one public function in this module. It owns dataset grouping and
materialization as well as ``FuncAnimation`` / saved frames / movie compile.

The module is separate from ``matplotlib.py`` because it owns the one
external-process dependency in this layer -- ``ffmpeg`` -- reached through
Matplotlib's ``FFMpegWriter``/``Animation.save``. Every entry point that needs
it resolves a binary via ``_ffmpeg.require_ffmpeg`` up front and raises a clear
``RuntimeError`` instead of failing deep inside the writer.
"""

from __future__ import annotations

import os.path
from collections.abc import Iterable
from typing import Annotated, TYPE_CHECKING

import numpy as np

from postgkyl.cli_spec import (
    CliType,
    CommandSpec,
    Execution,
    PipelineInput,
    ResultPolicy,
    Section,
    command,
)
from postgkyl.gdatastate import (
    GDataState,
    group_blocks,
    group_frames,
)

from . import matplotlib as backend
from ._ffmpeg import require_ffmpeg
from ._prep import materialize_plot_data

if TYPE_CHECKING:
  from matplotlib.figure import Figure

# Formats written through ffmpeg; PIL handles the rest (gif/webp/apng).
_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def _normalize_frames(data,
                      *,
                      multiblock: bool = False) -> list[list["GDataState"]]:
  """Group a flat input when requested and materialize every frame dataset."""
  items = list(data)
  if items and all(isinstance(item, GDataState) for item in items):
    items = group_frames(items) if multiblock else group_blocks(items)
  frames = [([materialize_plot_data(item)] if isinstance(item, GDataState) else
             [materialize_plot_data(dat) for dat in item]) for item in items]
  if not frames or any(not frame for frame in frames):
    raise ValueError("animate: no datasets to animate.")
  return frames


def _frame_value_range(frames: list[list["GDataState"]],
                       cutoff: float | None = None,
                       *,
                       yscale: float = 1.0,
                       zscale: float = 1.0,
                       yshift: float = 0.0,
                       zshift: float = 0.0) -> tuple[float, float]:
  """Value range spanning every dataset in every frame.

  Each dataset is scaled by ``yscale`` (1-D) or ``zscale`` (2-D) before its
  extrema are taken, matching the scale ``matplotlib.plot`` applies when it
  actually draws the values -- otherwise a fixed range computed here would
  not match the plotted (scaled) data.

  With ``cutoff`` (a central fraction in ``(0, 1]``), the range is clipped
  to that percentile band of the per-dataset extrema instead of the true
  min/max -- useful when a few outlier frames would otherwise wash out the
  color/y-axis scale for the rest of the animation.
  """
  extrema = []
  for frame in frames:
    for dat in frame:
      scaled = ((dat.values + yshift) * yscale if dat.num_dims == 1 else
                (dat.values + zshift) * zscale)
      extrema.append(np.nanmin(scaled))
      extrema.append(np.nanmax(scaled))
  extrema = np.array(extrema)
  vmin, vmax = float(extrema.min()), float(extrema.max())
  if cutoff:
    boundary = 100.0 * (1.0 - cutoff) / 2.0
    vmax = float(np.percentile(extrema, 100.0 - boundary))
    vmin = float(np.percentile(extrema, boundary))
  return vmin, vmax


def _draw_frame(frame: list["GDataState"], fig: "Figure", plot_kwargs: dict):
  """Redraw one frame (a list of datasets drawn together) onto ``fig``.

  When the caller hasn't given an explicit ``title``, it is generated from
  the first dataset's ``ctx`` (frame index and time) unless
  ``plot_kwargs['notitle']`` is set; an explicit ``title`` is always
  respected and shown on every frame.
  """
  kwargs = dict(plot_kwargs)
  if kwargs.pop("variable_range", False):
    _apply_value_range([frame], kwargs)
  kwargs.pop("cutoffglobalrange", None)
  if kwargs.pop("subplots", False):
    step = 2 if kwargs.get("quiver") or kwargs.get("streamline") else 1
    kwargs["num_axes"] = sum(dat.num_comps // step for dat in frame)
  notitle = kwargs.pop("notitle", False)
  if notitle:
    kwargs["title"] = ""
  if not notitle and kwargs.get("title") is None:
    dat0 = frame[0]
    parts = []
    if dat0.ctx.get("frame") is not None:
      parts.append(f"frame: {dat0.ctx['frame']:d}")
    if dat0.ctx.get("time") is not None:
      parts.append(f"time: {dat0.ctx['time']:.4e}")
    kwargs["title"] = " ".join(parts)
  return backend.plot(*frame, figure=fig, clear=True, no_show=True, **kwargs)


def _render_frame(index: int, frames: list[list["GDataState"]], fig: "Figure",
                  plot_kwargs: dict):
  """``FuncAnimation``'s per-frame callback: draw ``frames[index]``."""
  return _draw_frame(frames[index], fig, plot_kwargs)


def _save_frame_worker(args) -> str:
  """One frame, one process (see ``_save_frames``'s ``nproc`` path). Each
  worker builds its own figure -- Matplotlib figures are not shared across
  processes."""
  index, frame, plot_kwargs, prefix, dpi, figsize = args
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig = plt.figure(figsize=figsize)
  try:
    _draw_frame(frame, fig, plot_kwargs)
    path = f"{prefix}_{index}.png"
    fig.savefig(path, dpi=dpi)
  finally:
    plt.close(fig)
  return path


def _save_frames(frames: list[list["GDataState"]],
                 prefix: str,
                 *,
                 dpi: int | None = None,
                 figsize=None,
                 plot_kwargs: dict | None = None,
                 nproc: int = 1) -> list[str]:
  """Write ``<prefix>_<i>.png`` for every frame.

  Sequentially (``nproc == 1``), one figure is reused across every frame.
  With ``nproc > 1``, frames are split across a :class:`multiprocessing.Pool`
  of that many worker processes, each with its own figure.
  """
  plot_kwargs = plot_kwargs or {}
  if nproc > 1:
    from multiprocessing import Pool

    args_list = [(i, frames[i], plot_kwargs, prefix, dpi, figsize)
                 for i in range(len(frames))]
    with Pool(nproc) as pool:
      return pool.map(_save_frame_worker, args_list)

  import matplotlib.pyplot as plt

  fig = plt.figure(figsize=figsize)
  paths = []
  try:
    for i in range(len(frames)):
      _draw_frame(frames[i], fig, plot_kwargs)
      path = f"{prefix}_{i}.png"
      fig.savefig(path, dpi=dpi)
      paths.append(path)
  finally:
    plt.close(fig)
  return paths


def _compile_movie(frame_files: list[str],
                   output_file: str,
                   *,
                   fps: int | None = None,
                   duration: float = 100.0) -> None:
  """Compile PNG frames into an animation: PIL for gif/webp/apng, the
  Matplotlib ffmpeg writer for video containers. ``duration`` is the
  per-frame time in milliseconds, used when ``fps`` is not given."""
  from PIL import Image

  ext = os.path.splitext(output_file)[1].lower()
  if ext in (".gif", ".webp", ".apng"):
    from contextlib import ExitStack

    with ExitStack() as stack:
      images = [stack.enter_context(Image.open(f)) for f in frame_files]
      images[0].save(output_file,
                     save_all=True,
                     append_images=images[1:],
                     duration=duration,
                     loop=0,
                     optimize=False)
    return
  if ext in _VIDEO_EXTS:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    mpl.rcParams["animation.ffmpeg_path"] = require_ffmpeg("animate")
    movie_fps = fps if fps else 1.0e3 / duration
    writer = FFMpegWriter(fps=movie_fps)
    with Image.open(frame_files[0]) as first:
      width, height = first.size
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    try:
      with writer.saving(fig, output_file, dpi):
        for frame_file in frame_files:
          ax.clear()
          ax.axis("off")
          with Image.open(frame_file) as frame:
            ax.imshow(frame)
          writer.grab_frame()
    finally:
      plt.close(fig)
    return
  raise ValueError(f"animate: unsupported output format {ext!r}")


@command(
    CommandSpec(Section.RENDER,
                Execution.TERMINAL_ALL,
                result=ResultPolicy.SILENT))
def animate(data: Annotated[Iterable[GDataState | Iterable[GDataState]],
                            PipelineInput()],
            *,
            use: str | None = None,
            collected: bool = False,
            squeeze: bool = False,
            subplots: bool = False,
            num_subplot_row: int | None = None,
            num_subplot_col: int | None = None,
            transpose: bool = False,
            contour: bool = False,
            clevels: str | None = None,
            quiver: bool = False,
            streamline: bool = False,
            sdensity: float = 1.0,
            arrowstyle: str | None = None,
            group: int | None = None,
            scatter: bool = False,
            markersize: float | None = None,
            linewidth: float | None = None,
            linestyle: str | None = None,
            color: str | None = None,
            style: str | None = None,
            diverging: bool = False,
            arg: str | None = None,
            fixaspect: bool = False,
            logx: bool = False,
            logy: bool = False,
            logz: bool = False,
            xshift: float = 0.0,
            xscale: float = 1.0,
            yshift: float = 0.0,
            yscale: float = 1.0,
            zshift: float = 0.0,
            zscale: float = 1.0,
            xmin: float | None = None,
            xmax: float | None = None,
            ymin: float | None = None,
            ymax: float | None = None,
            zmin: float | None = None,
            zmax: float | None = None,
            xlim: Annotated[tuple[float, float] | str | None,
                            CliType(tuple[float, float] | None)] = None,
            ylim: Annotated[tuple[float, float] | str | None,
                            CliType(tuple[float, float] | None)] = None,
            zlim: Annotated[tuple[float, float] | str | None,
                            CliType(tuple[float, float] | None)] = None,
            no_legend: bool = False,
            no_colorbar: bool = False,
            forcelegend: bool = False,
            xlabel: str | None = None,
            ylabel: str | None = None,
            clabel: str | None = None,
            title: str | None = None,
            edgecolors: str | None = None,
            no_showgrid: bool = False,
            hashtag: bool = False,
            multiblock: bool = False,
            grouptags: bool = False,
            interval: int = 100,
            variable_range: bool = False,
            cutoffglobalrange: float | None = None,
            notitle: bool = False,
            no_show: bool = False,
            save: bool = False,
            saveas: str | None = None,
            fps: int | None = None,
            dpi: int | None = None,
            saveframes: str | None = None,
            figsize: Annotated[tuple[float, float] | str | None,
                               CliType(tuple[float, float] | None)] = None,
            nproc: int = 1,
            tmpdir: str | None = None):
  """Animate a sequence of frames, one frame per dataset (or dataset group).

  Args:
    data: a flat iterable of datasets (each becomes a single-dataset frame),
      or an iterable of frames where each frame is itself a list of
      datasets drawn together (overlaid, as in ``matplotlib.plot``).
    use: Select datasets with this tag.
    collected: Treat the leading axis of each dataset as time.
    squeeze: Draw all components in one panel.
    subplots: Draw each dataset in a separate block of panels.
    num_subplot_row: Number of subplot rows.
    num_subplot_col: Number of subplot columns.
    transpose: Transpose the display axes.
    contour: Draw contours.
    clevels: Contour count or start:end:count levels.
    quiver: Draw vector arrows.
    streamline: Draw streamlines.
    sdensity: Streamline density.
    arrowstyle: Streamline arrow style.
    group: Draw lineouts along coordinate 0 or 1.
    scatter: Draw point markers without connecting lines.
    markersize: Marker size in points.
    linewidth: Line width.
    linestyle: Matplotlib line style.
    color: Line or vector color.
    style: Matplotlib style name or file.
    diverging: Use a diverging colormap.
    arg: Matplotlib format string, for example *--.
    fixaspect: Use equal scaling on the display axes.
    logx: Use logarithmic x scaling.
    logy: Use logarithmic y scaling.
    logz: Use logarithmic z scaling.
    xshift: X shift, applied before scaling.
    xscale: X scale factor.
    yshift: Y shift, applied before scaling.
    yscale: Y scale factor.
    zshift: Z shift, applied before scaling.
    zscale: Z scale factor.
    xmin: X minimum limit.
    xmax: X maximum limit.
    ymin: Y minimum limit.
    ymax: Y maximum limit.
    zmin: Z minimum limit.
    zmax: Z maximum limit.
    xlim: X limits as a pair or comma-separated string; overrides individual bounds.
    ylim: Y limits as a pair or comma-separated string; overrides individual bounds.
    zlim: Z limits as a pair or comma-separated string; overrides individual bounds.
    no_legend: Suppress legends.
    no_colorbar: Suppress colorbars.
    forcelegend: Show a legend even for one curve.
    xlabel: Override the x label.
    ylabel: Override the y label.
    clabel: Override the c label.
    title: Title shown on every frame.
    edgecolors: Mesh cell edge color.
    no_showgrid: Suppress grid lines.
    hashtag: Display the Postgkyl hashtag.
    multiblock: Force datasets with the same frame index into one frame.
    grouptags: Build a separate animation for each dataset tag.
    interval: live-animation delay between frames, in milliseconds.
    variable_range: Recompute the value/color scale for every frame instead
      of holding ``ymin``/``ymax``/``zmin``/``zmax`` constant.
    cutoffglobalrange: clip the fixed range to this central percentile band
      (see ``_frame_value_range``); ``None`` uses the true min/max.
    notitle: suppress the per-frame frame/time title.
    no_show: Do not open a live window (the ``FuncAnimation`` path only).
    save: write to ``saveas`` (or ``anim.gif``) after building the frames.
    saveas: output path; its extension selects the writer (``.gif``/
      ``.webp``/``.apng`` via PIL, ``.mp4``/``.mov``/``.avi``/``.mkv`` via
      ffmpeg).
    fps: frames per second for the saved movie; defaults from ``interval``.
    dpi: resolution for saved frames/movies.
    saveframes: when given, write ``<saveframes>_<i>.png`` for every frame
      instead of building a live ``FuncAnimation``.
    figsize: figure size in inches, forwarded to ``matplotlib.plot``.
    nproc: parallel worker processes for frame generation (``saveframes``,
      or the ``tmpdir``-backed compile path below); ``1`` renders sequentially
      in-process.
    tmpdir: directory for the temporary frame directory used when ``nproc``
      is greater than 1 and ``saveframes`` is not given (frames are written
      there, compiled into the output, then discarded).

  Returns:
    The list of written frame paths when ``saveframes`` is set; otherwise
    the ``FuncAnimation`` (keep a reference -- Matplotlib does not keep the
    live animation alive for you). When ``nproc`` renders through the
    ``tmpdir`` compile path, or when saving WebP/APNG without ``saveframes``,
    the compiled output path is returned instead. With ``grouptags``, returns
    one result per tag.

  Raises:
    ValueError: no datasets to animate, or an unsupported ``saveas``
      extension.
    RuntimeError: saving to a video container without ffmpeg on ``PATH``.
  """
  if interval <= 0 or (fps is not None and fps <= 0) or nproc < 1:
    raise ValueError("animate: interval, fps, and nproc must be positive")
  if cutoffglobalrange is not None and not 0 < cutoffglobalrange <= 1:
    raise ValueError("animate: cutoffglobalrange must be in (0, 1]")
  if group not in (None, 0, 1):
    raise ValueError("animate: group must be 0 or 1")
  if isinstance(figsize, str):
    figsize = tuple(float(value) for value in figsize.split(","))
  plot_kwargs = dict(squeeze=squeeze,
                     subplots=subplots,
                     num_subplot_row=num_subplot_row,
                     num_subplot_col=num_subplot_col,
                     transpose=transpose,
                     contour=contour,
                     clevels=clevels,
                     quiver=quiver,
                     streamline=streamline,
                     sdensity=sdensity,
                     arrowstyle=arrowstyle,
                     scatter=scatter,
                     markersize=markersize,
                     linewidth=linewidth,
                     linestyle=linestyle,
                     color=color,
                     style=style,
                     diverging=diverging,
                     fixaspect=fixaspect,
                     logx=logx,
                     logy=logy,
                     logz=logz,
                     xshift=xshift,
                     xscale=xscale,
                     yshift=yshift,
                     yscale=yscale,
                     zshift=zshift,
                     zscale=zscale,
                     xmin=xmin,
                     xmax=xmax,
                     ymin=ymin,
                     ymax=ymax,
                     zmin=zmin,
                     zmax=zmax,
                     no_legend=no_legend,
                     no_colorbar=no_colorbar,
                     forcelegend=forcelegend,
                     xlabel=xlabel,
                     ylabel=ylabel,
                     clabel=clabel,
                     title=title,
                     edgecolors=edgecolors,
                     no_showgrid=no_showgrid,
                     hashtag=hashtag,
                     lineouts=group,
                     args=[arg] if arg else None,
                     notitle=notitle,
                     variable_range=variable_range,
                     cutoffglobalrange=cutoffglobalrange)
  for axis, limits in zip("xyz", (xlim, ylim, zlim)):
    if limits is not None:
      if isinstance(limits, str):
        limits = tuple(float(value) for value in limits.split(","))
      lower, upper = limits
      plot_kwargs[f"{axis}min"] = lower
      plot_kwargs[f"{axis}max"] = upper

  items = list(data)
  if use is not None:
    selected = []
    for item in items:
      if isinstance(item, GDataState):
        if item.tag == use:
          selected.append(item)
      else:
        frame = [dat for dat in item if dat.tag == use]
        if frame:
          selected.append(frame)
    items = selected
  if collected:
    expanded = []
    for item in items:
      if not isinstance(item, GDataState) or item.backend != "numpy":
        raise ValueError("animate: collected requires NumPy datasets")
      for index, values in enumerate(item.values):
        expanded.append(
            item._result(list(item.grid[1:]),
                         values,
                         frame=index,
                         time=float(item.grid[0][index])))
    items = expanded
  groups = {}
  if grouptags:
    for item in items:
      if not isinstance(item, GDataState):
        raise ValueError("animate: grouptags requires a flat dataset sequence")
      groups.setdefault(item.tag, []).append(item)
  else:
    groups[None] = items
  if not groups:
    raise ValueError("animate: no datasets to animate.")

  results = []
  for tag, datasets in groups.items():
    frames = _normalize_frames(datasets, multiblock=multiblock)
    options = dict(plot_kwargs)
    if multiblock and color is None and frames[0][0].num_dims == 1:
      options["color"] = "tab:blue"
    output = os.fspath(saveas) if saveas is not None else "anim.gif"
    prefix = os.fspath(saveframes) if saveframes is not None else None
    if grouptags and tag is not None:
      stem, extension = os.path.splitext(output)
      output = f"{stem}_{tag}{extension}"
      if prefix is not None:
        prefix = f"{prefix}_{tag}"
    results.append(
        _animate_frames(frames,
                        plot_kwargs=options,
                        interval=interval,
                        save=save or saveas is not None,
                        saveas=output,
                        fps=fps,
                        dpi=dpi,
                        saveframes=prefix,
                        figsize=figsize,
                        nproc=nproc,
                        tmpdir=tmpdir))
  if not no_show and saveframes is None and nproc == 1:
    import matplotlib.pyplot as plt
    plt.show()
  return results if grouptags else results[0]


def _apply_value_range(frames, kwargs):
  """Fill unspecified value bounds without constraining spatial coordinates."""
  for dimension in (1, 2):
    selected = [[dat for dat in frame if dat.num_dims == dimension]
                for frame in frames]
    selected = [frame for frame in selected if frame]
    if not selected:
      continue
    low, high = _frame_value_range(selected,
                                   kwargs.get("cutoffglobalrange"),
                                   yscale=kwargs.get("yscale", 1.0),
                                   zscale=kwargs.get("zscale", 1.0),
                                   yshift=kwargs.get("yshift", 0.0),
                                   zshift=kwargs.get("zshift", 0.0))
    if dimension == 2 and kwargs.get("diverging"):
      high = max(abs(low), abs(high))
      low = -high
    axis = ("x" if kwargs.get("transpose") else "y") if dimension == 1 else "z"
    for bound, value in (("min", low), ("max", high)):
      if kwargs.get(axis + bound) is None:
        kwargs[axis + bound] = value


def _animate_frames(frames, *, plot_kwargs, interval, save, saveas, fps, dpi,
                    saveframes, figsize, nproc, tmpdir):
  """Render one normalized sequence through the selected output path."""
  if not plot_kwargs["variable_range"]:
    _apply_value_range(frames, plot_kwargs)
  num_frames = len(frames)
  duration = 1.0e3 / fps if fps else float(interval)
  out_file = saveas or "anim.gif"
  if not os.path.splitext(out_file)[1]:
    out_file += ".gif"

  ext = os.path.splitext(out_file)[1].lower()
  if ext not in (".gif", ".webp", ".apng") + _VIDEO_EXTS:
    raise ValueError(f"animate: unsupported output format {ext!r}")
  if (save or nproc > 1) and ext in _VIDEO_EXTS:
    require_ffmpeg("animate")

  if saveframes:
    frame_files = _save_frames(frames,
                               saveframes,
                               dpi=dpi,
                               figsize=figsize,
                               plot_kwargs=plot_kwargs,
                               nproc=nproc)
    if save:
      _compile_movie(frame_files, out_file, fps=fps, duration=duration)
    return frame_files

  if nproc > 1 or (save and ext in (".webp", ".apng")):
    # No standing PNGs requested -- render into a scratch directory, compile,
    # then discard it. Mirrors the ``saveframes`` path with parallel workers,
    # so it always produces the compiled output (there is no live window to
    # hand parallel workers' figures back to).
    import tempfile

    with tempfile.TemporaryDirectory(dir=tmpdir) as tmp:
      tmp_prefix = f"{tmp}/frame"
      frame_files = _save_frames(frames,
                                 tmp_prefix,
                                 dpi=dpi,
                                 figsize=figsize,
                                 plot_kwargs=plot_kwargs,
                                 nproc=nproc)
      _compile_movie(frame_files, out_file, fps=fps, duration=duration)
    return out_file

  import matplotlib.pyplot as plt
  from matplotlib.animation import FuncAnimation

  fig = plt.figure(figsize=figsize)
  anim = FuncAnimation(fig,
                       _render_frame,
                       num_frames,
                       fargs=(frames, fig, plot_kwargs),
                       interval=interval,
                       blit=False)
  if save:
    import matplotlib as mpl

    writer = "pillow"
    if ext in _VIDEO_EXTS:
      mpl.rcParams["animation.ffmpeg_path"] = require_ffmpeg("animate")
      writer = "ffmpeg"
    anim.save(out_file, writer=writer, fps=fps or 1.0e3 / interval, dpi=dpi)
  return anim
