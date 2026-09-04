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
    materialize_point_values,
)

from . import matplotlib as backend
from ._ffmpeg import require_ffmpeg

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
  frames = [([materialize_point_values(item)] if isinstance(item, GDataState)
             else [materialize_point_values(dat) for dat in item])
            for item in items]
  if not frames:
    raise ValueError("animate: no datasets to animate.")
  return frames


def _frame_value_range(frames: list[list["GDataState"]],
                       cutoff: float | None = None,
                       *,
                       yscale: float = 1.0,
                       zscale: float = 1.0) -> tuple[float, float]:
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
      scaled = dat.values * (yscale if dat.num_dims == 1 else zscale)
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
  notitle = kwargs.pop("notitle", False)
  if not notitle and kwargs.get("title") is None:
    dat0 = frame[0]
    parts = []
    if dat0.ctx.get("frame") is not None:
      parts.append(f"frame: {dat0.ctx['frame']:d}")
    if dat0.ctx.get("time") is not None:
      parts.append(f"time: {dat0.ctx['time']:.4e}")
    kwargs["title"] = " ".join(parts)
  return backend.plot(*frame, figure=fig, clear=True, show=False, **kwargs)


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
    images = [Image.open(f) for f in frame_files]
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
    first = Image.open(frame_files[0])
    dpi = 100
    fig = plt.figure(figsize=(first.width / dpi, first.height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    try:
      with writer.saving(fig, output_file, dpi):
        for frame_file in frame_files:
          ax.clear()
          ax.axis("off")
          ax.imshow(Image.open(frame_file))
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
            multiblock: bool = False,
            grouptags: bool = False,
            interval: int = 100,
            fixed_range: bool = True,
            cutoffglobalrange: float | None = None,
            notitle: bool = False,
            show: bool = True,
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
    multiblock: Force datasets with the same frame index into one frame.
    grouptags: Build a separate animation for each dataset tag.
    interval: live-animation delay between frames, in milliseconds.
    fixed_range: hold a constant value/color scale across every frame
      (``ymin``/``ymax``/``zmin``/``zmax``).
    cutoffglobalrange: clip the fixed range to this central percentile band
      (see ``_frame_value_range``); ``None`` uses the true min/max.
    notitle: suppress the per-frame frame/time title.
    show: open a live window (the ``FuncAnimation`` path only).
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
    ``tmpdir`` compile path, the compiled output path is returned instead.

  Raises:
    ValueError: no datasets to animate, or an unsupported ``saveas``
      extension.
    RuntimeError: saving to a video container without ffmpeg on ``PATH``.
  """
  items = list(data)
  if items and grouptags and all(
      isinstance(item, GDataState) for item in items):
    tags: dict[str, list[GDataState]] = {}
    for item in items:
      tags.setdefault(item.tag, []).append(item)

    def suffixed(path, tag):
      if path is None:
        return None
      stem, extension = os.path.splitext(path)
      return f"{stem}_{tag}{extension}"

    return [
        animate(tagged,
                multiblock=multiblock,
                interval=interval,
                fixed_range=fixed_range,
                cutoffglobalrange=cutoffglobalrange,
                notitle=notitle,
                show=show,
                save=save,
                saveas=suffixed(saveas, tag),
                fps=fps,
                dpi=dpi,
                saveframes=suffixed(saveframes, tag),
                figsize=figsize,
                nproc=nproc,
                tmpdir=tmpdir) for tag, tagged in tags.items()
    ]

  frames = _normalize_frames(items, multiblock=multiblock)
  plot_kwargs = {}
  plot_kwargs["notitle"] = notitle

  if fixed_range:
    vmin, vmax = _frame_value_range(frames,
                                    cutoffglobalrange,
                                    yscale=plot_kwargs.get("yscale", 1.0),
                                    zscale=plot_kwargs.get("zscale", 1.0))
    # Applied as both the 1-D y-limits (ymin/ymax) and the 2-D color range
    # (zmin/zmax) -- whichever the frame's dimensionality actually uses.
    plot_kwargs.setdefault("ymin", vmin)
    plot_kwargs.setdefault("ymax", vmax)
    plot_kwargs.setdefault("zmin", vmin)
    plot_kwargs.setdefault("zmax", vmax)

  num_frames = len(frames)
  duration = 1.0e3 / fps if fps else float(interval)
  out_file = saveas or "anim.gif"
  if not os.path.splitext(out_file)[1]:
    out_file += ".gif"

  if saveframes:
    frame_files = _save_frames(frames,
                               saveframes,
                               dpi=dpi,
                               figsize=figsize,
                               plot_kwargs=plot_kwargs,
                               nproc=nproc)
    if save or saveas:
      _compile_movie(frame_files, out_file, fps=fps, duration=duration)
    return frame_files

  if nproc > 1:
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
  if save or saveas:
    import matplotlib as mpl

    mpl.rcParams["animation.ffmpeg_path"] = require_ffmpeg("animate")
    anim.save(out_file, writer="ffmpeg", fps=fps, dpi=dpi)
  if show:
    plt.show()
  return anim
