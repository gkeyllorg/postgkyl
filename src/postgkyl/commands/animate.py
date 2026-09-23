import os
import shutil
import tempfile
from matplotlib.animation import FuncAnimation, FFMpegWriter
from multiprocessing import Pool
from PIL import Image
import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from postgkyl.utils import verb_print, set_frame
import postgkyl.output.plot

# Formats written through ffmpeg (PIL cannot produce these video containers).
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def _save_frame_worker(args):
  """Worker for parallel frame saving; each process creates its own figure."""
  matplotlib.use("Agg")
  frame_idx, frame_data, kwargs, prefix, dpi, figsize = args
  fig = plt.figure(figsize=figsize)
  _update(0, [frame_data], fig, kwargs)
  plt.savefig(f"{prefix:s}_{frame_idx:d}.png", dpi=dpi)
  plt.close(fig)
# end


def _save_frames(data_list, num_frames, prefix, kwargs, figsize, fig=None):
  """Save frames as PNGs, using parallel workers when nproc > 1."""
  if kwargs["nproc"] > 1:
    args_list = [(i, data_list[i], kwargs, prefix, kwargs["dpi"], figsize)
        for i in range(num_frames)]
    with Pool(kwargs["nproc"]) as pool:
      pool.map(_save_frame_worker, args_list)
    # end
  else:
    for i in range(num_frames):
      _update(i, data_list, fig, kwargs)
      plt.savefig(f"{prefix:s}_{i:d}.png", dpi=kwargs["dpi"])
    # end
  # end
# end


def _compile_movie(frame_files, output_file, fps, duration, ctx):
  """Compile PNG frames into an animation."""
  ext = os.path.splitext(output_file)[1].lower()
  verb_print(ctx,f"Creating {output_file}...")
  if ext in (".gif", ".webp", ".apng"):
    images = [Image.open(f) for f in frame_files]
    images[0].save(
        output_file, save_all=True, append_images=images[1:],
        duration=duration, loop=0, optimize=False,
    )
  elif ext in VIDEO_EXTS:
    # PIL cannot write video containers; use matplotlib's ffmpeg writer.
    # duration is in milliseconds per frame, so fall back to it when fps is unset.
    movie_fps = fps if fps else 1.0e3 / duration
    writer = FFMpegWriter(fps=movie_fps)
    first = Image.open(frame_files[0])
    dpi = 100
    fig = plt.figure(figsize=(first.width / dpi, first.height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    with writer.saving(fig, output_file, dpi):
      for frame_file in frame_files:
        ax.clear()
        ax.axis("off")
        ax.imshow(Image.open(frame_file), origin="upper")
        writer.grab_frame()
      # end
    # end
    plt.close(fig)
  else:
    raise ValueError(f"Unsupported output format: {ext}")

  verb_print(ctx,f"{output_file} created.")
# end


def _update(frame, data, fig, kwargs):
  fig.clear()
  kwargs["figure"] = fig

  #global range function is called every frame to set scale limits for frame plot
  if kwargs["multiblock"] and kwargs["float"]:
    vmin, vmax, num_dims = globalrange(data[frame], kwargs)
    if num_dims == 1:
      kwargs["ymin"] = vmin
      kwargs["ymax"] = vmax
    else:
      kwargs["zmin"] = vmin
      kwargs["zmax"] = vmax
    # end
  # end

  #main plotting loop
  for i, dat in enumerate(data[frame]):
    kwargs["title"] = ""
    if not kwargs["notitle"]:
      if dat.ctx.get("frame") is not None:
        kwargs["title"] = f"{kwargs['title']:s} frame: {dat.ctx['frame']:d} "
      # end
      if dat.ctx.get("time") is not None:
        kwargs["title"] = f"{kwargs['title']:s} time: {dat.ctx['time']:.4e}"
      # end
    # end

    if i == 0:
      if kwargs.get("arg"):
        im = postgkyl.output.plot(dat, kwargs["arg"], **kwargs)
      else:
        im = postgkyl.output.plot(dat, **kwargs)
      # end
    else:
      kwargs_ncb = kwargs.copy()
      kwargs_ncb["colorbar"] = False
      if kwargs.get("arg"):
        im = postgkyl.output.plot(dat, kwargs["arg"], **kwargs_ncb)
      else:
        im = postgkyl.output.plot(dat, **kwargs_ncb)
      # end
    # end
  # end
  return im
# end

#Finds global minima and maxima for all inputed data objects
#also incorporates cutoffglobalrange
def globalrange(data,kwargs):
  vmin = float("inf")
  vmax = float("-inf")
  v_extrema = np.array([])
  for dat in data:
    num_dims = dat.get_num_dims()
    if num_dims == 1:
      val = dat.get_values()*kwargs["yscale"]
    else:
      val = dat.get_values()*kwargs["zscale"]
    # end
    if vmin > np.nanmin(val):
      vmin = np.nanmin(val)
    if vmax < np.nanmax(val):
      vmax = np.nanmax(val)
    # end
    v_extrema = np.append(v_extrema, np.nanmin(val))
    v_extrema = np.append(v_extrema, np.nanmax(val))
  # end
  v_extrema = np.sort(v_extrema)
  if kwargs["cutoffglobalrange"]:
    boundary = 100 * (1 - kwargs["cutoffglobalrange"]) / 2
    vmax = np.percentile(v_extrema, 100 - boundary)
    vmin = np.percentile(v_extrema, boundary)
    return vmin, vmax, num_dims
  else:
    return vmin, vmax, num_dims
  # end
# end


@click.command()
@click.option("--use", "-u", default=None, help="Specify a tag to plot.")
@click.option("--grouptags", is_flag=True, help="Group coresponding tagged frames.")
@click.option("--squeeze", "-p", is_flag=True, help="Squeeze the components into one panel.")
@click.option("--subplots", "-b", is_flag=True, help="Make subplots from multiple datasets.")
@click.option("--nsubplotrow", "nSubplotRow", type=click.INT,
    help="Manually set the number of rows for subplots.")
@click.option("--nsubplotcol", "nSubplotCol", type=click.INT,
    help="Manually set the number of columns for subplots.")
@click.option("--transpose", is_flag=True, help="Transpose axes.")
@click.option("--contour", "-c", is_flag=True, help="Make contour plot.")
@click.option("--clevels", type=click.STRING,
    help="Specify levels for contours: either integer or start:end:nlevels")
@click.option("--quiver", "-q", is_flag=True, help="Make quiver plot.")
@click.option("--streamline", "-l", is_flag=True, help="Make streamline plot.")
@click.option("--sdensity", type=click.FLOAT, help="Control density of the streamlines.")
@click.option("--arrowstyle", type=click.STRING, help="Set the style for streamline arrows.")
@click.option("--group", "-g", type=click.Choice(["0", "1"]), help="Switch to group mode.")
@click.option("--scatter", "-s", is_flag=True, help="Make scatter plot.")
@click.option("--markersize", type=click.FLOAT, help="Set marker size for scatter plots.")
@click.option("--linewidth", type=click.FLOAT, help="Set the linewidth.")
@click.option("--linestyle", type=click.Choice(["solid", "dashed", "dotted", "dashdot"]),
    help="Set the linestyle.")
@click.option("--color", type=click.STRING, help="Set color when available.")
@click.option("--style", help="Specify Matplotlib style file (default: Postgkyl).")
@click.option("--diverging", "-d", is_flag=True, help="Switch to diverging colormesh mode.")
@click.option("--arg", type=click.STRING, help="Additional plotting arguments, e.g., '*--'.")
@click.option("--fix-aspect", "-a", "fixaspect", is_flag=True,
    help="Enforce the same scaling on both axes.")
@click.option("--logx", is_flag=True, help="Set x-axis to log scale.")
@click.option("--logy", is_flag=True, help="Set y-axis to log scale.")
@click.option("--logz", is_flag=True, help="Set values of 2D plot to log scale.")
@click.option("--xshift", default=0.0, type=click.FLOAT, show_default=True,
    help="Value to shift the x-axis.")
@click.option("--yshift", default=0.0, type=click.FLOAT, show_default=True,
    help="Value to shift the y-axis.")
@click.option("--zshift", default=0.0, type=click.FLOAT, show_default=True,
    help="Value to shift the z-axis.")
@click.option("--xscale", default=1.0, type=click.FLOAT, show_default=True,
    help="Value to scale the x-axis.")
@click.option("--yscale", default=1.0, type=click.FLOAT, show_default=True,
    help="Value to scale the y-axis.")
@click.option("--zscale", default=1.0, type=click.FLOAT, show_default=True,
    help="Value to scale the z-axis.")
@click.option("--float", is_flag=True,
    help="Choose min/max levels based on current frame (i.e., each frame uses a different color range).")
@click.option("--xmax", default=None, type=click.FLOAT, help="Set maximal x-value.")
@click.option("--xmin", default=None, type=click.FLOAT, help="Set minimal x-values.")
@click.option("--ymax", default=None, type=click.FLOAT, help="Set maximal y-value.")
@click.option("--ymin", default=None, type=click.FLOAT, help="Set minimal y-values.")
@click.option("--zmax", default=None, type=click.FLOAT, help="Set maximal z-value.")
@click.option("--zmin", default=None, type=click.FLOAT, help="Set minimal z-values.")
@click.option("--xlim", default=None, type=click.STRING,
    help="Set limits for the x-coordinate (lower,upper).")
@click.option("--ylim", default=None, type=click.STRING,
    help="Set limits for the y-coordinate (lower,upper).")
@click.option("--zlim", default=None, type=click.STRING,
    help="Set limits for the z-coordinate (lower,upper).")
@click.option("--cutoffglobalrange", "-cogr", default=None, type=click.FLOAT,
              help="Specify middle percentile of data extrema to set y/z limits to")
@click.option("--legend/--no-legend", default=True, help="Show legend.")
@click.option("--colorbar/--no-colorbar", default=True,
              help="Show colorbar (2D animations), no colorbar improves animation performance")
@click.option("--force-legend", "forcelegend", is_flag=True,
    help="Force legend even when plotting a single dataset.")
@click.option("-x", "--xlabel", type=click.STRING, help="Specify a x-axis label.")
@click.option("-y", "--ylabel", type=click.STRING, help="Specify a y-axis label.")
@click.option("--clabel", type=click.STRING, help="Specify a label for colorbar.")
@click.option("--title", type=click.STRING, help="Specify a title.")
@click.option("--notitle", is_flag=True, help="Do not show title.")
@click.option("-i", "--interval", default=100, help="Specify the animation interval.")
@click.option("--save", is_flag=True, help="Save figure as PNG.")
@click.option("--saveas", type=click.STRING, default=None, help="Name to save the plot as.")
@click.option("--fps", type=click.INT, help="Specify frames per second for saving.")
@click.option("--dpi", type=click.INT, help="DPI (resolution) for output.")
@click.option("--edgecolors", "-e", type=click.STRING, help="Set color for cell edges.")
@click.option("--showgrid/--no-showgrid", default=True, help="Show grid-lines.")
@click.option("--collected", is_flag=True,
   help="Animate a dataset that has been collected, i.e. a single dataset with time taken to be the first index.")
@click.option("--hashtag", is_flag=True, help="Turns on the pgkyl hashtag!")
@click.option("--show/--no-show", default=True, help="Turn showing of the plot ON and OFF.")
@click.option("--saveframes", type=click.STRING,
    help="Save individual frames as PNGs.")
@click.option("--nproc", default=1, type=click.INT, show_default=True,
    help="Number of parallel processes for frame generation.")
@click.option("--tmpdir", default=None, type=click.STRING, show_default=True,
    help="Directory to place the temporary directory for parallel frame generation.")
@click.option("--figsize", help="Comma-separated values for x and y size.")
@click.option("-m", "--multiblock", is_flag=True, help="Plots blocks from each frame together")
@click.pass_context
def animate(ctx, **kwargs):
  """Animate the actively loaded dataset and show resulting plots in a loop.

  Typically, the datasets are loaded using wildcard/regex feature of the -f option to
  the main pgkyl executable.
  """
  verb_print(ctx, "Starting animate")
  data = ctx.obj["data"]

  # Accept str or path-like input for --saveas (e.g. a pathlib.Path).
  if kwargs["saveas"]:
    kwargs["saveas"] = str(kwargs["saveas"])
  # end
  supported_exts = (".gif", ".webp", ".apng") + VIDEO_EXTS
  if kwargs["saveas"] and not kwargs["saveas"].lower().endswith(supported_exts):
    raise click.ClickException(
        "Unsupported output format for --saveas; please use one of: "
        + ", ".join(supported_exts) + ".")
  # end
  # Video containers are written through ffmpeg, which must be on the PATH.
  if kwargs["saveas"] and kwargs["saveas"].lower().endswith(VIDEO_EXTS) \
      and shutil.which("ffmpeg") is None:
    raise click.ClickException(
        "ffmpeg is required to write " + ", ".join(VIDEO_EXTS) + " files but was "
        "not found. Please install ffmpeg or choose a .gif output instead.")
  # end

  if kwargs["xlim"]:
    kwargs["xmin"] = float(kwargs["xlim"].split(",")[0])
    kwargs["xmax"] = float(kwargs["xlim"].split(",")[1])
  # end
  if kwargs["ylim"]:
    kwargs["ymin"] = float(kwargs["ylim"].split(",")[0])
    kwargs["ymax"] = float(kwargs["ylim"].split(",")[1])
  # end
  if kwargs["zlim"]:
    kwargs["zmin"] = float(kwargs["zlim"].split(",")[0])
    kwargs["zmax"] = float(kwargs["zlim"].split(",")[1])
  # end

  if not kwargs["float"] and not kwargs["grouptags"]:
    vmin, vmax, num_dims = globalrange(data.iterator(kwargs["use"]), kwargs)
    if num_dims == 1:
      if kwargs["ymin"] is None:
        kwargs["ymin"] = vmin
      # end
      if kwargs["ymax"] is None:
        kwargs["ymax"] = vmax
      # end
    else:
      if kwargs["zmin"] is None:
        kwargs["zmin"] = vmin
      # end
      if kwargs["zmax"] is None:
        kwargs["zmax"] = vmax
      # end
    # end
  # end

  anims = []
  figs = []
  kwargs["legend"] = False

  figsize = None
  if kwargs["figsize"]:
    figsize = (int(kwargs["figsize"].split(",")[0]), int(kwargs["figsize"].split(",")[1]))
  # end

  # PIL requires duration in miliseconds.
  duration = int(1.0e3 / kwargs["fps"]) if kwargs["fps"] else kwargs["interval"]

  set_figure = False
  min_size = np.nan
  yset = False

  if kwargs["grouptags"]:
    #runs animation for each tag
    for tag in data.tag_iterator(kwargs["use"]):
      num_datasets = int(data.get_num_datasets(tag=tag))
      min_size = int(np.nanmin((min_size, num_datasets)))
    # end

    tag_iterator = list(data.tag_iterator(kwargs["use"]))
    kwargs["legend"] = True
    set_figure = True
    fig_num = int(0)

    for tag in tag_iterator:
      #sets scale for each tag animation
      vmin, vmax, num_dims = globalrange(data.iterator(tag), kwargs)
      if num_dims == 1:
        kwargs["ymin"] = vmin
        kwargs["ymax"] = vmax
        yset = True
      else:
        if yset: #so that ymin,ymax of 1D anim don't affect 2D anim
          kwargs["ymin"] = None
          kwargs["ymax"] = None
        # end
        kwargs["zmin"] = vmin
        kwargs["zmax"] = vmax
      # end

      #creating min list of lists (non-multiblock case)
      data_list = []
      for dat in data.iterator(tag):
        data_list.append([dat])
      # end
      figs.append(plt.figure(fig_num, figsize=figsize))
      fig_num += 1

      num_frames = int(np.nanmin((min_size, len(data_list))))
      file_name = f"anim_{tag:s}.gif" if tag is not None else "anim.gif"
      if kwargs["saveas"]:
        file_name = str(kwargs["saveas"])
      # end

      if kwargs["saveframes"]:
        # Save PNGs, then optionally compile a movie.
        _save_frames(data_list, num_frames, kwargs["saveframes"], kwargs, figsize, figs[-1])
        if kwargs["save"] or kwargs["saveas"]:
          frame_files = [f"{kwargs['saveframes']}_{i}.png" for i in range(num_frames)]
          _compile_movie(frame_files, file_name, kwargs["fps"], duration, ctx)
        # end
        kwargs["show"] = False
      elif kwargs["nproc"] > 1:
        # Parallel: use a temp dir, compile, then clean up.
        with tempfile.TemporaryDirectory(dir=kwargs["tmpdir"]) as tmpdir:
          tmp_prefix = os.path.join(tmpdir, "frame")
          _save_frames(data_list, num_frames, tmp_prefix, kwargs, figsize)
          frame_files = [f"{tmp_prefix}_{i}.png" for i in range(num_frames)]
          _compile_movie(frame_files, file_name, kwargs["fps"], duration, ctx)
        # end
        kwargs["show"] = False
      else:
        anims.append(
            FuncAnimation(figs[-1], _update, num_frames,
                fargs=(data_list, figs[-1], kwargs), interval=kwargs["interval"],
                blit=False)
        )
        if kwargs["save"] or kwargs["saveas"]:
          anims[-1].save(file_name, writer="ffmpeg", fps=kwargs["fps"], dpi=kwargs["dpi"])
        # end
      # end
    # end
  #animation code for multiblock case
  elif kwargs["multiblock"]:

    #set ctx frames for all data objects
    sorted_frame_list = set_frame(ctx)

    #create main list of lists (multiblock case)
    data_list = []
    #organize data objects so each interior list includes blocks from one frame
    for frame in sorted_frame_list:
      frame_data_list = [dat for dat in data.iterator(kwargs["use"]) if dat.ctx["frame"] == frame]
      data_list.append(frame_data_list)
    # end

    figs.append(plt.figure(figsize=figsize))
    #makes default color blue in 1D cases, this prevents blocks from having different colors
    if (not kwargs["color"] and data_list[0][0].get_num_dims() == 1):
      kwargs["color"] = "tab:blue"
    # end

    num_frames = int(np.nanmin((min_size, len(data_list))))
    file_name = kwargs["saveas"] if kwargs["saveas"] else "anim.gif"

    if kwargs["saveframes"]:
      _save_frames(data_list, num_frames, kwargs["saveframes"], kwargs, figsize, figs[-1])
      if kwargs["save"] or kwargs["saveas"]:
        frame_files = [f"{kwargs['saveframes']}_{i}.png" for i in range(num_frames)]
        _compile_movie(frame_files, file_name, kwargs["fps"], duration, ctx)
      # end
      kwargs["show"] = False
    elif kwargs["nproc"] > 1:
      with tempfile.TemporaryDirectory(dir=kwargs["tmpdir"]) as tmpdir:
        tmp_prefix = os.path.join(tmpdir, "frame")
        _save_frames(data_list, num_frames, tmp_prefix, kwargs, figsize)
        frame_files = [f"{tmp_prefix}_{i}.png" for i in range(num_frames)]
        _compile_movie(frame_files, file_name, kwargs["fps"], duration, ctx)
      # end
      kwargs["show"] = False
    else:
      anims.append(
          FuncAnimation(figs[-1], _update, num_frames,
              fargs=(data_list, figs[-1], kwargs), interval=kwargs["interval"],
              blit=False)
      )
      if kwargs["save"] or kwargs["saveas"]:
        anims[-1].save(file_name, writer="ffmpeg", fps=kwargs["fps"], dpi=kwargs["dpi"])
      # end
    # end

  else:

    #create main list of lists (non-multiblock case)
    data_list = []
    for dat in data.iterator(kwargs["use"]):
      data_list.append([dat])
    # end
    if set_figure:
      figs.append(plt.figure(fig_num, figsize=figsize))
    else:
      figs.append(plt.figure(figsize=figsize))
    # end

    num_frames = int(np.nanmin((min_size, len(data_list))))
    file_name = kwargs["saveas"] if kwargs["saveas"] else "anim.gif"

    if kwargs["saveframes"]:
      _save_frames(data_list, num_frames, kwargs["saveframes"], kwargs, figsize, figs[-1])
      if kwargs["save"] or kwargs["saveas"]:
        frame_files = [f"{kwargs['saveframes']}_{i}.png" for i in range(num_frames)]
        _compile_movie(frame_files, file_name, kwargs["fps"], duration, ctx)
      # end
      kwargs["show"] = False
    elif kwargs["nproc"] > 1:
      with tempfile.TemporaryDirectory(dir=kwargs["tmpdir"]) as tmpdir:
        tmp_prefix = os.path.join(tmpdir, "frame")
        _save_frames(data_list, num_frames, tmp_prefix, kwargs, figsize)
        frame_files = [f"{tmp_prefix}_{i}.png" for i in range(num_frames)]
        _compile_movie(frame_files, file_name, kwargs["fps"], duration, ctx)
      # end
      kwargs["show"] = False
    else:
      anims.append(
          FuncAnimation(figs[-1], _update, num_frames,
              fargs=(data_list, figs[-1], kwargs), interval=kwargs["interval"],
              blit=False)
      )
      if kwargs["save"] or kwargs["saveas"]:
        anims[-1].save(file_name, writer="ffmpeg", fps=kwargs["fps"], dpi=kwargs["dpi"])
      # end
    # end
  # end

  if kwargs["show"]:
    plt.show()
  # end
  verb_print(ctx, "Finishing animate")
