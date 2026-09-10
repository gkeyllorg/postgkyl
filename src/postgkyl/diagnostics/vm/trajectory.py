"""Particle-trajectory animation.

Ported from ``src_bak/postgkyl/apps/trajectory.py``. Animates one or more
position (+ optional velocity) time series in 3-D. Typer options become
explicit keyword-only parameters; the old CLI's tag-indexed dataset stack
(``ctx.obj.data``) is replaced by passing the datasets directly. Saving is
the caller's choice: this returns the ``FuncAnimation`` object -- call
``.save(path)`` on it, or ``plt.show()`` after creating it to display it
live.

Each dataset's ``grid[0]`` is expected to hold one time stamp per position
sample -- a Gkeyll dynvector's grid convention (``io/gkyl_reader.py``'s
``_read_t2_v1``: ``grid = [time]`` with ``len(time) == values.shape[0]``,
unlike a field file's ``num_cells + 1`` edges), the same convention
``src_bak`` read via ``dat.get_grid()[0]``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

if TYPE_CHECKING:
  from ...gdatastate.gdatastate import GDataState

_COLORS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9")


def _masked(coord: np.ndarray, lo: float | None,
            hi: float | None) -> np.ndarray:
  """Replace out-of-``[lo, hi]`` entries of ``coord`` with NaN, so they are
  simply not drawn (masking, not clipping -- matches ``src_bak``)."""
  out = coord
  if lo is not None:
    out = np.where(out > lo, out, np.nan)
  if hi is not None:
    out = np.where(out < hi, out, np.nan)
  return out


def _update(i, ax, datasets, leap, no_velocity, xmin, xmax, ymin, ymax, zmin,
            zmax):
  """``FuncAnimation`` frame callback: redraw every dataset's trajectory up
  to (and current position at) frame ``i``."""
  ax.cla()
  t_idx = int(i * leap)
  time = None

  for s, dataset in enumerate(datasets):
    time = dataset.grid[0]
    coords = dataset.values
    color = _COLORS[s % len(_COLORS)]

    x = _masked(coords[:, 0], xmin, xmax)
    y = _masked(coords[:, 1], ymin, ymax)
    z = _masked(coords[:, 2], zmin, zmax)

    ax.plot(x, y, z, color=color)
    ax.scatter(x[t_idx], y[t_idx], z[t_idx], color=color)

    if not no_velocity and dataset.num_comps == 6:
      if t_idx + leap >= len(time):
        dt = time[-1] - time[t_idx]
      else:
        dt = time[int(t_idx + leap)] - time[t_idx]
      dx = coords[t_idx, 3] * dt
      dy = coords[t_idx, 4] * dt
      dz = coords[t_idx, 5] * dt
      ax.plot([x[t_idx], x[t_idx] + dx], [y[t_idx], y[t_idx] + dy],
              [z[t_idx], z[t_idx] + dz],
              color=color)

  if time is not None:
    ax.set_title(f"T: {time[t_idx]:.4e}")
  ax.set_xlabel("$z_0$")
  ax.set_ylabel("$z_1$")
  ax.set_zlabel("$z_2$")
  ax.set_xlim3d(xmin, xmax)
  ax.set_ylim3d(ymin, ymax)
  ax.set_zlim3d(zmin, zmax)


def trajectory(
    *datasets: "GDataState",
    fixaspect: bool = False,
    interval: int = 100,
    no_velocity: bool = False,
    numframes: int | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    elevation: float | None = None,
    azimuth: float | None = None,
) -> FuncAnimation:
  """Animate one or more particle trajectories in 3-D.

  Args:
    datasets: One or more datasets, each holding a position (3-component,
      ``x, y, z``) or position+velocity (6-component,
      ``x, y, z, vx, vy, vz``) time series, with ``grid[0]`` one time stamp
      per sample (the dynvector convention).
    fixaspect: Enforce the same scaling on all three axes.
    interval: Animation frame interval, in milliseconds.
    no_velocity: Do not draw a velocity vector at the current position.
    numframes: Number of animation frames; ``None`` uses one frame per
      sample. When given, samples are subsampled evenly (by
      ``floor(num_samples / numframes)``).
    xmin: Optional lower x bound; outside points are masked.
    xmax: Optional upper x bound; outside points are masked.
    ymin: Optional lower y bound; outside points are masked.
    ymax: Optional upper y bound; outside points are masked.
    zmin: Optional lower z bound; outside points are masked.
    zmax: Optional upper z bound; outside points are masked.
    elevation: Initial 3-D elevation angle in degrees.
    azimuth: Initial 3-D azimuth angle in degrees.

  Returns:
    The ``FuncAnimation``.

  Raises:
    ValueError: if no datasets are given.
  """
  if not datasets:
    raise ValueError("trajectory() requires at least one dataset.")

  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")

  num_pos = int(datasets[0].num_cells[0])
  leap = 1
  if numframes:
    leap = int(math.floor(num_pos / numframes))
    num_pos = int(numframes)

  anim = FuncAnimation(fig,
                       _update,
                       num_pos,
                       fargs=(ax, datasets, leap, no_velocity, xmin, xmax, ymin,
                              ymax, zmin, zmax),
                       interval=interval)

  ax.view_init(elev=elevation, azim=azimuth)
  if fixaspect:
    # Equal-scale 3-D axes: modern Matplotlib's Axes3D takes a box aspect
    # ratio (`set_box_aspect`), not the numeric `aspect=` src_bak passed to
    # `plt.setp` (that spelling only ever worked for 2-D axes).
    ax.set_box_aspect((1.0, 1.0, 1.0))

  return anim
