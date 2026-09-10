"""Dataset -> plottable-array preparation, shared by every render backend.

Private to ``render/``: this is the one concern the old tree split across
``utils/load_plot_data.py`` (dataset -> grid/values/dimensionality) and
``utils/axis_and_grid_prep.py`` (squeeze collapsed axes, resolve axis/colorbar
label defaults). Here it collapses to a single function over
:class:`~postgkyl.gdatastate.gdatastate.GDataState` -- the new container already exposes
``grid``/``values``/``num_dims`` uniformly, so there is no dual "GData or
tuple" input to dispatch on (contrast the old ``load_plot_data``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState


def default_axis_labels(num_dims: int) -> list[str]:
  """Default per-axis labels ``$z_0$``, ``$z_1$``, ... (mathtext)."""
  return [rf"$z_{i}$" for i in range(num_dims)]


def format_axis_label(label: str, shift: float, scale: float) -> str:
  """Annotate an axis label with its shift/scale, matching the old style."""
  if shift != 0.0 and scale != 1.0:
    return rf"({label:s} + {shift:.2e}) $\times$ {scale:.2e}"
  if shift != 0.0:
    return rf"{label:s} + {shift:.2e}"
  if scale != 1.0:
    return rf"{label:s} $\times$ {scale:.2e}"
  return label


def squeeze_collapsed_axes(
    grid: list[np.ndarray],
    values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Drop grid axes with exactly one cell (e.g. a ``select()``-ed coordinate).

  Curvilinear (multi-dimensional, ``.map()``-produced) coordinate arrays
  cannot simply be indexed on the dropped axis -- every coordinate array
  spans all dimensions jointly -- so each is averaged along it first (a
  size-1 axis is unaffected by the mean); the now-redundant axis entry is
  then removed from the coordinate list.

  Args:
    grid: One nodal (edge) coordinate array per dimension.
    values: Cell values, shape ``(*cells, num_comps)``.

  Returns:
    ``(grid, values)`` with every size-1 axis removed.
  """
  num_dims = len(grid)
  cells = values.shape[:num_dims]
  drop = [d for d in range(num_dims) if cells[d] <= 1]
  if not drop:
    return list(grid), values

  grid = [np.asarray(g) for g in grid]
  if any(g.ndim > 1 for g in grid):
    for d in range(num_dims):
      for i in reversed(drop):
        grid[d] = np.mean(grid[d], axis=i)
  for i in reversed(drop):
    grid.pop(i)
  values = np.squeeze(values, tuple(drop))
  return grid, values


def subplot_grid(num_comps: int,
                 num_rows: int | None = None,
                 num_cols: int | None = None) -> tuple[int, int]:
  """Choose a near-square ``(rows, cols)`` layout for ``num_comps`` panels."""
  if num_rows is not None:
    return num_rows, int(np.ceil(num_comps / num_rows))
  if num_cols is not None:
    return int(np.ceil(num_comps / num_cols)), num_cols
  sr = np.sqrt(num_comps)
  if sr == np.ceil(sr):
    return int(sr), int(sr)
  if np.ceil(sr) * np.floor(sr) >= num_comps:
    return int(np.floor(sr)), int(np.ceil(sr))
  return int(np.ceil(sr)), int(np.ceil(sr))


@dataclass(frozen=True)
class PlotPanel:
  """Squeezed, label-resolved view of one dataset, ready for a render call."""
  grid: list[np.ndarray]
  values: np.ndarray
  num_dims: int
  num_comps: int
  xlabel: str
  ylabel: str
  clabel: str


def resolve_axis_labels(*,
                        xlabel: str | None,
                        ylabel: str | None,
                        zlabel: str | None,
                        clabel: str,
                        num_dims: int,
                        xshift: float = 0.0,
                        yshift: float = 0.0,
                        zshift: float = 0.0,
                        xscale: float = 1.0,
                        yscale: float = 1.0,
                        zscale: float = 1.0) -> tuple[str, str, str, str]:
  """Infer default ``$z_i$`` labels and apply shift/scale annotations.

  Shared by the 2-D (``matplotlib``, no real ``z`` axis) and 3-D
  (``plotly``, ``z`` is a genuine coordinate) backends: with ``num_dims``
  dimensions, defaults are ``z_0..z_{num_dims-1}`` distributed across
  ``xlabel``/``ylabel``/``zlabel`` in that order (only as many as apply).
  """
  labels = default_axis_labels(max(num_dims, 3))
  if xlabel is None:
    xlabel = labels[0] if num_dims > 0 else ""
  if ylabel is None:
    ylabel = labels[1] if num_dims > 1 else ""
  if zlabel is None:
    zlabel = labels[2] if num_dims > 2 else labels[-1]
  xlabel = format_axis_label(xlabel, xshift, xscale)
  ylabel = format_axis_label(ylabel, yshift, yscale)
  zlabel = format_axis_label(zlabel, zshift, zscale)
  if zscale != 1.0:
    clabel = (rf"{clabel:s} $\times$ {zscale:.3e}"
              if clabel else rf"$\times$ {zscale:.3e}")
  return xlabel, ylabel, zlabel, clabel


def prep_plot_data(data: "GDataState",
                   *,
                   xlabel: str | None = None,
                   ylabel: str | None = None,
                   clabel: str = "",
                   xshift: float = 0.0,
                   yshift: float = 0.0,
                   zshift: float = 0.0,
                   xscale: float = 1.0,
                   yscale: float = 1.0,
                   zscale: float = 1.0) -> PlotPanel:
  """Squeeze collapsed axes and resolve axis/colorbar label defaults.

  Args:
    data: The dataset to prepare (point-value/NumPy-backed; the ``plot``
      verb has already bridged any modal data through its NumPy shadow).
    xlabel: Explicit x-axis label; auto-derived (``$z_0$``) when ``None``.
    ylabel: Explicit y-axis label; auto-derived (``$z_1$``) when ``None``
      and the (squeezed) dataset is 2-D, else empty.
    clabel: Colorbar label base text; annotated with ``zscale`` when it is
      not 1.
    xshift, yshift, zshift: Additive shifts recorded in the axis labels
      (the caller applies them to the plotted arrays).
    xscale, yscale, zscale: Multiplicative scales recorded in the axis
      labels (the caller applies them to the plotted arrays).

  Returns:
    A :class:`PlotPanel` with the squeezed grid/values and resolved labels.
  """
  grid, values = squeeze_collapsed_axes(list(data.grid), data.values)
  num_dims = len(grid)
  xlabel, ylabel, _zlabel, clabel = resolve_axis_labels(xlabel=xlabel,
                                                        ylabel=ylabel,
                                                        zlabel="",
                                                        clabel=clabel,
                                                        num_dims=num_dims,
                                                        xshift=xshift,
                                                        yshift=yshift,
                                                        zshift=zshift,
                                                        xscale=xscale,
                                                        yscale=yscale,
                                                        zscale=zscale)

  return PlotPanel(grid=grid,
                   values=values,
                   num_dims=num_dims,
                   num_comps=values.shape[-1],
                   xlabel=xlabel,
                   ylabel=ylabel,
                   clabel=clabel)
