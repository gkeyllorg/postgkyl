"""The ``plot`` terminal verb over one dataset or a block-aware collection.

Point-value forms plot directly at their physical point locations. Modal
coefficients refuse until the caller explicitly chooses ``interpolate()``,
``to_nodal()``, or ``to_quad()``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import TYPE_CHECKING

from postgkyl import render
from postgkyl.gdatastate import flatten_datasets, group_blocks

from ._materialize import materialize_for_render

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState
# end


def _indexed_path(path: str | None, index: int, indexed: bool) -> str | None:
  if path is None or not indexed:
    return path
  # end
  stem, extension = os.path.splitext(path)
  return f"{stem}_{index}{extension}"
# end


def plot(data: "GDataState | Iterable[GDataState]", *,
    multiblock: bool = False, squeeze: bool = False,
    transpose: bool = False, contour: bool = False, surface: bool = False,
    diverging: bool = False, lineouts: int | None = None,
    xmin: float | None = None, xmax: float | None = None,
    xscale: float = 1.0, xshift: float = 0.0,
    ymin: float | None = None, ymax: float | None = None,
    yscale: float = 1.0, yshift: float = 0.0,
    zmin: float | None = None, zmax: float | None = None,
    zscale: float = 1.0, zshift: float = 0.0,
    style: str | None = None, legend: bool = True, colorbar: bool = True,
    xlabel: str | None = None, ylabel: str | None = None,
    clabel: str | None = None, title: str | None = None,
    logx: bool = False, logy: bool = False, logz: bool = False,
    fixaspect: bool = False, showgrid: bool = True, hashtag: bool = False,
    xkcd: bool = False, markersize: float | None = None,
    linewidth: float | None = None, cmap: str | None = None,
    save: bool = False, saveas: str | None = None, dpi: int = 200,
    show: bool = True):
  """Render selected data with automatic multiblock-family grouping.

  Args:
    data: A dataset or selected dataset collection.
    multiblock: Force every selected dataset onto one figure. Without this
      override, blocks of one field are grouped automatically while distinct
      fields and frames receive separate figures.
    squeeze: Remove singleton spatial axes before choosing a plot layout.
    transpose: Swap horizontal and vertical display axes.
    contour: Draw two-dimensional values as contours.
    surface: Draw two-dimensional values as a three-dimensional surface.
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
    style: Matplotlib style name or style-file path.
    legend: Draw legends for line plots.
    colorbar: Draw color bars for field plots.
    xlabel: Horizontal-axis label override.
    ylabel: Vertical-axis label override.
    clabel: Color-bar label override.
    title: Figure-title override.
    logx: Use logarithmic horizontal coordinates.
    logy: Use a logarithmic vertical axis.
    logz: Use logarithmic values or colors.
    fixaspect: Use equal physical scaling on coordinate axes.
    showgrid: Draw plot grid lines.
    hashtag: Prefix labels with a hash marker.
    xkcd: Draw using Matplotlib's XKCD context.
    markersize: Line-marker size.
    linewidth: Line width.
    cmap: Matplotlib colormap name.
    save: Save to an automatically derived output name.
    saveas: Explicit image output path.
    dpi: Saved-image resolution.
    show: Display the figure interactively.

  Returns:
    One Matplotlib figure when the selection is one field, otherwise a list
    containing one figure per field/frame family.
  """
  states = flatten_datasets([data])
  if not states:
    raise ValueError("nothing to plot")
  # end
  families = [states] if multiblock else group_blocks(states)
  indexed = len(families) > 1 or (
      save and saveas is not None and not os.path.splitext(saveas)[1])
  figures = []
  for index, family in enumerate(families):
    figures.append(render.plot(
        *[materialize_for_render(dataset) for dataset in family],
        squeeze=squeeze, transpose=transpose, contour=contour,
        surface=surface, diverging=diverging, lineouts=lineouts,
        xmin=xmin, xmax=xmax, xscale=xscale, xshift=xshift,
        ymin=ymin, ymax=ymax, yscale=yscale, yshift=yshift,
        zmin=zmin, zmax=zmax, zscale=zscale, zshift=zshift,
        style=style, legend=legend, colorbar=colorbar, xlabel=xlabel,
        ylabel=ylabel, clabel=clabel, title=title, logx=logx, logy=logy,
        logz=logz, fixaspect=fixaspect, showgrid=showgrid, hashtag=hashtag,
        xkcd=xkcd, markersize=markersize, linewidth=linewidth, cmap=cmap,
        save=save, saveas=_indexed_path(saveas, index, indexed), dpi=dpi,
        show=show))
  # end
  return figures[0] if len(figures) == 1 else figures
# end
