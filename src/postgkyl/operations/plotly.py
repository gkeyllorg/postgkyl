"""The ``plotly`` verb -- terminal; hands the dataset to the Plotly render backend.

Mirrors ``operations/plot.py``: point-value forms (nodal/quad) plot
directly via ``materialize_for_render``; raw modal coefficients refuse (the
user chooses ``.interpolate()``, ``.to_nodal()``, or ``.to_quad()`` first).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl import render

from ._materialize import materialize_for_render

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState
# end


def plotly(data: "GDataState", *, squeeze: bool = False,
    scatter: bool = False, marker_radius: float = 4.0,
    markerstyle: str = "circle", diverging: bool = False,
    xscale: float = 1.0, xshift: float = 0.0,
    yscale: float = 1.0, yshift: float = 0.0,
    zscale: float = 1.0, zshift: float = 0.0,
    cmin: float | None = None, cmax: float | None = None,
    style: str | None = None, background: str = "dark",
    invert_cmap: bool = False, legend: bool = True,
    colorbar: bool = True, xlabel: str | None = None,
    ylabel: str | None = None, zlabel: str | None = None,
    clabel: str | None = None, title: str | None = None,
    logx: bool = False, logy: bool = False, logz: bool = False,
    logc: bool = False, aspect: str | None = None,
    showgrid: bool = True, color: str | None = None,
    opacity: float | None = 1.0, maximum_points_per_axis: int = 0,
    surface_count: int = 32, cylindrical_to_cartesian: bool = False,
    cmap: str | None = None, save: bool = False,
    saveas: str | None = None, show: bool = False,
    azimuthal_angle: float = 0.0, polar_angle: float = 85.0,
    rotation_period: float = 40.0, fps: int = 1):
  """Render a single dataset with Plotly (terminal verb; see ``render.plotly``).

  ``save``/``saveas``/``show`` (and the rotating-export camera parameters)
  are handled entirely by ``render.plotly``, and default to inert -- pass
  ``show=True`` for an auto-rotating browser preview, or
  ``save=True``/``saveas=...`` to write it. Returns the Plotly figure.

  Args:
    data: Point-value dataset to render.
    squeeze: Remove singleton spatial axes.
    scatter: Render three-dimensional values as points.
    marker_radius: Scatter marker radius.
    markerstyle: Scatter marker symbol.
    diverging: Use a diverging colormap.
    xscale: Horizontal-coordinate scale.
    xshift: Horizontal-coordinate shift.
    yscale: Vertical-coordinate scale.
    yshift: Vertical-coordinate shift.
    zscale: Third-coordinate or value scale.
    zshift: Third-coordinate or value shift.
    cmin: Color-scale lower bound.
    cmax: Color-scale upper bound.
    style: Plot style name.
    background: Background theme name.
    invert_cmap: Reverse the colormap.
    legend: Draw a legend.
    colorbar: Draw a color bar.
    xlabel: Horizontal-axis label.
    ylabel: Vertical-axis label.
    zlabel: Third-axis label.
    clabel: Color-bar label.
    title: Figure title.
    logx: Use a logarithmic horizontal axis.
    logy: Use a logarithmic vertical axis.
    logz: Use a logarithmic third axis.
    logc: Use logarithmic color values.
    aspect: Scene aspect mode.
    showgrid: Draw grid lines.
    color: Explicit trace color.
    opacity: Trace opacity.
    maximum_points_per_axis: Downsample limit per axis.
    surface_count: Volume-rendering surface count.
    cylindrical_to_cartesian: Convert cylindrical coordinates for display.
    cmap: Colormap name.
    save: Save to an automatically derived name.
    saveas: Explicit output path.
    show: Open the plot in a browser.
    azimuthal_angle: Initial camera azimuth.
    polar_angle: Initial camera polar angle.
    rotation_period: Camera rotation period for animated exports.
    fps: Frames per second for animated exports.
  """
  return render.plotly(materialize_for_render(data), squeeze=squeeze,
      scatter=scatter, marker_radius=marker_radius, markerstyle=markerstyle,
      diverging=diverging, xscale=xscale, xshift=xshift, yscale=yscale,
      yshift=yshift, zscale=zscale, zshift=zshift, cmin=cmin, cmax=cmax,
      style=style, background=background, invert_cmap=invert_cmap,
      legend=legend, colorbar=colorbar, xlabel=xlabel, ylabel=ylabel,
      zlabel=zlabel, clabel=clabel, title=title, logx=logx, logy=logy,
      logz=logz, logc=logc, aspect=aspect, showgrid=showgrid, color=color,
      opacity=opacity, maximum_points_per_axis=maximum_points_per_axis,
      surface_count=surface_count,
      cylindrical_to_cartesian=cylindrical_to_cartesian, cmap=cmap,
      save=save, saveas=saveas, show=show, azimuthal_angle=azimuthal_angle,
      polar_angle=polar_angle, rotation_period=rotation_period, fps=fps)
# end
