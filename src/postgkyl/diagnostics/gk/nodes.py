"""Gyrokinetic grid-node diagnostic.

Ported from ``src_bak/postgkyl/apps/gk_nodes.py``: plots the nodes of a
(possibly multiblock, possibly mapc2p) grid, connected by their cell edges,
with an optional overlay of the poloidal-flux contours/colormap and a vacuum-
vessel wall outline.
"""

from __future__ import annotations

from itertools import cycle
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from postgkyl.io.geometry import is_geo_mapc2p

from . import utils

_XY_LABEL_FONT_SIZE = 17
_TITLE_FONT_SIZE = 17
_TICK_FONT_SIZE = 14
_COLORBAR_LABEL_FONT_SIZE = 14


def nodes_to_RZ(nodes: np.ndarray,
                is_mapc2p: bool) -> tuple[np.ndarray, np.ndarray]:
  """Compute the major-radius/vertical-location (R, Z) variables from a
  grid-nodes array.

  Args:
    nodes: Node coordinates, shape ``(*cell_shape, 3)`` holding Cartesian
      (X, Y, Z) for mapc2p geometry, or ``(*cell_shape, 2+)`` holding
      (R, Z, [phi]) otherwise. A size-1 ``y`` axis is sliced out (at index
      0) for 3-D cell shapes.
    is_mapc2p: Whether ``nodes`` holds Cartesian coordinates (True) or
      already (R, Z, ...) coordinates (False).

  Returns:
    ``(majorR, vertZ)``.
  """
  yidx = 0  # Index in the y direction to slice 3-D node arrays at.

  nx_nod = np.shape(nodes)
  cdim = np.size(nx_nod) - 1
  cart_dim = 3

  lo_idx = [[0 for _ in range(cdim)] + [cd] for cd in range(cart_dim)]
  up_idx = [[nx_nod[d] for d in range(cdim)] + [cd + 1]
            for cd in range(cart_dim)]

  if cdim == 3:
    for cd in range(cart_dim):
      lo_idx[cd][1] = yidx
      up_idx[cd][1] = yidx + 1

  slices = [[slice(lo_idx[cd][d], up_idx[cd][d]) for d in range(cdim + 1)]
            for cd in range(cart_dim)]

  if is_mapc2p:
    cart_x = [np.squeeze(nodes[tuple(slices[d])]) for d in range(cart_dim)]
    major_r = np.sqrt(np.power(cart_x[0], 2) + np.power(cart_x[1], 2))
    vert_z = cart_x[2]
  else:
    major_r = np.squeeze(nodes[tuple(slices[0])])
    vert_z = np.squeeze(nodes[tuple(slices[1])])

  return major_r, vert_z


def multib_tag(base: str, block_idx: int, num_blocks: int) -> str:
  """Tag a per-block artifact, adding a ``_b<idx>`` suffix only when there
  is more than one block."""
  return f"{base}_b{block_idx}" if num_blocks > 1 else base


def _parse_levels(clevels: str | None, cnlevels: int) -> np.ndarray | int:
  if clevels is None:
    return cnlevels
  if ":" in clevels:
    s = clevels.split(":")
    return np.linspace(float(s[0]), float(s[1]), int(s[2]))
  return np.array([float(v) for v in clevels.split(",") if v])


def _surface_rz(values: np.ndarray,
                mapc2p: bool) -> tuple[np.ndarray, np.ndarray]:
  """Convert an unsliced surface-node array to R and Z."""
  if mapc2p:
    return np.hypot(values[..., 0], values[..., 1]), values[..., 2]
  return values[..., 0], values[..., 1]


def _read_node_coordinates(file_name: str, expected: tuple[int, ...]):
  """Read a point-node file with the shape implied by the corner grid."""
  if not os.path.isfile(file_name):
    raise ValueError(f"Missing geometry nodes file: '{file_name}'.")
  _, values, gdata = utils.read_gfile(file_name)
  if values.shape[:-1] != expected:
    raise ValueError(f"Geometry nodes file '{file_name}' has shape "
                     f"{values.shape[:-1]}; expected {expected}.")
  if not np.all(np.isfinite(values)) or not np.any(values):
    raise ValueError(
        f"Geometry nodes file '{file_name}' is nonfinite or all zero.")
  return _surface_rz(values, is_geo_mapc2p(gdata.ctx))


def _read_surface(prefix: str, direction: int, corner_shape: tuple[int, ...]):
  expected = tuple(n if d == direction else 2 * (n - 1)
                   for d, n in enumerate(corner_shape))
  return _read_node_coordinates(f"{prefix}-geo_surf{direction}_nodes.gkyl",
                                expected)


def _draw_curves(ax, corners, surfaces, color, toroidal_sampling):
  """Connect each edge's corners and its two Gauss surface samples."""
  cr, cz = corners
  ndim = cr.ndim
  if ndim == 2:
    planes = [(cr, cz, surfaces[0], surfaces[1])]
  else:
    sample_indices = (range(surfaces[0][0].shape[1])
                      if toroidal_sampling == "all" else (0, ))
    planes = []
    for sample in sample_indices:
      cell, quad = divmod(sample, 2)
      weight = (0.5 + (-1 if quad == 0 else 1) / (2 * np.sqrt(3)))
      r = (1 - weight) * cr[:, cell, :] + weight * cr[:, cell + 1, :]
      z = (1 - weight) * cz[:, cell, :] + weight * cz[:, cell + 1, :]
      s0 = tuple(v[:, sample, :] for v in surfaces[0])
      s2 = tuple(v[:, sample, :] for v in surfaces[2])
      planes.append((r, z, s0, s2))
  segments = []
  for r, z, s0, s1 in planes:
    for i in range(r.shape[0]):
      for j in range(r.shape[1] - 1):
        segments.append([(r[i, j], z[i, j]), (s0[0][i, 2 * j], s0[1][i, 2 * j]),
                         (s0[0][i, 2 * j + 1], s0[1][i, 2 * j + 1]),
                         (r[i, j + 1], z[i, j + 1])])
    for j in range(r.shape[1]):
      for i in range(r.shape[0] - 1):
        segments.append([(r[i, j], z[i, j]), (s1[0][2 * i, j], s1[1][2 * i, j]),
                         (s1[0][2 * i + 1, j], s1[1][2 * i + 1, j]),
                         (r[i + 1, j], z[i + 1, j])])
  ax.add_collection(LineCollection(segments, color=color))


def nodes(
    name: str,
    *,
    path: str = "./",
    multib: str = "-10",
    nodes_file: str | None = None,
    int_nodes: bool = False,
    surf_nodes: bool = False,
    toroidal_sampling: str = "nearest",
    psi_file: str | None = None,
    wall_file: str | None = None,
    contour: bool = False,
    clevels: str | None = None,
    cnlevels: int = 11,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xlabel: str = "R (m)",
    ylabel: str = "Z (m)",
    zlabel: str = r"$\psi$",
    title: str | None = None,
    left_indent: float = 0.0,
    add_width: float = 0.0,
    multib_unicolor: bool = False,
    no_show: bool = False,
    saveas: str | None = None,
) -> plt.Figure:
  """Plot the nodes of a (possibly multiblock) grid, with optional overlays.

  Args:
    name: Simulation name (also the file prefix).
    path: Directory holding the simulation output.
    multib: ``"-10"`` (default) for a single block; ``"-1"`` to discover
      every block; otherwise a comma list or ``'start:stop[:step]'`` slice
      of block indices.
    nodes_file: Override for the ``<name>-geo_corn_nodes.gkyl`` file
      (``*`` stands for the block index); an absolute path is used as-is.
    int_nodes: Mark interior quadrature nodes without connecting them.
      For 3-D data, use the first toroidal quadrature plane.
    surf_nodes: Draw curved cell edges through surface nodes and mark them.
      Uses surf0 and surf1 in 2-D, or surf0 and surf2 in a 3-D R/Z slice.
    toroidal_sampling: For 3-D curved edges, ``'nearest'`` uses the first
      toroidal Gauss plane and ``'all'`` draws every plane.
    psi_file: Optional poloidal-flux file to overlay (interpolated p2 tensor
      basis); an absolute path is used as-is.
    wall_file: Optional CSV ``(R, Z)`` vacuum-vessel wall outline to overlay.
    contour: Draw ``psi_file`` as contour lines instead of a colormesh.
    clevels: Contour levels: comma-separated values, or a
      ``'start:stop:nlevels'`` range; defaults to ``cnlevels`` automatic
      levels when ``None``.
    cnlevels: Number of automatic contour levels (ignored if ``clevels`` is
      given).
    xlim: Optional horizontal-axis limits.
    ylim: Optional vertical-axis limits.
    xlabel: Horizontal-axis label.
    ylabel: Vertical-axis label.
    zlabel: Poloidal-flux colorbar label.
    title: Figure title.
    left_indent: Horizontal axes-position adjustment in figure units.
    add_width: Axes-width adjustment in figure units.
    multib_unicolor: Use one color for every block instead of cycling.
    no_show: Return without opening a plot window.
    saveas: If given, save the figure to this path.

  Returns:
    The populated Figure.
  """
  path = path.rstrip("/") + "/"
  if toroidal_sampling not in ("nearest", "all"):
    raise ValueError(f"Unknown toroidal_sampling: '{toroidal_sampling}'.")
  file_prefix = f"{path}{name}-" if multib == "-10" else f"{path}{name}_b*-"

  if nodes_file:
    resolved_nodes_file = nodes_file if nodes_file[
        0] == "/" else path + nodes_file
  else:
    resolved_nodes_file = file_prefix + "geo_corn_nodes.gkyl"

  blocks = utils.get_block_indices(multib, resolved_nodes_file)

  major_r_ex = [1e9, -1e9]
  vert_z_ex = [1e9, -1e9]
  block_nodes = {}
  for block_idx in blocks:
    grid, nodes, gdat = utils.read_gfile(
        resolved_nodes_file.replace("*", str(block_idx)))
    mapc2p = is_geo_mapc2p(gdat.ctx)
    major_r, vert_z = nodes_to_RZ(nodes, mapc2p)
    spatial_dim = nodes.ndim - 1
    prefix = file_prefix.replace("*", str(block_idx)).rstrip("-")
    interior = None
    if int_nodes:
      expected = tuple(2 * (n - 1) for n in nodes.shape[:-1])
      interior = _read_node_coordinates(f"{prefix}-geo_int_nodes.gkyl",
                                        expected)
      if spatial_dim == 3:
        interior = tuple(v[:, 0, :] for v in interior)
    surfaces = {}
    if surf_nodes:
      if spatial_dim == 1:
        raise ValueError("Surface nodes require 2D or 3D geometry.")
      directions = (0, 1) if spatial_dim == 2 else (0, 2)
      for direction in directions:
        surfaces[direction] = _read_surface(prefix, direction, nodes.shape[:-1])
    block_nodes[block_idx] = (major_r, vert_z, _surface_rz(nodes, mapc2p),
                              surfaces, interior)
    major_r_ex = [
        min(major_r_ex[0], np.amin(major_r)),
        max(major_r_ex[1], np.amax(major_r))
    ]
    vert_z_ex = [
        min(vert_z_ex[0], np.amin(vert_z)),
        max(vert_z_ex[1], np.amax(vert_z))
    ]

  length_r = major_r_ex[1] - major_r_ex[0]
  length_z = vert_z_ex[1] - vert_z_ex[0]
  aspect_ratio = length_r / length_z

  ax_pos = [0.14 + left_indent, 0.10, 0.72 + add_width, 0.84]
  cax_pos = [ax_pos[0] + ax_pos[2] + 0.02, ax_pos[1], 0.025, ax_pos[3]]
  fig_width = min(9.0, max(5.0, 8.0 * aspect_ratio + 2.0))
  fig_height = min(8.0, max(3.5, 0.75 * fig_width / aspect_ratio + 1.0))
  fig = plt.figure(figsize=(fig_width, fig_height))
  ax = fig.add_axes(ax_pos)

  color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
  block_colors = cycle([color_list[0]] if multib_unicolor else color_list)

  for block_idx in blocks:
    major_r, vert_z, full_corners, surfaces, interior = block_nodes[block_idx]
    ax.plot(major_r, vert_z, marker=".", color="k", linestyle="none")

    cell_color = next(block_colors)
    if surf_nodes:
      _draw_curves(ax, full_corners, surfaces, cell_color, toroidal_sampling)
      for direction in ((0, 1) if full_corners[0].ndim == 2 else (0, 2)):
        surf_r, surf_z = surfaces[direction]
        if full_corners[0].ndim == 3 and toroidal_sampling == "nearest":
          surf_r, surf_z = surf_r[:, 0, :], surf_z[:, 0, :]
        ax.plot(surf_r.ravel(),
                surf_z.ravel(),
                marker=".",
                color=cell_color,
                linestyle="none")
    elif major_r.ndim <= 1:
      ax.plot(major_r, vert_z, color=cell_color, linestyle="-")
    else:
      segs_constx = np.stack((major_r, vert_z), axis=2)
      segs_consty = segs_constx.transpose(1, 0, 2)
      ax.add_collection(LineCollection(segs_constx, color=cell_color))
      ax.add_collection(LineCollection(segs_consty, color=cell_color))

    if interior is not None:
      ax.plot(interior[0].ravel(),
              interior[1].ravel(),
              marker="x",
              markersize=3,
              color=cell_color,
              linestyle="none")

  colorbar = True
  if psi_file:
    resolved_psi = psi_file if psi_file[0] == "/" else path + psi_file
    psi_grid, psi_values, _ = utils.read_interpolated_gfile(resolved_psi,
                                                            poly_order=2,
                                                            basis_type="tensor")
    psi_grid_cc = [
        0.5 * (psi_grid[d][:-1] + psi_grid[d][1:]) for d in range(len(psi_grid))
    ]

    levels = _parse_levels(clevels, cnlevels)
    if isinstance(levels, np.ndarray) and levels.size == 1:
      colorbar = False

    if contour:
      im = ax.contour(psi_grid_cc[0], psi_grid_cc[1], psi_values.transpose(),
                      levels)
    else:
      im = ax.pcolormesh(psi_grid[0],
                         psi_grid[1],
                         psi_values.transpose(),
                         cmap="inferno")

    if colorbar:
      cbar_ax = fig.add_axes(cax_pos)
      cbar = fig.colorbar(im, ax=ax, cax=cbar_ax)
      cbar.ax.tick_params(labelsize=_TICK_FONT_SIZE)
      cbar.set_label(zlabel,
                     rotation=90,
                     labelpad=0,
                     fontsize=_COLORBAR_LABEL_FONT_SIZE)

  if wall_file:
    resolved_wall = wall_file if wall_file[0] == "/" else path + wall_file
    wall_data = np.loadtxt(resolved_wall, delimiter=",")
    ax.plot(wall_data[:, 0], wall_data[:, 1], color="grey")

  ax.set_xlabel(xlabel, fontsize=_XY_LABEL_FONT_SIZE)
  ax.set_ylabel(ylabel, fontsize=_XY_LABEL_FONT_SIZE)
  ax.set_title(title, fontsize=_TITLE_FONT_SIZE)
  if xlim:
    ax.set_xlim(xlim[0], xlim[1])
  if ylim:
    ax.set_ylim(ylim[0], ylim[1])
  ax.set_aspect("equal", adjustable="box")
  utils.set_tick_font_size(ax, _TICK_FONT_SIZE)

  if saveas:
    fig.savefig(saveas)
  if not no_show:
    plt.show()

  return fig
