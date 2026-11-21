"""The anomalous diffusivity a gyrokinetic tokamak run used.

The run writes ``<species>_anom_diff_0``: the diffusivity times ``g^xx J``
as the solver uses it (the library takes the diffusivity at the cells'
corner nodes and weak-multiplies it by ``geo_int_gxxj``). Weak division by
``geo_int_gxxj`` gives back the diffusivity's expansion in each cell;
evaluated at the cells' corners it gives the node values again, and cells
sharing a node agree up to roundoff where the diffusivity was continuous.

Where a run wrote no ``anom_diff`` file, its diffusivity was uniform: give
it as ``diffusivity`` (the value is in the run's own settings, not in its
output files).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from postgkyl import dg
from postgkyl.gdata import GData

from .flux_coords import CellEdges, layout_cell_edges, midplane_row_radii
from .multib_layout import MultibLayout, midplane_node_values, multib_layout

# The reference cell's corners, (x, y) in [-1, 1]^2, and the node each is of
# cell (i, j): node (i + di, j + dj).
_CORNERS = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
_NODE_OFFSETS = ((0, 0), (1, 0), (0, 1), (1, 1))


@dataclass(frozen=True, eq=False)
class AnomalousDiffusivity:
  """A species' anomalous diffusivity (m^2/s), per block, rows inside out.

  Attributes:
    species: The species.
    uniform: The diffusivity is the given uniform value (no file was read).
    nodes: Each block's values at its corner nodes ``(nr + 1, nt + 1)``:
      the mean of the cells sharing each node.
    cells: Each block's cell means ``(nr, nt)``.
    gap: The largest spread between the cells sharing a node, relative to
      the largest ``|D|`` of its block (0 for a continuous diffusivity).
  """

  species: str
  uniform: bool
  nodes: tuple[np.ndarray, ...]
  cells: tuple[np.ndarray, ...]
  gap: float

  def __str__(self) -> str:
    kind = "uniform (given)" if self.uniform else "from anom_diff_0 / gxxj"
    lines = [
        f"{self.species} anomalous diffusivity, {kind}; largest "
        f"relative gap between cells at a node {self.gap:.2e}"
    ]
    for b, nd in enumerate(self.nodes):
      lines.append(f"  block {b}: D from {np.nanmin(nd):.4g} to "
                   f"{np.nanmax(nd):.4g} m^2/s")
    return "\n".join(lines)


@dataclass(frozen=True, eq=False)
class MidplaneDiffusivity:
  """Node values at Z = 0 along one side's midplane, inside out.

  Attributes:
    side: ``'outboard'`` or ``'inboard'``.
    R: Each node row's major radius at Z = 0 (m); a node row two blocks
      share appears once, with the inner block.
    values: The values there.
    blocks: The block of each node row.
  """

  side: str
  R: np.ndarray
  values: np.ndarray
  blocks: np.ndarray

  def __str__(self) -> str:
    lines = [f"{self.side} midplane:", f"{'R (m)':>10} {'value':>13} block"]
    for R, v, b in zip(self.R, self.values, self.blocks):
      lines.append(f"{R:10.5f} {v:13.6g} {int(b):5d}")
    return "\n".join(lines)


def node_values(coeffs: np.ndarray, basis_type: str,
                poly_order: int) -> tuple[np.ndarray, float]:
  """A 2D DG field's values at its cells' corner nodes.

  Args:
    coeffs: The modal coefficients ``(nx, ny, num_basis)`` of one field.
    basis_type: Its basis (``'serendipity'``, ``'tensor'``, ...).
    poly_order: Its polynomial order.

  Returns:
    ``(values, gap)``: the mean of the cells sharing each node
    ``(nx + 1, ny + 1)``, and the largest spread between them (0 for a
    continuous field).
  """
  corner = coeffs @ dg.basis_matrix(basis_type, 2, poly_order,
                                    np.array(_CORNERS)).T
  nx, ny = corner.shape[:2]
  total = np.zeros((nx + 1, ny + 1))
  count = np.zeros((nx + 1, ny + 1))
  hi = np.full((nx + 1, ny + 1), -np.inf)
  lo = np.full((nx + 1, ny + 1), np.inf)
  for c, (di, dj) in enumerate(_NODE_OFFSETS):
    at = (slice(di, di + nx), slice(dj, dj + ny))
    total[at] += corner[..., c]
    count[at] += 1
    hi[at] = np.maximum(hi[at], corner[..., c])
    lo[at] = np.minimum(lo[at], corner[..., c])
  return total / count, float((hi - lo).max())


def _diffusivity(layout: MultibLayout, species: str,
                 diffusivity: float | None) -> AnomalousDiffusivity:
  """:func:`anomalous_diffusivity` of a loaded layout."""
  nodes, cells, gap = [], [], 0.0
  for b, nd in enumerate(layout.nodes):
    if diffusivity is not None:
      nodes.append(np.full(nd.shape[:2], float(diffusivity)))
      cells.append(
          np.full((nd.shape[0] - 1, nd.shape[1] - 1), float(diffusivity)))
      continue
    D = (GData(layout.file_name(b, f"{species}_anom_diff", 0)) /
         GData(layout.file_name(b, "geo_int_gxxj")))
    basis_type, poly_order = D.ctx["basis_type"], D.ctx["poly_order"]
    coeffs = np.asarray(D.get_values())
    values, spread = node_values(coeffs, basis_type, poly_order)
    means = dg.modal.cell_means(coeffs, 2,
                                dg.num_basis(2, poly_order, basis_type))
    nodes.append(layout.rows(b, values))
    cells.append(layout.rows(b, means[..., 0]))
    gap = max(gap, spread / max(np.abs(values).max(), 1e-300))
  return AnomalousDiffusivity(species, diffusivity is not None, tuple(nodes),
                              tuple(cells), gap)


def anomalous_diffusivity(
    name: str,
    species: str,
    *,
    path: str = "./",
    geqdsk: str | None = None,
    diffusivity: float | None = None) -> AnomalousDiffusivity:
  """The anomalous diffusivity a run used for a species, per block.

  Reads each block's ``<species>_anom_diff_0`` and ``geo_int_gxxj`` and
  weak-divides the first by the second; or, with ``diffusivity``, reads no
  file and gives that uniform value everywhere.

  Args:
    name: The run's file prefix (without ``_b<k>``).
    species: The species.
    path: Directory holding the run's output.
    geqdsk: The run's equilibrium file, for the layout (see
      :func:`multib_layout`).
    diffusivity: The run's uniform diffusivity (m^2/s), for a run that
      wrote no ``anom_diff`` file.

  Returns:
    The :class:`AnomalousDiffusivity`.
  """
  return _diffusivity(multib_layout(name, path=path, geqdsk=geqdsk), species,
                      diffusivity)


def midplane_node_profile(layout: MultibLayout, side: str,
                          nodes: tuple[np.ndarray, ...],
                          edges: CellEdges) -> MidplaneDiffusivity:
  """Node values at Z = 0 along a midplane stack, inside out.

  Along each node row of the stack's blocks, linear in theta between the
  two nodes bracketing Z = 0, at the R :func:`~.flux_coords.midplane_row_radii`
  gives the row (on its curve where the run has one); a node row two blocks
  share (their separatrix) appears once, with the inner block.

  Args:
    layout: The run's layout.
    side: The stack's side (a key of ``layout.stacks``).
    nodes: Each block's node values ``(nr + 1, nt + 1)``, rows inside out.
    edges: The run's :func:`~.flux_coords.layout_cell_edges`.

  Returns:
    The :class:`MidplaneDiffusivity`.
  """
  stack = layout.stacks[side]
  Rs, vs, bs = [], [], []
  for b in stack.blocks:
    nd = layout.nodes[b]
    R = midplane_row_radii(layout, edges, b, stack.crossing)
    v = midplane_node_values(nd, nodes[b], stack.crossing)
    if Rs and abs(R[0] - Rs[-1][-1]) <= 1e-12 * max(abs(R[0]), 1.0):
      R, v = R[1:], v[1:]
    Rs.append(R)
    vs.append(v)
    bs.append(np.full(R.size, b))
  return MidplaneDiffusivity(side, np.concatenate(Rs), np.concatenate(vs),
                             np.concatenate(bs))


def midplane_diffusivity(
    name: str,
    species: str,
    side: Literal["outboard", "inboard"],
    *,
    path: str = "./",
    geqdsk: str | None = None,
    diffusivity: float | None = None) -> MidplaneDiffusivity:
  """The anomalous diffusivity at Z = 0 along one side's midplane.

  The node values of :func:`anomalous_diffusivity` along the midplane stack
  (the core block, then the SOL block over it), interpolated to Z = 0 as
  :func:`midplane_node_profile` does.

  Args:
    name: The run's file prefix (without ``_b<k>``).
    species: The species.
    side: ``'outboard'`` or ``'inboard'``.
    path: Directory holding the run's output.
    geqdsk: The run's equilibrium file, for the layout (see
      :func:`multib_layout`).
    diffusivity: The run's uniform diffusivity (m^2/s), for a run that
      wrote no ``anom_diff`` file.

  Returns:
    The :class:`MidplaneDiffusivity` of the diffusivity (m^2/s).

  Raises:
    KeyError: if the run has no midplane on that side.
  """
  layout = multib_layout(name, path=path, geqdsk=geqdsk)
  if side not in layout.stacks:
    raise KeyError(f"No {side} midplane in {name}: it has "
                   f"{', '.join(layout.stacks) or 'none'}")
  return midplane_node_profile(layout, side,
                               _diffusivity(layout, species, diffusivity).nodes,
                               layout_cell_edges(layout))
