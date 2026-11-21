"""Flux coordinates of a gyrokinetic tokamak run's cells: psi_N and L_theta.

- Cell edges: a Gkeyll cell edge is a curve through its two corners and its
  two Gauss-Legendre surface nodes (``geo_surf0_nodes`` on the constant-psi
  edges, ``geo_surf1_nodes`` on the constant-theta ones). It is drawn and
  measured as the cubic through those four points (:func:`edge_polylines`).
  Where a block lacks the surface nodes, or one of them lies away from its
  edge (a from-file geometry writes zeros there), every block's edges are
  their chords instead.
- psi_N = (psi - simag) / (sibry - simag), the geqdsk's (EFIT's)
  normalization. psi is the position map's first coordinate
  (``geo_corn_mc2nu_pos``: the computational psi's non-uniform value),
  evaluated at the cells' corners with Gkeyll's basis.
- L_theta is the poloidal length (in R, Z) along each cell row's centre line
  (the mean of its two node rows' edges) of an open theta chain, measured from
  the chain's target end: an outer target before an inner one, a lower before
  an upper.

The public callables take a run's ``name``, ``path`` and ``geqdsk`` and build
its :func:`~postgkyl.diagnostics.gk.multib_layout.multib_layout`; the pure
cores take arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from postgkyl import dg, io
from postgkyl.gdata import GData

from .multib_layout import (MultibLayout, block_prefixes, midplane_node_values,
                            midplane_radii, multib_layout)
from .nodes import read_surface_nodes

# Where an edge's two Gauss-Legendre surface nodes lie along it, t in [0, 1]
# (its corners at 0 and 1): the nodes +-1/sqrt(3) of [-1, 1].
GAUSS_T = (0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3))
# Where an edge curve is sampled between its corners: the Gauss nodes and five
# evenly spaced points.
EDGE_T = np.sort(np.r_[np.linspace(0, 1, 7)[1:-1], GAUSS_T])
# A 2D cell's corners as the (psi, theta) offsets of their nodes from the
# cell's first node; corner (di, dj) lies at (2 di - 1, 2 dj - 1) of the
# reference cell [-1, 1]^2.
_CORNERS = ((0, 0), (1, 0), (0, 1), (1, 1))


# ------------------------------------------------------------ edge curves
def lagrange_weights(t: np.ndarray, knots: np.ndarray) -> np.ndarray:
  """``(len(t), len(knots))``: the Lagrange polynomials of ``knots`` at
  each ``t``; row ``s`` times the values at the knots is the interpolating
  polynomial's value at ``t[s]``."""
  knots = np.asarray(knots, dtype=float)
  n = len(knots)
  return np.array([[
      np.prod([(s - knots[m]) / (knots[k] - knots[m]) for m in range(n)
               if m != k]) for k in range(n)
  ] for s in np.atleast_1d(t)])


def edge_polylines(corners: np.ndarray, surf0: np.ndarray,
                   surf1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """A block's cell edges as polylines along their curves.

  Each edge is the cubic through its two corners and its two Gauss nodes
  (at ``GAUSS_T``), sampled at ``EDGE_T`` between the corners.

  Args:
    corners: The corner nodes ``(nr + 1, nt + 1, 2)`` in (R, Z).
    surf0: The two Gauss nodes of each constant-psi edge ``(nr + 1, 2 nt, 2)``.
    surf1: The two Gauss nodes of each constant-theta edge
      ``(2 nr, nt + 1, 2)``.

  Returns:
    ``(rows, cols)``: the node rows ``(nr + 1, k nt + 1, 2)`` and the node
    columns ``(nt + 1, k nr + 1, 2)``, ``k = len(EDGE_T) + 1`` points to an
    edge.
  """
  weights = lagrange_weights(EDGE_T, np.r_[0.0, GAUSS_T, 1.0])

  def along(a, g1, g2, b):  # edges (..., n, 2) -> polylines (..., k n + 1, 2)
    inner = np.einsum("sk,k...->...s", weights, np.stack([a, g1, g2, b]))
    pts = np.moveaxis(np.concatenate([a[..., None], inner], axis=-1), -1, -2)
    flat = pts.reshape(*pts.shape[:-3], -1, 2)
    return np.concatenate([flat, b[..., -1:, :]], axis=-2)

  rows = along(corners[:, :-1], surf0[:, 0::2], surf0[:, 1::2], corners[:, 1:])
  cols = along(*(a.transpose(1, 0, 2) for a in (corners[:-1], surf1[0::2],
                                                surf1[1::2], corners[1:])))
  return rows, cols


def nodes_by_edges(corners: np.ndarray, surf0: np.ndarray,
                   surf1: np.ndarray) -> bool:
  """Whether every Gauss node lies by its edge: within one chord length of
  the chord's middle.

  A curved edge bends away from its chord, by up to 0.88 of it at STEP's X
  point; a node misread or never written lies anywhere (a from-file
  geometry's files hold zeros).
  """
  tol = 1e-9 * np.ptp(corners)
  for a, b, gauss in ((corners[:, :-1], corners[:, 1:], (surf0[:, 0::2],
                                                         surf0[:, 1::2])),
                      (corners[:-1], corners[1:], (surf1[0::2], surf1[1::2]))):
    chord = np.hypot(*np.moveaxis(b - a, -1, 0))
    middle = 0.5 * (a + b)
    if any((np.hypot(*np.moveaxis(g - middle, -1, 0)) > chord + tol).any()
           for g in gauss):
      return False
  return True


def cell_polygons(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
  """Each cell as the region its edges bound: ``(nr, nt, 4 k, 2)``.

  Args:
    rows: A block's node rows ``(nr + 1, k nt + 1, 2)``
      (:attr:`CellEdges.rows`).
    cols: Its node columns ``(nt + 1, k nr + 1, 2)``.

  Returns:
    Each cell's boundary from its first corner: along its lower node row,
    up its upper node column, back along its upper node row and down its
    lower node column, ``k`` points to an edge.
  """
  nr, nt = rows.shape[0] - 1, cols.shape[0] - 1
  k = (rows.shape[1] - 1) // nt
  a = np.arange(k)
  j, i = np.arange(nt)[:, None] * k, np.arange(nr)[:, None] * k
  bottom, top = rows[:-1][:, j + a], rows[1:][:, j + k - a]
  right, left = cols[1:][:, i + a], cols[:-1][:, i + k - a]
  return np.concatenate(
      [bottom,
       right.transpose(1, 0, 2, 3), top,
       left.transpose(1, 0, 2, 3)],
      axis=2)


@dataclass(frozen=True, eq=False)
class CellEdges:
  """Every block's cell edges as polylines, rows inside out.

  Attributes:
    name: The run's file prefix.
    path: Its directory.
    rows: Each block's node rows ``(nr + 1, k nt + 1, 2)`` in (R, Z).
    cols: Each block's node columns ``(nt + 1, k nr + 1, 2)``.
    curved: The edges are curves through the surface nodes; else chords
      (``k = 1``).
    note: Why the edges are chords; empty when curved.
  """

  name: str
  path: str
  rows: tuple[np.ndarray, ...]
  cols: tuple[np.ndarray, ...]
  curved: bool
  note: str = ""

  @property
  def points_per_edge(self) -> int:
    """``k``: the polyline points to an edge (``len(EDGE_T) + 1`` or 1)."""
    return (self.rows[0].shape[1] - 1) // (self.cols[0].shape[0] - 1)

  def __str__(self) -> str:
    kind = (f"curved, {self.points_per_edge} points to an edge"
            if self.curved else f"chords ({self.note})")
    lines = [f"{self.name}: cell edges {kind}"]
    for b, (r, c) in enumerate(zip(self.rows, self.cols)):
      pts = np.concatenate([r.reshape(-1, 2), c.reshape(-1, 2)])
      lo, hi = pts.min(axis=0), pts.max(axis=0)
      lines.append(
          f"  block {b}: {r.shape[0] - 1} x {c.shape[0] - 1} cells, "
          f"R {lo[0]:.4g}..{hi[0]:.4g} m, Z {lo[1]:.4g}..{hi[1]:.4g} m")
    return "\n".join(lines)


def layout_cell_edges(layout: MultibLayout) -> CellEdges:
  """:func:`cell_edges` of a layout already built (not a CLI command: it
  takes a :class:`MultibLayout`).

  The polylines are built in the files' orientation, then their rows
  reversed where the layout reads a block reversed.
  """
  rows, cols, note = [], [], ""
  for b, prefix in enumerate(block_prefixes(layout.name, layout.path)):
    corners = layout.rows(b, layout.nodes[b])  # the files' orientation
    try:
      surf0, surf1 = (np.stack(read_surface_nodes(prefix, d,
                                                  corners.shape[:-1]),
                               axis=-1) for d in (0, 1))
    except ValueError as err:
      note = f"no surface nodes ({err})"
      break
    if not nodes_by_edges(corners, surf0, surf1):
      note = f"block {b}'s surface nodes lie off their edges"
      break
    r, c = edge_polylines(corners, surf0, surf1)
    rows.append(r[::-1] if layout.reversed[b] else r)
    cols.append(c[:, ::-1] if layout.reversed[b] else c)
  if note:
    rows = list(layout.nodes)
    cols = [nd.transpose(1, 0, 2) for nd in layout.nodes]
  return CellEdges(layout.name, layout.path, tuple(rows), tuple(cols), not note,
                   note)


def midplane_row_radii(layout: MultibLayout, edges: CellEdges, block: int,
                       crossing: float | None) -> np.ndarray:
  """R where each node row of a block crosses Z = 0: along the row's curve
  (its cell edges through the surface nodes) where the run has them, else
  along the chords between its nodes (:func:`~.multib_layout.midplane_radii`).

  On a coarse grid the chords can misplace the crossing by centimetres, and
  the radii then need not grow outward (ASDEX's 8 theta cells across the main
  SOL).

  Args:
    layout: The run's layout.
    edges: Its :func:`layout_cell_edges`.
    block: The block.
    crossing: Theta node position of the crossing to use where a row crosses
      Z = 0 more than once, as for :func:`~.multib_layout.midplane_radii`.
  """
  if not edges.curved:
    return midplane_radii(layout.nodes[block], crossing)
  rows, k = edges.rows[block], edges.points_per_edge
  return midplane_node_values(rows, rows[..., 0],
                              None if crossing is None else crossing * k)


def cell_edges(name: str,
               *,
               path: str = "./",
               geqdsk: str | None = None) -> CellEdges:
  """Every block's cell edges as curves through the Gauss surface nodes (or
  chords where a block lacks them), rows inside out.

  Args:
    name: The run's file prefix (without ``_b<k>``).
    path: Directory holding the run's output.
    geqdsk: The run's equilibrium file; its magnetic axis orients the rows
      where the topology does not (:func:`multib_layout`).

  Returns:
    The :class:`CellEdges`.
  """
  return layout_cell_edges(multib_layout(name, path=path, geqdsk=geqdsk))


# ------------------------------------------------------------------ psi_N
def corner_values(coeffs: np.ndarray, basis_type: str,
                  poly_order: int) -> tuple[np.ndarray, float]:
  """A 2D modal field at its cells' corners, as node values.

  Args:
    coeffs: One field's modal coefficients ``(nr, nt, num_basis)``.
    basis_type: Its basis (the file's metadata).
    poly_order: Its polynomial order.

  Returns:
    ``(values, gap)``: the values ``(nr + 1, nt + 1)`` at the nodes, node
    ``(i, j)`` taking cell ``(max(i - 1, 0), max(j - 1, 0))``'s; and the
    largest disagreement between the cells sharing a node (0 for a
    continuous field).
  """
  nr, nt = coeffs.shape[:2]
  points = np.array([(2 * di - 1, 2 * dj - 1) for di, dj in _CORNERS],
                    dtype=float)
  at = coeffs @ dg.basis_matrix(basis_type, 2, poly_order, points).T
  out = np.empty((nr + 1, nt + 1))
  lo, hi = np.full_like(out, np.inf), np.full_like(out, -np.inf)
  for c, (di, dj) in enumerate(_CORNERS):  # the last write wins
    node = (slice(di, di + nr), slice(dj, dj + nt))
    out[node] = at[..., c]
    lo[node] = np.minimum(lo[node], at[..., c])
    hi[node] = np.maximum(hi[node], at[..., c])
  return out, float((hi - lo).max())


def flux_rows(psi: np.ndarray, simag: float,
              sibry: float) -> tuple[np.ndarray, np.ndarray]:
  """psi_N of a block's node rows and cell rows.

  Args:
    psi: psi at the block's nodes ``(nr + 1, nt + 1)``.
    simag: psi at the magnetic axis.
    sibry: psi at the boundary (the separatrix).

  Returns:
    ``(nodes, cells)``: each node row's mean over theta of
    ``(psi - simag) / (sibry - simag)`` ``(nr + 1,)``, and each cell row's
    mean of its two node rows ``(nr,)``.
  """
  n = (psi.mean(axis=1) - simag) / (sibry - simag)
  return n, 0.5 * (n[1:] + n[:-1])


def _modal(file_name: str) -> tuple[np.ndarray, str, int]:
  """A 2D modal file's coefficients ``(nr, nt, ncomp)``, basis and order."""
  data = GData(file_name)
  ctx = data.ctx
  return np.asarray(data.get_values()), ctx["basis_type"], ctx["poly_order"]


def _block_psi(layout: MultibLayout, block: int) -> tuple[np.ndarray, float]:
  """psi at a block's nodes ``(nr + 1, nt + 1)``, rows inside out, and its
  continuity gap (hidden from the CLI: takes a layout)."""
  coeffs, basis_type, poly_order = _modal(
      layout.file_name(block, "geo_corn_mc2nu_pos"))
  nb = dg.num_basis(2, poly_order, basis_type)
  psi, gap = corner_values(coeffs[..., :nb], basis_type, poly_order)
  return layout.rows(block, psi), gap


@dataclass(frozen=True, eq=False)
class FluxSurfaces:
  """psi_N of every block's node rows and cell rows, rows inside out.

  Attributes:
    name: The run's file prefix.
    path: Its directory.
    geqdsk: The equilibrium file whose axis and boundary flux normalize psi.
    nodes: Each block's node rows' psi_N ``(nr + 1,)``.
    cells: Each block's cell rows' psi_N ``(nr,)``.
    separatrix: psi_N of each midplane SOL block's separatrix node row (1
      when the run's separatrix is its geqdsk's boundary).
    gap: The largest psi_N disagreement between cells sharing a node (0 for
      a continuous position map).
  """

  name: str
  path: str
  geqdsk: str
  nodes: tuple[np.ndarray, ...]
  cells: tuple[np.ndarray, ...]
  separatrix: dict[int, float]
  gap: float

  def __str__(self) -> str:
    lines = [
        f"{self.name}: psi_N (geqdsk {self.geqdsk}), continuity gap "
        f"{self.gap:.3g}"
    ]
    for b, n in enumerate(self.nodes):
      lines.append(f"  block {b}: node rows {n[0]:.6g}..{n[-1]:.6g}")
    for b, v in self.separatrix.items():
      lines.append(f"  separatrix of block {b}: {v:.6g}")
    return "\n".join(lines)


def layout_flux_surfaces(layout: MultibLayout, geqdsk: str) -> FluxSurfaces:
  """:func:`flux_surfaces` of a layout already built (not a CLI command:
  it takes a :class:`MultibLayout`)."""
  head = io.read_geqdsk_header(geqdsk)
  nodes, cells, gap = [], [], 0.0
  for b in range(layout.num_blocks):
    psi, g = _block_psi(layout, b)
    n, c = flux_rows(psi, head.simag, head.sibry)
    nodes.append(n)
    cells.append(c)
    gap = max(gap, g / abs(head.sibry - head.simag))
  sep = {
      st.sol: float(nodes[st.sol][st.sep_row])
      for st in layout.stacks.values()
  }
  return FluxSurfaces(layout.name, layout.path, geqdsk, tuple(nodes),
                      tuple(cells), sep, gap)


def flux_surfaces(name: str, *, path: str = "./", geqdsk: str) -> FluxSurfaces:
  """psi_N of every block's node rows and cell rows.

  psi_N = (psi - simag) / (sibry - simag) with the geqdsk's axis and
  boundary flux, so measurements mapped by the same geqdsk compare.

  Args:
    name: The run's file prefix (without ``_b<k>``).
    path: Directory holding the run's output.
    geqdsk: The run's equilibrium file.

  Returns:
    The :class:`FluxSurfaces`.
  """
  return layout_flux_surfaces(multib_layout(name, path=path, geqdsk=geqdsk),
                              geqdsk)


# ---------------------------------------------------------------- L_theta
def centre_line(
    block_rows: Sequence[np.ndarray]) -> tuple[np.ndarray, tuple[int, ...]]:
  """The cell-centre lines of blocks joined end to end.

  Args:
    block_rows: Each block's node rows ``(nr + 1, n_b, 2)`` in chain order,
      each continuing the previous one's last point.

  Returns:
    ``(line, starts)``: each cell row's centre line, the mean of its two node
    rows, ``(nr, sum(n_b - 1) + 1, 2)``; and each block's first index on it.
  """
  lines, starts, i0 = [], [], 0
  for rows in block_rows:
    mid = 0.5 * (rows[1:] + rows[:-1])
    lines.append(mid if not lines else mid[:, 1:])
    starts.append(i0)
    i0 += mid.shape[1] - 1
  return np.concatenate(lines, axis=1), tuple(starts)


def arc_lengths(lines: np.ndarray) -> np.ndarray:
  """Arc length along polylines ``(n_lines, n_points, 2)`` from their first
  point: ``(n_lines, n_points)``."""
  d = np.hypot(*np.diff(lines, axis=1).transpose(2, 0, 1))
  return np.concatenate([np.zeros(
      (len(lines), 1)), np.cumsum(d, axis=1)],
                        axis=1)


def at_cell_centres(lengths: np.ndarray, start: int, num_cells: int,
                    points_per_edge: int) -> np.ndarray:
  """``lengths`` ``(nr, n)`` at a block's cell centres ``(nr, num_cells)``:
  its cells span ``points_per_edge`` points of the line each from ``start``,
  their centres half way, linear between points."""
  idx = start + points_per_edge * (np.arange(num_cells) + 0.5)
  return np.array(
      [np.interp(idx, np.arange(lengths.shape[1]), row) for row in lengths])


@dataclass(frozen=True, eq=False)
class ChainLengths:
  """L_theta along one open theta chain.

  Attributes:
    blocks: The chain's blocks from its start end.
    kind: Its first block's region: ``'sol'``, ``'pf'``, or ``'core'`` (a
      half domain's core, mirror to mirror).
    start: Where L_theta = 0: a target's name, or ``'<side> midplane'`` (a
      half domain's mirror).
    end: The chain's other end, named the same way.
    cells: Each block's L_theta at its cell centres ``(nr, nt)``, rows inside
      out, theta in the files' order.
    total: Each cell row's length from end to end ``(nr,)``.
    marks: ``(label, L_theta per cell row)`` in order along the chain: the
      X-point (where a SOL chain's block over a PF block meets one over the
      core, or a PF chain's blocks meet), each Z = 0 crossing (``'outboard
      midplane'`` or ``'inboard midplane'``) and, for a SOL chain, its highest
      point (``'top'``) when that lies inside the chain.
  """

  blocks: tuple[int, ...]
  kind: str
  start: str
  end: str
  cells: dict[int, np.ndarray]
  total: np.ndarray
  marks: tuple[tuple[str, np.ndarray], ...]


@dataclass(frozen=True, eq=False)
class PoloidalLengths:
  """L_theta of every cell of a run's open theta chains.

  Attributes:
    name: The run's file prefix.
    path: Its directory.
    chains: The open chains whose blocks' rows align (core rings and chains
      of unequal row counts are left out).
    curved: L_theta is measured along curved cell edges; else chords.
    note: Why the edges are chords; empty when curved.
  """

  name: str
  path: str
  chains: tuple[ChainLengths, ...]
  curved: bool
  note: str = ""

  def __str__(self) -> str:
    lines = [
        f"{self.name}: L_theta along " +
        ("curved cell edges" if self.curved else f"chords ({self.note})")
    ]
    for ch in self.chains:
      lines.append(f"  {ch.kind} chain {list(ch.blocks)}: {ch.start} -> "
                   f"{ch.end}, {ch.total.min():.4g}..{ch.total.max():.4g} m")
      lines += [
          f"    {label}: {np.mean(v):.4g} m (mean over rows)"
          for label, v in ch.marks
      ]
    return "\n".join(lines)


def _chain(layout: MultibLayout, edges: CellEdges,
           blocks: tuple[int, ...]) -> ChainLengths:
  """L_theta along one open chain (hidden from the CLI: takes a layout)."""
  ends = {(t.leg, t.theta_end): t for t in layout.targets}
  ends.update({
      (t.pf, t.theta_end): t
      for t in layout.targets if t.pf is not None
  })
  first, last = ends.get((blocks[0], 0)), ends.get((blocks[-1], -1))

  def rank(t):
    return (0 if t.side == "outer" else 1, 0 if t.updown == "lower" else 1)

  flip = last is not None and (first is None or rank(last) < rank(first))
  order = blocks[::-1] if flip else blocks
  k = edges.points_per_edge
  line, starts = centre_line(
      [edges.rows[b][:, ::-1] if flip else edges.rows[b] for b in order])
  lengths = arc_lengths(line)
  cells = {}
  for b, i0 in zip(order, starts):
    lc = at_cell_centres(lengths, i0, layout.nodes[b].shape[1] - 1, k)
    cells[b] = lc[:, ::-1] if flip else lc

  region, under = layout.region, layout.under

  def below(b):  # the region of the block under b
    return region[under[b]] if under[b] is not None else None

  side = layout.side

  kind = region[blocks[0]]
  marks = [("X-point", lengths[:, j1])
           for b0, b1, j1 in zip(order, order[1:], starts[1:])
           if kind == "pf" or {below(b0), below(b1)} == {"pf", "core"}]
  z = line[..., 1]
  cross = [np.where(np.diff(np.sign(zr)) != 0)[0] for zr in z]
  for c in range(min(len(x) for x in cross)):
    vals, rs = [], []
    for zr, lr, rr, x in zip(z, lengths, line[..., 0], cross):
      j = x[c]
      w = zr[j] / (zr[j] - zr[j + 1])
      vals.append(lr[j] + w * (lr[j + 1] - lr[j]))
      rs.append(rr[j])
    marks.append((f"{side(np.mean(rs))} midplane", np.array(vals)))
  top = np.argmax(z, axis=1)
  if kind == "sol" and np.all((top > 0) & (top < z.shape[1] - 1)):
    marks.append(("top", lengths[np.arange(len(lengths)), top]))
  if flip:
    first, last = last, first

  def name(t, end):  # a target, else the mirror: a half domain's midplane
    if t is not None:
      return t.name
    return f"{side(line[:, end, 0].mean())} midplane"

  return ChainLengths(tuple(order), kind, name(first, 0), name(last, -1), cells,
                      lengths[:, -1],
                      tuple(sorted(marks, key=lambda m: float(np.mean(m[1])))))


def _lengths(layout: MultibLayout, edges: CellEdges) -> PoloidalLengths:
  """L_theta of a run's open chains (hidden from the CLI: takes a layout)."""
  chains = tuple(
      _chain(layout, edges, c.blocks) for c in layout.chains
      if not c.ring and len({layout.nodes[b].shape[0]
                             for b in c.blocks}) == 1)
  return PoloidalLengths(layout.name, layout.path, chains, edges.curved,
                         edges.note)


def poloidal_lengths(name: str,
                     *,
                     path: str = "./",
                     geqdsk: str | None = None) -> PoloidalLengths:
  """L_theta, the poloidal length along each cell row of the open theta
  chains (SOL legs and main SOL, private flux) from the chain's target end.

  Measured along the cell-centre line of each cell row (the mean of its two
  node rows' edges: curves through the Gauss surface nodes, or chords where
  the run lacks them), from the chain's target end, an outer target before
  an inner one, a lower before an upper.

  Args:
    name: The run's file prefix (without ``_b<k>``).
    path: Directory holding the run's output.
    geqdsk: The run's equilibrium file; its magnetic axis tells an outboard
      midplane crossing from an inboard one (default: the core blocks'
      mean R).

  Returns:
    The :class:`PoloidalLengths`.
  """
  layout = multib_layout(name, path=path, geqdsk=geqdsk)
  return _lengths(layout, layout_cell_edges(layout))


__all__ = [
    "CellEdges", "ChainLengths", "FluxSurfaces", "PoloidalLengths",
    "arc_lengths", "at_cell_centres", "cell_edges", "cell_polygons",
    "centre_line", "corner_values", "edge_polylines", "flux_rows",
    "flux_surfaces", "lagrange_weights", "nodes_by_edges", "poloidal_lengths"
]
