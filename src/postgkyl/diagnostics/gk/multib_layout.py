"""The layout of a gyrokinetic tokamak run's blocks, from its own files.

Nothing is typed in: the regions, the theta chains, the targets and the
midplanes all follow from the block topology (``<name>-block_topo.gkyl``)
and each block's corner nodes (``<name>_b<k>-geo_corn_nodes.gkyl``); a
single-block run has neither topology file nor ``_b<k>`` in its names.

- Region: a block whose psi-upper face is physical is SOL, one whose
  psi-lower face is physical is core or private flux (PF: it has a theta face
  on a target; core: it has none). A block with both or neither psi face
  physical is core when its theta chain closes, else SOL.
- A physical theta face on Z = 0 (span-relative) is the reflecting midplane
  of a half domain; any other physical theta face is a target plate.
- SOL theta chains run from one physical face to the next. Each chain's
  blocks that cross Z = 0 give the midplane: the side (outboard or inboard,
  by R against the core's middle R, else the magnetic axis) and the core
  block under it make a :class:`MidplaneStack`. A target is named upper or
  lower by its Z, outer or inner by the side of its chain's nearest midplane
  crossing.
- Rows run inside out: where the files' psi index grows inward (STEP), or
  one block's index runs against its own topology entry (TCV's fixture), the
  block's rows are reversed and the record says so (``reversed``).
- A block closing on itself in theta whose outer rows end on a sheath (a
  limited run: the sheath boundary condition's ``_bc_z*_vcutsq`` files cover
  those rows only) has its LCFS at the first such row: rows below are core,
  the others SOL, their theta ends targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from postgkyl import io
from postgkyl.diagnostics import discovery

from . import utils
from .nodes import nodes_to_RZ

PHYSICAL = 5  # gkyl_oriented_edge: a physical boundary (gkyl_block_topo.h)
_LOWER_POS, _LOWER_NEG, _UPPER_POS, _UPPER_NEG = 1, 2, 3, 4
_TARGET_ORDER = ("lower outer target", "upper outer target",
                 "lower inner target", "upper inner target")


@dataclass(frozen=True)
class ThetaChain:
  """Blocks joined along theta, in theta order.

  Attributes:
    blocks: The blocks, from the chain's theta-lower end.
    ring: The chain closes on itself (a core).
    start: What its theta-lower end is: ``'ring'``, ``'mirror'`` (a half
      domain's reflecting midplane), ``'target'`` or ``'open'``.
    end: The same for its theta-upper end.
    side: For a SOL chain, ``'outboard'``/``'inboard'`` when every midplane
      crossing lies on that side, ``'SOL'`` when it crosses both; else ``''``.
    labels: For a SOL chain, each block's name along it (``'lower outer
      leg'``, ``'main SOL'``, ``'outboard'``, ...); else empty.
  """

  blocks: tuple[int, ...]
  ring: bool
  start: str
  end: str
  side: str = ""
  labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class Target:
  """A target plate: the theta end of a SOL leg block.

  Attributes:
    name: ``'<lower|upper> <outer|inner> target'`` (with ``' (block k)'``
      appended when the name repeats).
    leg: The SOL block holding the plate.
    pf: The private-flux block under it, or ``None``.
    theta_end: ``0`` (the leg's theta-lower face) or ``-1`` (its upper).
    mid_block: The block of the leg's chain at the midplane, whose crossing
      places the chain's flux surfaces at their midplane radius; ``None`` when
      no block of the chain reaches Z = 0.
    mid_crossing: Theta node position of that block's crossing, where it
      crosses Z = 0 more than once; else ``None``.
    updown: ``'lower'`` or ``'upper'``.
    side: ``'outer'`` or ``'inner'``.
    open_row: The leg's first row on the plate: its LCFS row in a block
      holding closed rows too, else 0.
  """

  name: str
  leg: int
  pf: int | None
  theta_end: int
  mid_block: int | None
  mid_crossing: float | None
  updown: str
  side: str
  open_row: int = 0


@dataclass(frozen=True)
class MidplaneStack:
  """The blocks crossing Z = 0 on one side: a SOL block over a core block.

  Attributes:
    side: ``'outboard'`` or ``'inboard'``.
    core: The core block under the SOL block, or ``None``.
    sol: The SOL block reaching the midplane.
    mirror: The midplane is the reflecting boundary of a half domain.
    crossing: Theta node position of the crossing, where the block crosses
      Z = 0 more than once; else ``None``.
    sep_row: The SOL block's node row on the separatrix (its LCFS row in a
      block holding closed rows too, else 0).
  """

  side: str
  core: int | None
  sol: int
  mirror: bool
  crossing: float | None = None
  sep_row: int = 0

  @property
  def blocks(self) -> tuple[int, ...]:
    """The stack's blocks inside out: the core block, then the SOL block."""
    return tuple(b for b in (self.core, self.sol) if b is not None)


@dataclass(frozen=True, eq=False)
class MultibLayout:
  """What each block of a run is and where it lies.

  Attributes:
    name: The run's file prefix.
    path: Its directory.
    single_block: The run is the single-block app's (no ``_b<k>`` names).
    nodes: Each block's corner nodes ``(nr + 1, nt + 1, 2)`` in (R, Z),
      rows inside out (``reversed`` applied).
    reversed: Each block's rows are the file's reversed.
    connections: The topology ``(num_blocks, 2, 2, 3)`` with psi growing
      outward (psi faces swapped where the files' psi grows inward).
    region: Each block's region: ``'core'``, ``'sol'`` or ``'pf'``.
    under: Each block's aligned psi-lower neighbour, or ``None``.
    lcfs_row: Each block's LCFS node row where it holds closed and open
      field lines (a limited run), else ``None``.
    chains: The theta chains.
    targets: The target plates, lower outer, upper outer, lower inner, upper
      inner first.
    stacks: The midplane stacks, by side.
    half_domain: A theta face is the reflecting midplane of a half domain.
    side_r: The R telling outboard from inboard (and outer from inner
      targets): the magnetic axis when the run's geqdsk was given, else the
      core blocks' middle R, else the nodes' mean R.
  """

  name: str
  path: str
  single_block: bool
  nodes: tuple[np.ndarray, ...]
  reversed: tuple[bool, ...]
  connections: np.ndarray
  region: tuple[str, ...]
  under: tuple[int | None, ...]
  lcfs_row: tuple[int | None, ...]
  chains: tuple[ThetaChain, ...]
  targets: tuple[Target, ...]
  stacks: dict[str, MidplaneStack]
  half_domain: bool
  side_r: float = np.nan

  @property
  def num_blocks(self) -> int:
    return len(self.nodes)

  def side(self, R: float) -> str:
    """``'outboard'`` or ``'inboard'``: R against :attr:`side_r`."""
    return "outboard" if R > self.side_r else "inboard"

  def block_name(self, block: int) -> str:
    """A block's file prefix without the directory: ``<name>_b<k>``, or
    ``<name>`` for a single-block run."""
    return self.name if self.single_block else f"{self.name}_b{block}"

  def file_name(self, block: int, what: str, frame: int | None = None) -> str:
    """The path of a block's ``<what>[_<frame>].gkyl`` file."""
    tail = "" if frame is None else f"_{frame}"
    return os.path.join(self.path,
                        f"{self.block_name(block)}-{what}{tail}.gkyl")

  def rows(self, block: int, values: np.ndarray) -> np.ndarray:
    """A block's cell or node array (psi first) with its rows inside out.

    Only for point values and cell averages: an odd-in-psi modal coefficient
    would change sign.
    """
    return values[::-1] if self.reversed[block] else values

  def file_row(self, block: int, row: int) -> int:
    """The file's psi cell index of a block's inside-out cell row (and back)."""
    nr = self.nodes[block].shape[0] - 1
    return nr - 1 - row if self.reversed[block] else row

  def region_rows(self, block: int, region: str) -> slice:
    """The block's inside-out cell rows in a region (``'core'``, ``'sol'``,
    ``'pf'``; ``'all'``: every row): all or none, except in a block holding
    both closed and open field lines."""
    nr = self.nodes[block].shape[0] - 1
    if region == "all":
      return slice(0, nr)
    m = self.lcfs_row[block]
    if m is not None:
      return {"core": slice(0, m), "sol": slice(m, nr)}.get(region, slice(0, 0))
    return slice(0, nr) if self.region[block] == region else slice(0, 0)

  def __str__(self) -> str:
    kind = "single block" if self.single_block else f"{self.num_blocks} blocks"
    lines = [
        f"{self.name}: {kind}, " +
        ("half domain" if self.half_domain else "full domain")
    ]
    for b, nd in enumerate(self.nodes):
      extra = f", LCFS row {self.lcfs_row[b]}" if self.lcfs_row[
          b] is not None else ""
      lines.append(f"  block {b}: {self.region[b]}, "
                   f"{nd.shape[0] - 1} x {nd.shape[1] - 1} cells" +
                   (", rows reversed" if self.reversed[b] else "") + extra)
    for c in self.chains:
      lines.append(f"  chain {list(c.blocks)}: {c.start} -> {c.end}" +
                   (f" ({c.side})" if c.side else ""))
    for t in self.targets:
      lines.append(f"  {t.name}: leg block {t.leg}, PF block {t.pf}")
    for s in self.stacks.values():
      lines.append(f"  {s.side} midplane: blocks {list(s.blocks)}" +
                   (" (reflecting)" if s.mirror else ""))
    return "\n".join(lines)


# ----------------------------------------------------------- node geometry
def closes(nodes: np.ndarray) -> bool:
  """A block's first and last theta node columns coincide (it closes on
  itself in theta)."""
  return bool(
      np.allclose(nodes[:, 0], nodes[:, -1], rtol=0, atol=1e-9 * np.ptp(nodes)))


def cell_centres(nodes: np.ndarray) -> np.ndarray:
  """The mean of each cell's four corners ``(nr, nt, 2)``."""
  return 0.25 * (nodes[:-1, :-1] + nodes[1:, :-1] + nodes[:-1, 1:] +
                 nodes[1:, 1:])


def _pick(k: np.ndarray, j_near: float | None, offset: float) -> int:
  """The sign change to use: the first, or the one nearest ``j_near`` (for a
  row crossing Z = 0 more than once, as a single null's main SOL does)."""
  return int(k[0]) if j_near is None else int(k[np.argmin(
      np.abs(k + offset - j_near))])


def _at_seam(nodes: np.ndarray, j_near: float | None) -> bool:
  """``j_near`` asks for the crossing at the seam of a block closing on
  itself: within half a node of either end."""
  nt = nodes.shape[1] - 1
  return j_near is not None and min(j_near, nt - j_near) < 0.5 and closes(nodes)


def midplane_node_values(nodes: np.ndarray,
                         values: np.ndarray,
                         crossing: float | None = None) -> np.ndarray:
  """Node values ``(nr + 1, nt + 1)`` at Z = 0 along each node row: linear
  in theta between the two nodes bracketing Z = 0 (the nearest node where a
  row does not cross).

  Args:
    nodes: A block's corner nodes ``(nr + 1, nt + 1, 2)``.
    values: A field at those nodes.
    crossing: Theta node position of the crossing to use where a row crosses
      more than once (default: the first); at either end of a block closing
      on itself, the crossing across its seam.
  """
  out = []
  seam = _at_seam(nodes, crossing)
  for V, Z in zip(values, nodes[..., 1]):
    j_near = crossing
    if seam:
      V, Z = V[[-2, 0, 1]], Z[[-2, 0, 1]]
      j_near = 1
    k = np.where(np.diff(np.sign(Z)) != 0)[0]
    if len(k) == 0:
      out.append(V[np.argmin(np.abs(Z))])
      continue
    k = _pick(k, j_near, 0.5)
    w = Z[k] / (Z[k] - Z[k + 1])
    out.append(V[k] + w * (V[k + 1] - V[k]))
  return np.array(out)


def midplane_radii(nodes: np.ndarray,
                   crossing: float | None = None) -> np.ndarray:
  """R where each node row crosses Z = 0 (:func:`midplane_node_values` of
  R)."""
  return midplane_node_values(nodes, nodes[..., 0], crossing)


def midplane_weights(
    nodes: np.ndarray,
    crossing: float | None = None) -> list[tuple[int, int, float]]:
  """Per cell row ``(j0, j1, w)``: the row's value at Z = 0 is
  ``(1 - w) v[i, j0] + w v[i, j1]``, linear in theta between the two cells
  whose centres bracket Z = 0; at a reflecting midplane (Z = 0 is the block's
  edge) the cell next to it. ``crossing`` as in :func:`midplane_radii`."""
  zc = cell_centres(nodes)[..., 1]
  nt = zc.shape[1]
  seam = _at_seam(nodes, crossing)
  out = []
  for z in zc:
    if seam and z[-1] * z[0] < 0:
      out.append((nt - 1, 0, float(z[-1] / (z[-1] - z[0]))))
      continue
    k = np.where(np.diff(np.sign(z)) != 0)[0]
    if len(k) == 0:
      j = int(np.argmin(np.abs(z)))
      out.append((j, j, 0.0))
      continue
    k = _pick(k, crossing, 1.0)
    out.append((k, k + 1, float(z[k] / (z[k] - z[k + 1]))))
  return out


def _row_crossings(Z: np.ndarray, ztol: float) -> list[float]:
  """Theta node positions where one node row meets Z = 0, in order (each sign
  change once), else the nearest node within ``ztol`` of Z = 0."""
  out: list[float] = []
  for k in np.where(np.diff(np.sign(Z)) != 0)[0]:
    p = k + Z[k] / (Z[k] - Z[k + 1])
    if not out or p - out[-1] > 1e-9:
      out.append(float(p))
  if not out:
    j = int(np.argmin(np.abs(Z)))
    if abs(Z[j]) <= ztol:
      out.append(float(j))
  return out


def _gaps(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
  """The distance of each point to a polyline ``(n, 2)``."""
  a, d = poly[:-1], np.diff(poly, axis=0)
  t = np.clip(((points[:, None] - a) * d).sum(-1) / np.maximum(
      (d * d).sum(-1), 1e-300), 0, 1)
  return np.hypot(*(a + t[..., None] * d -
                    points[:, None]).transpose(2, 0, 1)).min(1)


# ---------------------------------------------------------------- topology
def theta_chains(conn: np.ndarray) -> list[tuple[tuple[int, ...], bool]]:
  """The blocks joined along theta, each ``(blocks in theta order, ring)``;
  ``ring``: the chain closes on itself."""
  nb = len(conn)

  def link(b, e):  # the block continuing the chain across theta face e
    if conn[b, 1, e, 2] == PHYSICAL:
      return None
    p, d, edge = (int(v) for v in conn[b, 1, e])
    return p if d == 1 and edge == (
        _LOWER_POS if e == 1 else _UPPER_POS) else None

  seen, out = set(), []
  for b0 in range(nb):
    if b0 in seen:
      continue
    b, path = b0, [b0]
    while (p := link(b, 0)) is not None and p not in path:
      b = p
      path.append(b)
    ring = link(b, 0) is not None and link(b, 0) in path
    start = b0 if ring else b
    chain, b = [start], start
    while (p := link(b, 1)) is not None and p != start and p not in chain:
      b = p
      chain.append(b)
    seen.update(chain)
    out.append((tuple(chain), ring))
  return out


def flip_psi(conn: np.ndarray) -> np.ndarray:
  """The topology with psi reversed: each block's psi faces swapped, and each
  connection into a psi face pointed at the other one (lower <-> upper, the
  theta orientation kept)."""
  c = conn.copy()
  c[:, 0, 0], c[:, 0, 1] = conn[:, 0, 1], conn[:, 0, 0]
  swap = {
      _LOWER_POS: _UPPER_POS,
      _UPPER_POS: _LOWER_POS,
      _LOWER_NEG: _UPPER_NEG,
      _UPPER_NEG: _LOWER_NEG
  }
  for b, d, e in np.ndindex(c.shape[:3]):
    if c[b, d, e, 2] != PHYSICAL and c[b, d, e, 1] == 0:
      c[b, d, e, 2] = swap[int(c[b, d, e, 2])]
  return c


def psi_grows_outward(conn: np.ndarray,
                      nodes: dict[int, np.ndarray],
                      r_axis: float | None = None) -> bool:
  """Whether the psi index grows outward (row 0 a block's inner edge).

  A closed chain (the core) with one physical psi face says it: that face is
  the core's inner edge. Without one, the geometry: where a block's rows
  cross Z = 0, whether R moves away from the magnetic axis (``r_axis``, else
  the nodes' mean R) as the psi index grows.
  """
  phys = lambda b, e: conn[b, 0, e, 2] == PHYSICAL  # noqa: E731
  for blocks, ring in theta_chains(conn):
    faces = {(bool(phys(b, 0)), bool(phys(b, 1))) for b in blocks}
    closed = ring or all(
        conn[b, 1, e, 2] == PHYSICAL and np.abs(nodes[b][:, 0 if e == 0 else -1,
                                                         1]).max() <= 1e-9 *
        np.ptp(nodes[b][..., 1]) for b, e in ((blocks[0], 0), (blocks[-1], 1)))
    if closed and faces == {(True, False)}:
      return True
    if closed and faces == {(False, True)}:
      return False
  all_r = np.concatenate([nodes[b][..., 0].ravel() for b in nodes])
  r0 = float(r_axis) if r_axis is not None else float(all_r.mean())
  for b in sorted(nodes):
    nd = nodes[b]
    if nd[..., 1].min() < 0 < nd[..., 1].max():
      inner, outer = midplane_radii(nd[[0, -1]])
      return bool((outer - inner) * (1 if inner > r0 else -1) > 0)
  return True


def contrary_blocks(conn: np.ndarray,
                    nodes: dict[int, np.ndarray]) -> tuple[int, ...]:
  """The blocks whose psi index runs against their own topology entry.

  Across each connected psi face, the topology names a block whose psi faces
  lie against the block's other psi face (nearer it, by the median distance
  of the face's nodes). Such a topology joins faces that do not touch;
  reading the block reversed makes its rows run as the topology says.
  """

  def gap(face, p):
    return min(np.median(_gaps(face, nodes[p][e])) for e in (0, -1))

  out = []
  for b, nd in nodes.items():
    against = [
        gap(nd[e - 1], int(conn[b, 0, e, 0]))
        < gap(nd[-e], int(conn[b, 0, e, 0])) for e in (0, 1)
        if conn[b, 0, e, 2] != PHYSICAL
    ]
    if against and all(against):
      out.append(b)
  return tuple(out)


def single_block_topology(nodes: np.ndarray) -> np.ndarray:
  """A single-block run's connections (the app writes none): every face
  physical, except theta when the block closes on itself (periodic)."""
  conn = np.zeros((1, 2, 2, 3), dtype=int)
  conn[..., 2] = PHYSICAL
  if closes(nodes):
    conn[0, 1, 0], conn[0, 1, 1] = (0, 1, _UPPER_POS), (0, 1, _LOWER_POS)
  return conn


def derive_layout(conn: np.ndarray,
                  nodes: dict[int, np.ndarray],
                  r_axis: float | None = None,
                  lcfs: dict[int, int] | None = None) -> dict:
  """The regions, chains, targets and midplane stacks of a topology whose
  psi grows outward (:func:`flip_psi` it first where it does not).

  Args:
    conn: The topology ``(num_blocks, 2, 2, 3)``.
    nodes: Each block's corner nodes ``(nr + 1, nt + 1, 2)``, rows inside out.
    r_axis: R of the magnetic axis, telling outboard from inboard; else the
      core blocks' middle R, else the nodes' mean R.
    lcfs: For a block closing on itself in theta whose rows from its LCFS on
      end on a sheath at its theta ends (a limited run), that LCFS row.

  Returns:
    A dict of :class:`MultibLayout`'s ``region``, ``under``, ``chains``,
    ``targets``, ``stacks``, ``half_domain`` and ``side_r`` fields.
  """
  nb = len(conn)
  lcfs = lcfs or {}
  all_z = np.concatenate([nodes[b][..., 1].ravel() for b in range(nb)])
  ztol = 1e-9 * np.ptp(all_z)

  def phys(b, d, e):
    return conn[b, d, e, 2] == PHYSICAL

  def face_kind(b, e):  # a physical theta face: on Z = 0 a mirror
    col = nodes[b][:, 0 if e == 0 else -1, 1]
    return "mirror" if np.abs(col).max() <= ztol else "target"

  chains = []
  for blocks, ring in theta_chains(conn):
    c = dict(blocks=blocks, ring=ring)
    for which, b, e in (("start", blocks[0], 0), ("end", blocks[-1], 1)):
      c[which] = "ring" if ring else face_kind(b, e) if phys(b, 1,
                                                             e) else "open"
    c["closed"] = ring or (c["start"] == c["end"] == "mirror")
    chains.append(c)
  chain_of = {b: c for c in chains for b in c["blocks"]}

  region = {}
  for b in range(nb):
    lo, hi = phys(b, 0, 0), phys(b, 0, 1)
    if hi and not lo:
      region[b] = "sol"
    elif lo and not hi:
      tgt = any(phys(b, 1, e) and face_kind(b, e) == "target" for e in (0, 1))
      region[b] = "pf" if tgt else "core"
    else:  # both psi faces physical (a single block) or neither
      region[b] = "core" if chain_of[b]["closed"] else "sol"

  def inner_of(b):  # the aligned block under b, or None
    if phys(b, 0, 0):
      return None
    p, d, e = (int(v) for v in conn[b, 0, 0])
    return p if d == 0 and e == _UPPER_POS and nodes[p].shape[1] == nodes[
        b].shape[1] else None

  cores = [b for b in range(nb) if region[b] == "core"]
  if r_axis is not None:
    r_ref = float(r_axis)
  elif cores:
    core_r = np.concatenate([nodes[b][..., 0].ravel() for b in cores])
    r_ref = 0.5 * float(core_r.min() + core_r.max())
  else:
    r_ref = float(
        np.mean(np.concatenate([nodes[b][..., 0].ravel() for b in range(nb)])))
  side_of = lambda R: "outboard" if R > r_ref else "inboard"  # noqa: E731

  def crossings(b, row):  # (theta node position, R) of a row at Z = 0
    R0, Z0 = nodes[b][row, :, 0], nodes[b][row, :, 1]
    out = []
    for p in _row_crossings(Z0, ztol):
      k = min(int(p), len(R0) - 2)
      out.append((p, R0[k] + (p - k) * (R0[k + 1] - R0[k])))
    return out

  sol_chains = sorted([
      c for c in chains if not c["closed"] and all(region[b] == "sol"
                                                   for b in c["blocks"])
  ],
                      key=lambda c: c["blocks"][0])
  span = np.ptp(np.concatenate([nodes[b][..., 0].ravel() for b in range(nb)]))
  stacks, targets = {}, []
  for c in sol_chains:
    cross = []  # (block, theta node position, R); a shared node once
    for b in c["blocks"]:
      for p, R in crossings(b, 0):
        if not cross or abs(R - cross[-1][2]) > 1e-9 * span:
          cross.append((b, p, R))
    nby = {b: sum(1 for x in cross if x[0] == b) for b in c["blocks"]}
    c["cross"] = [(b, p if nby[b] > 1 else None, side_of(R))
                  for b, p, R in cross]
    sides = list(dict.fromkeys(s for _, _, s in c["cross"]))
    c["side"] = sides[0] if len(sides) == 1 else "SOL"
    for b, j, side in c["cross"]:
      if side in stacks:
        continue
      core = inner_of(b)
      core = core if core is not None and region[core] == "core" else None
      at_mirror = "mirror" in (c["start"], c["end"])
      if len(c["cross"]) > 1:  # at the mirror only if this end is on it
        at_mirror = ((c["start"] == "mirror" and b == c["blocks"][0])
                     or (c["end"] == "mirror" and b == c["blocks"][-1]))
      stacks[side] = MidplaneStack(side, core, b, at_mirror, j)
    labels, last = [], len(c["blocks"]) - 1
    for k, b in enumerate(c["blocks"]):
      ends = [
          e for e, first in ((0, k == 0), (1, k == last))
          if first and c["start" if e == 0 else "end"] == "target"
      ]
      for e in ends:
        zt = nodes[b][:, 0 if e == 0 else -1, 1].mean()
        updown = "upper" if zt > 0 else "lower"
        near = (
            c["cross"][0] if e == 0 else c["cross"][-1]) if c["cross"] else None
        if near is not None:
          io_side = "outer" if near[2] == "outboard" else "inner"
        else:
          io_side = "outer" if nodes[b][:, 0 if e == 0 else -1,
                                        0].mean() > r_ref else "inner"
        name = f"{updown} {io_side} target"
        if any(t.name == name for t in targets):
          name += f" (block {b})"
        targets.append(
            Target(name, b, inner_of(b), 0 if e == 0 else -1,
                   near[0] if near else None, near[1] if near else None, updown,
                   io_side))
      own = list(dict.fromkeys(s for bb, _, s in c["cross"] if bb == b))
      if len(ends) == 1:
        labels.append(f"{targets[-1].updown} {targets[-1].side} leg")
      else:
        labels.append(own[0] if len(own) == 1 else "main SOL" if len(own) >
                      1 else c["side"] if c["side"] != "SOL" else "SOL")
    c["labels"] = tuple(labels)
  # A limited block: its LCFS row's midplane crossings make the stacks, its
  # open rows at each theta end a target.
  for b, m in lcfs.items():
    if not chain_of[b]["ring"] or len(chain_of[b]["blocks"]) != 1:
      continue
    cross = crossings(b, m)
    nt = nodes[b].shape[1] - 1
    if abs(nodes[b][m, 0, 1]) <= ztol and not any(
        min(p, nt - p) < 0.5 for p, _ in cross):  # its seam on Z = 0
      cross.insert(0, (0.0, float(nodes[b][m, 0, 0])))
    for p, R in cross:
      stacks.setdefault(
          side_of(R),
          MidplaneStack(side_of(R), None, b, False,
                        p if len(cross) > 1 else None, m))
    mid = max(cross, key=lambda x: x[1])[0] if cross else None
    for e in (0, -1):
      updown = "upper" if cell_centres(nodes[b][m:])[:, e,
                                                     1].mean() > 0 else "lower"
      io_side = "outer" if nodes[b][m:, e, 0].mean() > r_ref else "inner"
      targets.append(
          Target(f"{updown} {io_side} target", b, None, e, b if cross else None,
                 mid, updown, io_side, m))
  targets.sort(key=lambda t: (_TARGET_ORDER.index(t.name) if t.name in
                              _TARGET_ORDER else len(_TARGET_ORDER), t.name))
  return dict(region=tuple(region[b] for b in range(nb)),
              under=tuple(inner_of(b) for b in range(nb)),
              chains=tuple(
                  ThetaChain(c["blocks"], c["ring"], c["start"], c["end"],
                             c.get("side", ""), c.get("labels", ()))
                  for c in chains),
              targets=tuple(targets),
              stacks=stacks,
              half_domain=any("mirror" in (c["start"], c["end"])
                              for c in chains),
              side_r=r_ref)


# ------------------------------------------------------------------ reading
def block_prefixes(name: str, path: str = "./") -> tuple[str, ...]:
  """Each block's file prefix: ``<path>/<name>_b<k>`` for a multiblock run
  (its ``<name>-block_topo.gkyl`` gives the count), else ``<path>/<name>``
  (the single-block app writes no topology and no ``_b<k>``)."""
  base = os.path.join(path.rstrip("/") or ".", name)
  topo = f"{base}-block_topo.gkyl"
  if not os.path.exists(topo):
    return (base, )
  return tuple(f"{base}_b{b}"
               for b in range(int(io.read_metadata(topo)["num_blocks"])))


def _corner_nodes(file_name: str) -> np.ndarray:
  """(R, Z) of a block's corner nodes ``(nr + 1, nt + 1, 2)``."""
  _, values, gdata = utils.read_gfile(file_name)
  mapc2p = io.geometry.is_geo_mapc2p(gdata.ctx)
  return np.stack(nodes_to_RZ(values, mapc2p), axis=-1)


def psi_cells(grid: dict, sub: dict) -> range:
  """The psi cells of a field's grid (metadata ``lower``, ``upper``,
  ``cells``; psi first) that a file on a sub-range of it covers."""
  x0 = grid["lower"][0]
  dx = (grid["upper"][0] - x0) / grid["cells"][0]
  return range(int(round((sub["lower"][0] - x0) / dx)),
               int(round((sub["upper"][0] - x0) / dx)))


def _sheath_files(path: str, frame: int) -> dict[str, str]:
  """Each block prefix's first ``<prefix>-<species>_bc_z<lower|upper>_vcutsq``
  file at ``frame`` (the sheath boundary condition writes them where it
  applies: a limited run's SOL rows only). One listing of the directory: a
  run's directory can hold 10^5 files."""
  tail = f"_vcutsq_{frame}.gkyl"
  found: dict[str, str] = {}
  for n in sorted(os.listdir(path)):
    if n.endswith(tail) and "_bc_z" in n and "-" in n:
      found.setdefault(os.path.join(path,
                                    n.split("-", 1)[0]), os.path.join(path, n))
  return found


def _sheath_rows(prefix: str, frame: int, sheath: str | None) -> range | None:
  """The file psi cells of a block that the sheath file covers, or ``None``
  where it has none."""
  if sheath is None:
    return None
  return psi_cells(
      utils.read_gfile(f"{prefix}-field_{frame}.gkyl")[2].ctx,
      utils.read_gfile(sheath)[2].ctx)


def multib_layout(name: str,
                  *,
                  path: str = "./",
                  geqdsk: str | None = None,
                  frame: int | None = None) -> MultibLayout:
  """The layout of a run's blocks: regions, theta chains, targets and
  midplanes, derived from its topology and corner nodes.

  Args:
    name: The run's file prefix (without ``_b<k>``).
    path: Directory holding the run's output.
    geqdsk: The run's equilibrium file; its magnetic axis tells inboard from
      outboard where the run has no core block, and the psi direction where
      no closed chain says it. ``None``: the nodes' mean R.
    frame: The frame whose sheath files mark a limited run's LCFS (default:
      the first frame written).

  Returns:
    The :class:`MultibLayout`.
  """
  path = path.rstrip("/") or "."
  prefixes = block_prefixes(name, path)
  single = not os.path.exists(os.path.join(path, f"{name}-block_topo.gkyl"))
  nodes = {
      b: _corner_nodes(f"{p}-geo_corn_nodes.gkyl")
      for b, p in enumerate(prefixes)
  }
  if single:
    conn = single_block_topology(nodes[0])
    contrary: tuple[int, ...] = ()
  else:
    meta = io.read_metadata(os.path.join(path, f"{name}-block_topo.gkyl"))
    conn = np.array(meta["connections"]).reshape(len(nodes), int(meta["ndim"]),
                                                 2, 3)
    contrary = contrary_blocks(conn, nodes)
  r_axis = io.read_geqdsk_header(geqdsk).rmaxis if geqdsk else None
  flip = not psi_grows_outward(conn, {
      b: nd[::-1] if b in contrary else nd
      for b, nd in nodes.items()
  }, r_axis)
  if flip:
    conn = flip_psi(conn)
  rev = tuple(flip != (b in contrary) for b in range(len(nodes)))
  nodes = {b: nd[::-1] if rev[b] else nd for b, nd in nodes.items()}

  if frame is None:
    frames = sorted(discovery.available_frames(f"{prefixes[0]}-field_"))
    frame = frames[0] if frames else None
  lcfs = {}
  sheath = _sheath_files(path, frame) if frame is not None else {}
  for b in range(len(nodes)) if frame is not None else ():
    open_rows = _sheath_rows(prefixes[b], frame, sheath.get(prefixes[b]))
    if open_rows is not None and len(open_rows):
      nr = nodes[b].shape[0] - 1
      rows = sorted(nr - 1 - i if rev[b] else i for i in open_rows)
      if 0 < rows[0] and rows == list(range(rows[0], nr)):
        lcfs[b] = rows[0]
  parts = derive_layout(conn, nodes, r_axis, lcfs)
  return MultibLayout(name=name,
                      path=path,
                      single_block=single,
                      nodes=tuple(nodes[b] for b in range(len(nodes))),
                      reversed=rev,
                      connections=conn,
                      lcfs_row=tuple(lcfs.get(b) for b in range(len(nodes))),
                      **parts)
