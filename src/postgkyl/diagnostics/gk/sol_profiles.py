"""Per-frame scrape-off-layer profiles of a gyrokinetic tokamak run.

The value of a quantity at the midplane across the flux surfaces, and the heat
flux onto each target plate with its width lambda_q, from each block's cell
averages laid out by :func:`~.multib_layout.multib_layout` (rows inside out).

- Midplane: through one side's :class:`~.multib_layout.MidplaneStack` (its
  core block, then its SOL block), each cell row's value at Z = 0 is linear in
  theta between the two cells whose centres bracket Z = 0 (at a reflecting
  midplane, the cell next to it), placed at the R where the row crosses
  Z = 0: the mean of its two node rows' crossings. The separatrix is the SOL
  block's separatrix node row there.
- Target: ``|q|`` in the plate cell of each SOL flux surface of the leg block,
  each surface placed along the plate and at its distance from the separatrix
  at the midplane (the crossing of the leg's chain). lambda_q is the midplane
  distance holding a fraction (0.63) of the profile's integral, ``q``
  piecewise constant between the surfaces' node rows; it is undefined (nan)
  where those rows do not move away from the separatrix at the midplane row
  after row, as on a grid too coarse in theta for the crossings, linear
  between nodes, to follow the flux surfaces.

Quantities are the gyrokinetic registry's (:func:`~.load_quantity.
load_quantity`): each block's datasets are summed in their modal form over the
listed quantities and species, then reduced to cell averages.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from postgkyl.dg.modal import cell_means
from postgkyl.gdatastate.layout import require_dg_layout

from .flux_coords import CellEdges, layout_cell_edges, midplane_row_radii
from .load_quantity import load_quantity
from .multib_layout import (MidplaneStack, MultibLayout, Target, midplane_radii,
                            midplane_weights, multib_layout)

# The registry quantities summing to the parallel flux of kinetic energy,
# (m/2) M3par + (m/2) M3perp, per species.
HEAT_FLUX = ("qpar", "qperp")
LAMBDA_FRACTION = 0.63
# The sign of R - R_sep into the SOL on each side.
_OUTWARD = {"outboard": 1.0, "inboard": -1.0}


def _freeze(record, *names: str) -> None:
  """Store read-only copies of a record's array fields, of the same length."""
  arrays = [np.array(getattr(record, n)) for n in names]
  if len({len(a) for a in arrays}) > 1:
    raise ValueError(f"{type(record).__name__}: " + ", ".join(names) +
                     " must have the same length, got " +
                     ", ".join(str(len(a)) for a in arrays))
  for n, a in zip(names, arrays):
    a.setflags(write=False)
    object.__setattr__(record, n, a)


def _check_fraction(fraction: float) -> None:
  if not 0 < fraction <= 1:
    raise ValueError(f"fraction must lie in (0, 1], got {fraction}")


@dataclass(frozen=True)
class MidplaneProfile:
  """A quantity at Z = 0 across one side's flux surfaces, inside out: the
  core block's cell rows, then the SOL block's.

  Attributes:
    stack: The midplane stack (its side, core and SOL blocks, crossing).
    block: Each row's block.
    row: Each row's psi cell index in its block's files.
    R: Each row's R at Z = 0 (m).
    value: The quantity at Z = 0, in the registry's units (SI; temperatures
      in J).
    R_sep: R of the separatrix at Z = 0 (m).
  """

  stack: MidplaneStack
  block: np.ndarray
  row: np.ndarray
  R: np.ndarray
  value: np.ndarray
  R_sep: float

  def __post_init__(self):
    _freeze(self, "block", "row", "R", "value")

  @property
  def distance(self) -> np.ndarray:
    """Each row's distance from the separatrix at Z = 0 (m), positive into the
    SOL on either side: ``R - R_sep`` outboard, ``R_sep - R`` inboard."""
    return _OUTWARD[self.stack.side] * (self.R - self.R_sep)

  def __str__(self) -> str:
    st = self.stack
    lines = [
        f"{st.side} midplane" + (" (reflecting)" if st.mirror else "") +
        f": blocks {list(st.blocks)}, R_sep = {self.R_sep:.6g} m",
        f"  {'block':>5} {'row':>4} {'R (m)':>12} {'distance (m)':>13} "
        f"{'value':>13}"
    ]
    for b, i, R, d, v in zip(self.block, self.row, self.R, self.distance,
                             self.value):
      lines.append(f"  {b:5d} {i:4d} {R:12.6f} {d:13.5e} {v:13.5e}")
    return "\n".join(lines)


@dataclass(frozen=True)
class TargetProfile:
  """The heat flux onto one target plate per SOL flux surface, from the
  separatrix out.

  Attributes:
    target: The plate (its leg block, theta end, midplane block, ...).
    row: Each surface's psi cell index in the leg block's files.
    heat_flux: ``|q|`` in the plate cell (W/m^2), ``q`` the parallel flux of
      kinetic energy (:data:`HEAT_FLUX`) summed over the species.
    plate_distance: Each surface's distance along the plate from its
      separatrix corner (m): the middle of its face between its two nodes,
      along the straight segments joining the face's nodes.
    distance: Each surface's distance from the separatrix at the midplane (m),
      the middle of its two node rows' distances; ``None`` where no block of
      the leg's chain reaches Z = 0 (``target.mid_block`` is ``None``).
    lambda_q: The midplane distance holding ``fraction`` of the integral of
      ``heat_flux`` over it (m); nan without ``distance``, without a positive
      integral, or where the surfaces' node rows do not move away from the
      separatrix at the midplane, row after row (a grid too coarse in theta
      for the crossings, linear between nodes, to follow the flux surfaces).
    fraction: The fraction of the integral ``lambda_q`` holds.
  """

  target: Target
  row: np.ndarray
  heat_flux: np.ndarray
  plate_distance: np.ndarray
  distance: np.ndarray | None
  lambda_q: float
  fraction: float

  def __post_init__(self):
    _check_fraction(self.fraction)
    _freeze(self, "row", "heat_flux", "plate_distance",
            *(() if self.distance is None else ("distance", )))
    if self.distance is None and not np.isnan(self.lambda_q):
      raise ValueError("TargetProfile: lambda_q needs a midplane distance")

  def __str__(self) -> str:
    t = self.target
    where = (f"midplane block {t.mid_block}" if self.distance is not None else
             "no block of its chain reaches Z = 0")
    lam = (f", lambda_q = {self.lambda_q:.5e} m "
           f"({self.fraction:g} of the integral)"
           if not np.isnan(self.lambda_q) else "" if self.distance is None else
           ", no lambda_q (no heat flux, or midplane distances not increasing)")
    lines = [
        f"{t.name}: leg block {t.leg}, theta end {t.theta_end}, {where}" + lam,
        f"  {'row':>4} {'distance (m)':>13} {'plate (m)':>12} "
        f"{'|q| (W/m^2)':>13}"
    ]
    dist = (self.distance if self.distance is not None else np.full(
        len(self.row), np.nan))
    for i, d, p, q in zip(self.row, dist, self.plate_distance, self.heat_flux):
      lines.append(f"  {i:4d} {d:13.5e} {p:12.5e} {q:13.5e}")
    return "\n".join(lines)


# ---------------------------------------------------- pure geometry and maths
def midplane_values(nodes: np.ndarray,
                    cells: np.ndarray,
                    crossing: float | None = None) -> np.ndarray:
  """A block's cell averages at Z = 0, one per cell row: linear in theta
  between the two cells whose centres bracket Z = 0, or the cell next to Z = 0
  where it is the block's edge (a reflecting midplane).

  Args:
    nodes: The block's corner nodes ``(nr + 1, nt + 1, 2)``.
    cells: Its cell averages ``(nr, nt)``, rows as ``nodes``.
    crossing: Which crossing, as for :func:`~.multib_layout.midplane_radii`.
  """
  return np.array([(1 - w) * cells[i, j0] + w * cells[i, j1]
                   for i, (j0, j1,
                           w) in enumerate(midplane_weights(nodes, crossing))])


def midplane_cell_radii(nodes: np.ndarray,
                        crossing: float | None = None) -> np.ndarray:
  """R at Z = 0 of each cell row of a block: the mean of its two node rows'
  crossings (:func:`~.multib_layout.midplane_radii`)."""
  return _mids(midplane_radii(nodes, crossing))


def _mids(edges: np.ndarray) -> np.ndarray:
  return 0.5 * (edges[1:] + edges[:-1])


def _row_radii(layout: MultibLayout, edges: CellEdges | None, block: int,
               crossing: float | None) -> np.ndarray:
  """Each node row's R at Z = 0: on the curved cell edges where given
  (:func:`~.flux_coords.midplane_row_radii`), else on the chords."""
  if edges is None:
    return midplane_radii(layout.nodes[block], crossing)
  return midplane_row_radii(layout, edges, block, crossing)


def heat_flux_width(edges: np.ndarray, flux: np.ndarray,
                    fraction: float) -> float:
  """Where the integral of a flux from ``edges[0]`` reaches ``fraction`` of
  its total, the flux constant between consecutive edges (linear in between).

  Args:
    edges: Non-decreasing positions ``(n + 1,)``.
    flux: The flux between each pair of edges ``(n,)``.
    fraction: The fraction of the total, in (0, 1].

  Returns:
    The position on ``edges``' axis; nan when the total is not positive.

  Raises:
    ValueError: ``edges`` decrease somewhere, or ``fraction`` lies outside
      (0, 1].
  """
  _check_fraction(fraction)
  if np.any(np.diff(edges) < 0):
    raise ValueError("heat_flux_width: edges must not decrease")
  integral = np.concatenate([[0.0], np.cumsum(flux * np.diff(edges))])
  if not integral[-1] > 0:
    return np.nan
  return float(np.interp(fraction * integral[-1], integral, edges))


def plate_distance(face: np.ndarray) -> np.ndarray:
  """Each cell's distance along a face from its first node: the middle of the
  cell's segment, along the straight segments joining the face's nodes
  ``(n + 1, 2)``."""
  s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(face, axis=0).T))])
  return 0.5 * (s[1:] + s[:-1])


def _stack(layout: MultibLayout, side: str) -> MidplaneStack:
  if side not in layout.stacks:
    raise ValueError(f"{layout.name} has no {side} midplane; its midplanes: "
                     f"{', '.join(layout.stacks) or 'none'}")
  return layout.stacks[side]


def stack_profile(layout: MultibLayout,
                  side: str,
                  cells: Mapping[int, np.ndarray],
                  edges: CellEdges | None = None) -> MidplaneProfile:
  """A quantity at Z = 0 through one side's midplane stack.

  Args:
    layout: The run's layout.
    side: ``'outboard'`` or ``'inboard'``.
    cells: Each stack block's cell averages ``(nr, nt)``, rows inside out.
    edges: The run's cell edges, to find each row's R at Z = 0 on its curve;
      ``None``: on the chords between nodes.
  """
  st = _stack(layout, side)
  radii = {b: _row_radii(layout, edges, b, st.crossing) for b in st.blocks}
  nodes = layout.nodes
  rows = {b: range(nodes[b].shape[0] - 1) for b in st.blocks}
  return MidplaneProfile(
      stack=st,
      block=np.concatenate([np.full(len(rows[b]), b) for b in st.blocks]),
      row=np.array([layout.file_row(b, i) for b in st.blocks for i in rows[b]]),
      R=np.concatenate([_mids(radii[b]) for b in st.blocks]),
      value=np.concatenate([
          midplane_values(nodes[b], cells[b], st.crossing) for b in st.blocks
      ]),
      R_sep=float(radii[st.sol][st.sep_row]))


def plate_profile(layout: MultibLayout,
                  target: Target,
                  flux: np.ndarray,
                  fraction: float = LAMBDA_FRACTION,
                  edges: CellEdges | None = None) -> TargetProfile:
  """The heat flux onto one target plate per SOL flux surface.

  Args:
    layout: The run's layout.
    target: One of its targets.
    flux: The leg block's heat-flux cell averages ``(nr, nt)``, rows inside
      out.
    fraction: The fraction of the integral lambda_q holds.
    edges: As for :func:`stack_profile`.
  """
  r0, end = target.open_row, target.theta_end
  nodes = layout.nodes[target.leg]
  heat_flux = np.abs(flux[r0:, end])
  distance, lam = None, np.nan
  if target.mid_block is not None:
    radii = _row_radii(layout, edges, target.mid_block,
                       target.mid_crossing)[r0:]
    gap = np.abs(radii - radii[0])
    distance = 0.5 * (gap[1:] + gap[:-1])
    if np.all(np.diff(gap) >= 0):
      lam = heat_flux_width(gap, heat_flux, fraction)
  return TargetProfile(target=target,
                       row=np.array([
                           layout.file_row(target.leg, i)
                           for i in range(r0, nodes.shape[0] - 1)
                       ]),
                       heat_flux=heat_flux,
                       plate_distance=plate_distance(nodes[r0:, end]),
                       distance=distance,
                       lambda_q=lam,
                       fraction=fraction)


# ------------------------------------------------------------------ reading
def _names(values, what: str) -> list[str]:
  if isinstance(values, str):
    raise TypeError(f"{what} is a list of names, e.g. [{values!r}]")
  if not values:
    raise ValueError(f"{what} needs at least one name")
  return list(values)


def _cell_averages(layout: MultibLayout,
                   block: int,
                   quantity: list[str],
                   species: list[str] | None,
                   frame: int,
                   gamma_e: float | None = None,
                   gamma_i: float | None = None) -> np.ndarray:
  """One block's cell averages ``(nr, nt)``, rows inside out, of the sum of
  every dataset :func:`load_quantity` returns for the quantities and
  species."""
  stem = layout.block_name(block)
  total = None
  for q in quantity:
    for data in load_quantity(q,
                              ",".join(species) if species else None,
                              stem,
                              str(frame),
                              path=layout.path,
                              gamma_e=gamma_e,
                              gamma_i=gamma_i):
      total = data if total is None else total + data
  dg = require_dg_layout(total)
  if dg.num_fields != 1:
    raise ValueError(f"{', '.join(quantity)}: {dg.num_fields} components; "
                     "a profile needs a scalar quantity")
  return layout.rows(block,
                     cell_means(total.values, dg.ndim, dg.block_size)[..., 0])


def midplane_profile(quantity: list[str],
                     name: str,
                     frame: int,
                     *,
                     species: list[str] | None = None,
                     path: str = "./",
                     side: Literal["outboard", "inboard"] = "outboard",
                     geqdsk: str | None = None,
                     gamma_e: float | None = None,
                     gamma_i: float | None = None) -> MidplaneProfile:
  """A quantity at the midplane (Z = 0) of a tokamak run across its flux
  surfaces, from the core edge out through the SOL, with each surface's R and
  distance from the separatrix.

  The quantity is the sum of the listed registry quantities over the listed
  species, each block's datasets summed in their modal form, reduced to cell
  averages. Through the side's midplane stack (the core block, then the SOL
  block), each cell row's value at Z = 0 is linear in theta between the two
  cells whose centres bracket Z = 0 (at a reflecting midplane, the cell next
  to it), at the R where the row crosses Z = 0.

  Args:
    quantity: Registry quantities to sum (``quantity_catalog()`` lists
      them): ``['M0']`` the density, ``['temp']`` the temperature (J),
      ``['field']`` the potential, ``['qpar', 'qperp']`` the parallel flux of
      kinetic energy.
    name: The run's file prefix (without ``_b<k>``).
    frame: The frame.
    species: Species to sum over; ``None`` for species-independent
      quantities. A quantity combining several species (``mach_hot_i``) reads
      the whole list once.
    path: Directory holding the run's output.
    side: The midplane: ``'outboard'`` or ``'inboard'``.
    geqdsk: The run's equilibrium file, for its layout (see
      :func:`~.multib_layout.multib_layout`).

    gamma_e: The electrons' adiabatic index in the sound-speed quantities
      (``c_s_hot_i``, ``mach_hot_i``); ``None``: their default (1).
    gamma_i: The ions' adiabatic index there; ``None``: their default (3).
  Returns:
    The :class:`MidplaneProfile`.
  """
  quantity = _names(quantity, "quantity")
  species = None if species is None else _names(species, "species")
  layout = multib_layout(name, path=path, geqdsk=geqdsk)
  cells = {
      b: _cell_averages(layout, b, quantity, species, frame, gamma_e, gamma_i)
      for b in _stack(layout, side).blocks
  }
  return stack_profile(layout, side, cells, layout_cell_edges(layout))


def target_profiles(
    name: str,
    frame: int,
    *,
    species: list[str],
    path: str = "./",
    geqdsk: str | None = None,
    fraction: float = LAMBDA_FRACTION) -> tuple[TargetProfile, ...]:
  """The heat flux onto each target plate of a tokamak run per SOL flux
  surface, with each surface's distance from the separatrix at the midplane
  and along the plate, and the heat-flux width lambda_q.

  The heat flux is ``|q|`` in the plate cell, ``q`` the parallel flux of
  kinetic energy (``qpar + qperp``) summed over the species, reduced to cell
  averages. lambda_q is the midplane distance holding ``fraction`` of its
  integral over the midplane distance, constant between the surfaces' node
  rows; it is nan where no block of the leg's chain reaches the midplane, or
  where the node rows' midplane distances do not increase outward.

  Args:
    name: The run's file prefix (without ``_b<k>``).
    frame: The frame.
    species: Species to sum over.
    path: Directory holding the run's output.
    geqdsk: The run's equilibrium file, for its layout (see
      :func:`~.multib_layout.multib_layout`).
    fraction: The fraction of the integral lambda_q holds.

  Returns:
    One :class:`TargetProfile` per target, in the layout's order (lower
    outer, upper outer, lower inner, upper inner first).
  """
  species = _names(species, "species")
  _check_fraction(fraction)
  layout = multib_layout(name, path=path, geqdsk=geqdsk)
  edges = layout_cell_edges(layout)
  flux = {
      b: _cell_averages(layout, b, list(HEAT_FLUX), species, frame)
      for b in sorted({t.leg
                       for t in layout.targets})
  }
  return tuple(
      plate_profile(layout, t, flux[t.leg], fraction, edges)
      for t in layout.targets)


__all__ = [
    "HEAT_FLUX",
    "LAMBDA_FRACTION",
    "MidplaneProfile",
    "TargetProfile",
    "midplane_values",
    "midplane_cell_radii",
    "heat_flux_width",
    "plate_distance",
    "stack_profile",
    "plate_profile",
    "midplane_profile",
    "target_profiles",
]
