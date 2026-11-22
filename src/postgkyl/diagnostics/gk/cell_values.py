"""Cell averages of a gyrokinetic quantity in the blocks of a tokamak run, at
its frames.

The quantity is the sum of registry quantities (:func:`~.load_quantity.
load_quantity`) over species, each block's datasets summed in their modal
form at each frame, then reduced to cell averages. A block's cell averages are
``(n_f, nr, nt)``: frames, then the block's cell rows inside out (the layout's
orientation, :meth:`~.multib_layout.MultibLayout.rows`), then its theta cells.

These are what a view of the run draws per cell (maps, movies, probes), and
what the midplane and target profiles (:mod:`.sol_profiles`) interpolate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

import numpy as np

from postgkyl.cli_spec import CliType
from postgkyl.dg.modal import cell_means
from postgkyl.gdatastate.layout import require_dg_layout

from .distf import FrameSpec
from .load_quantity import load_quantity
from .multib_layout import MultibLayout, multib_layout


@dataclass(frozen=True, eq=False)
class CellValues:
  """Cell averages of a quantity per block at a run's frames.

  Attributes:
    name: The run's file prefix.
    path: Its directory.
    quantity: The registry quantities summed.
    species: The species summed over, or ``None`` (species-independent
      quantities).
    blocks: The blocks held, in the order asked for.
    frames: The frame numbers ``(n_f,)``, ascending.
    time: Each frame's time (s) ``(n_f,)``.
    values: Each block's cell averages ``(n_f, nr, nt)``, rows inside out,
      in the order of ``blocks``; in the registry's units (SI, temperatures
      in J).
  """

  name: str
  path: str
  quantity: tuple[str, ...]
  species: tuple[str, ...] | None
  blocks: tuple[int, ...]
  frames: np.ndarray
  time: np.ndarray
  values: tuple[np.ndarray, ...]

  def __post_init__(self):
    frames, time = np.array(self.frames), np.array(self.time)
    if len(frames) == 0 or frames.shape != time.shape or frames.ndim != 1:
      raise ValueError("CellValues: frames and time must be one value per "
                       f"frame, at least one; got {frames.shape} and "
                       f"{time.shape}")
    values = tuple(np.array(v) for v in self.values)
    if len(values) != len(self.blocks) or any(
        v.ndim != 3 or len(v) != len(frames) for v in values):
      raise ValueError(f"CellValues: values must hold one (frames = "
                       f"{len(frames)}, rows, theta cells) array per block "
                       f"of {list(self.blocks)}")
    for a in (frames, time, *values):
      a.setflags(write=False)
    object.__setattr__(self, "frames", frames)
    object.__setattr__(self, "time", time)
    object.__setattr__(self, "values", values)

  def block(self, b: int) -> np.ndarray:
    """Block ``b``'s cell averages ``(n_f, nr, nt)``.

    Raises:
      ValueError: The record does not hold block ``b``.
    """
    if b not in self.blocks:
      raise ValueError(f"{self.name}: cell values of blocks "
                       f"{list(self.blocks)}, not {b}")
    return self.values[self.blocks.index(b)]

  def __str__(self) -> str:
    n = len(self.frames)
    what = " + ".join(self.quantity) + ("" if self.species is None else
                                        f" over {', '.join(self.species)}")
    lines = [
        f"{self.name}: cell averages of {what}, " +
        (f"frame {self.frames[0]}"
         if n == 1 else f"{n} frames ({self.frames[0]} to {self.frames[-1]})"),
        f"  {'block':>5} {'cells':>9} {'min':>13} {'max':>13}  "
        "(finite cells, every frame)"
    ]
    for b, v in zip(self.blocks, self.values):
      finite = v[np.isfinite(v)]
      lo, hi = ((np.min(finite), np.max(finite)) if finite.size else
                (np.nan, np.nan))
      lines.append(f"  {b:5d} {f'{v.shape[1]}x{v.shape[2]}':>9} "
                   f"{lo:13.5e} {hi:13.5e}")
    return "\n".join(lines)


def name_list(values, what: str) -> list[str]:
  """A non-empty list of names (a bare string is refused: it would be read
  as its characters)."""
  if isinstance(values, str):
    raise TypeError(f"{what} is a list of names, e.g. [{values!r}]")
  if not values:
    raise ValueError(f"{what} needs at least one name")
  return list(values)


def _frame_text(frame: FrameSpec) -> str:
  """A frame specification as :func:`load_quantity` reads it."""
  if isinstance(frame, (list, tuple)):
    return ",".join(str(int(k)) for k in frame)
  return str(frame)


def _block_cells(
    layout: MultibLayout,
    block: int,
    quantity: list[str],
    species: list[str] | None,
    frame: str,
    gamma_e: float | None = None,
    gamma_i: float | None = None) -> tuple[list[int], list[float], np.ndarray]:
  """One block's frames of the spec, their times, and its cell averages there
  ``(n_f, nr, nt)``, rows inside out.

  Raises:
    ValueError: A dataset carries no frame number, the quantities or species
      are not all written at the same frames, or their sum has more than one
      component.
  """
  stem = layout.block_name(block)
  by_frame: dict[int, list] = {}
  for q in quantity:
    for data in load_quantity(q,
                              ",".join(species) if species else None,
                              stem,
                              frame,
                              path=layout.path,
                              gamma_e=gamma_e,
                              gamma_i=gamma_i):
      if "frame" not in data.ctx:
        raise ValueError(f"{stem}: a file of {q} carries no frame number in "
                         "its metadata")
      by_frame.setdefault(int(data.ctx["frame"]), []).append(data)
  if len({len(v) for v in by_frame.values()}) > 1:
    raise ValueError(f"{stem}: {', '.join(quantity)} not written for every "
                     "species at the same frames")
  frames = sorted(by_frame)
  cells = []
  for k in frames:
    total = by_frame[k][0]
    for data in by_frame[k][1:]:
      total = total + data
    dg = require_dg_layout(total)
    if dg.num_fields != 1:
      raise ValueError(f"{', '.join(quantity)}: {dg.num_fields} components; "
                       "cell values need a scalar quantity")
    cells.append(
        layout.rows(block,
                    cell_means(total.values, dg.ndim, dg.block_size)[..., 0]))
  time = [by_frame[k][0].ctx.get("time", np.nan) for k in frames]
  return frames, time, np.array(cells)


def layout_cell_values(layout: MultibLayout,
                       quantity: list[str],
                       frame: FrameSpec = ":",
                       *,
                       species: list[str] | None = None,
                       blocks: Sequence[int] | None = None,
                       gamma_e: float | None = None,
                       gamma_i: float | None = None) -> CellValues:
  """:func:`cell_values` of a layout already built (not a CLI command: it
  takes a :class:`~.multib_layout.MultibLayout`).

  Raises:
    ValueError: The blocks hold different frames of the quantity, none of
      ``frame``, or as for :func:`cell_values`.
  """
  quantity = name_list(quantity, "quantity")
  species = None if species is None else name_list(species, "species")
  blocks = (list(range(layout.num_blocks))
            if blocks is None else [int(b) for b in blocks])
  if not blocks:
    raise ValueError("blocks needs at least one block")
  bad = [b for b in blocks if not 0 <= b < layout.num_blocks]
  if bad:
    raise ValueError(f"{layout.name} has blocks 0 to {layout.num_blocks - 1}, "
                     f"not {bad}")
  read = [
      _block_cells(layout, b, quantity, species, _frame_text(frame), gamma_e,
                   gamma_i) for b in blocks
  ]
  frames, time = read[0][:2]
  for b, (k, _, _) in zip(blocks, read):
    if k != frames:
      raise ValueError(f"{layout.name}: blocks {blocks[0]} and {b} hold "
                       f"different frames of {', '.join(quantity)}")
  if not frames:
    raise ValueError(f"{layout.name}: no frame of {frame!r} holds "
                     f"{', '.join(quantity)}")
  return CellValues(layout.name, layout.path, tuple(quantity),
                    None if species is None else tuple(species), tuple(blocks),
                    np.array(frames), np.array(time),
                    tuple(c for _, _, c in read))


def cell_values(quantity: list[str],
                name: str,
                frame: Annotated[FrameSpec, CliType(str)] = ":",
                *,
                species: list[str] | None = None,
                path: str = "./",
                blocks: list[int] | None = None,
                geqdsk: str | None = None,
                gamma_e: float | None = None,
                gamma_i: float | None = None) -> CellValues:
  """Cell averages of a quantity in each block of a tokamak run, at its
  frames, rows inside out (from the core or private flux side outward).

  The quantity is the sum of the listed registry quantities over the listed
  species, each block's datasets summed in their modal form at each frame,
  reduced to cell averages. The layout is built once for all the frames.

  Args:
    quantity: Registry quantities to sum (``quantity_catalog()`` lists
      them): ``['M0']`` the density, ``['temp']`` the temperature,
      ``['field']`` the potential, ``['qpar', 'qperp']`` the parallel flux of
      kinetic energy.
    name: The run's file prefix (without ``_b<k>``).
    frame: Frame number, comma list, or ``'start:stop[:step]'`` range
      (``':'``: all); frames without the quantity's files are skipped.
    species: Species to sum over; ``None`` for species-independent
      quantities. A quantity combining several species (``mach_hot_i``) reads
      the whole list once.
    path: Directory holding the run's output.
    blocks: The blocks to read; ``None``: all.
    geqdsk: The run's equilibrium file, for its layout (the psi direction
      where no closed chain fixes it; see :func:`~.multib_layout.
      multib_layout`).
    gamma_e: The electrons' adiabatic index in the sound-speed quantities
      (``c_s_hot_i``, ``mach_hot_i``); ``None``: their default (1).
    gamma_i: The ions' adiabatic index there; ``None``: their default (3).

  Returns:
    The :class:`CellValues`.

  Raises:
    ValueError: The blocks hold different frames of the quantity, or none of
      ``frame``; the species are not all written at the same frames; the
      quantity has more than one component; or a block does not exist.
  """
  quantity = name_list(quantity, "quantity")
  layout = multib_layout(name, path=path, geqdsk=geqdsk)
  return layout_cell_values(layout,
                            quantity,
                            frame,
                            species=species,
                            blocks=blocks,
                            gamma_e=gamma_e,
                            gamma_i=gamma_i)


__all__ = ["CellValues", "cell_values", "layout_cell_values", "name_list"]
