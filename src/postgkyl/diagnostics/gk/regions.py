"""Statistics of a gyrokinetic quantity over a region of a tokamak run.

A region is the core, the scrape-off layer (``'sol'``), the private flux
(``'pf'``), or ``'all'`` of the run, made of whole blocks, or of a block's
rows below and above its LCFS where it holds closed and open field lines
(:meth:`MultibLayout.region_rows`). Per frame:

- :func:`region_average`: the volume average ``int q J dx / int J dx``,
  with ``J`` the configuration-space Jacobian (``geo_int_jacobgeo``); each
  cell's ``int q J`` is the cell mean of the weak product ``q J`` (exact for
  the two DG expansions) times the cell's logical size.
- :func:`region_median_change`: the median over the region's cells of each
  cell mean's change relative to a reference frame.
- :func:`nonpositive_cells`: how many of the region's cell means are not
  positive (a density that went negative).

The quantity is any of :func:`load_quantity`'s, computed per block with DG
arithmetic. Cells whose value is not finite are left out of every statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from postgkyl import dg
from postgkyl.diagnostics import discovery
from postgkyl.gdata import GData

from .load_quantity import load_quantity
from .multib_layout import MultibLayout, block_prefixes, multib_layout

Region = Literal["core", "sol", "pf", "all"]


@dataclass(frozen=True, eq=False)
class RegionSeries:
  """One statistic of a quantity over a region, at each frame.

  Attributes:
    statistic: What ``values`` holds: ``'volume average'``, ``'median
      relative change'`` or ``'non-positive cells'``.
    quantity: The quantity (a :func:`load_quantity` name).
    species: Its species, or ``None``.
    region: The region.
    frames: The frame numbers.
    time: Each frame's time (s).
    values: The statistic at each frame (``nan`` where the region holds no
      cell with a finite value).
  """

  statistic: str
  quantity: str
  species: str | None
  region: str
  frames: np.ndarray
  time: np.ndarray
  values: np.ndarray

  def __str__(self) -> str:
    what = self.quantity + (f" ({self.species})" if self.species else "")
    lines = [
        f"{self.statistic} of {what} over region '{self.region}':",
        f"{'frame':>6} {'time (s)':>12} {'value':>13}"
    ]
    for k, t, v in zip(self.frames, self.time, self.values):
      lines.append(f"{int(k):6d} {t:12.5e} {v:13.6g}")
    return "\n".join(lines)


def _cell_means(data: GData) -> np.ndarray:
  """The cell means ``(nr, nt)`` of a scalar modal field, file rows.

  Raises:
    ValueError: if the field has more than one component.
  """
  values = np.asarray(data.get_values())
  ndim = values.ndim - 1
  means = dg.modal.cell_means(
      values, ndim,
      dg.num_basis(ndim, data.ctx["poly_order"], data.ctx["basis_type"]))
  if means.shape[-1] != 1:
    raise ValueError("Region statistics take a scalar quantity; this one has "
                     f"{means.shape[-1]} components")
  return means[..., 0]


def _block_frames(quantity: str,
                  species: str | None,
                  layout: MultibLayout,
                  frame: str,
                  gamma_e: float | None = None,
                  gamma_i: float | None = None) -> list[list[GData]]:
  """The quantity per block (outer) and frame (inner).

  Raises:
    ValueError: if the blocks' frames differ.
  """
  out = [
      load_quantity(quantity,
                    species,
                    layout.block_name(b),
                    str(frame),
                    path=layout.path,
                    gamma_e=gamma_e,
                    gamma_i=gamma_i) for b in range(layout.num_blocks)
  ]
  frames = [[d.ctx.get("frame") for d in block] for block in out]
  if any(f != frames[0] for f in frames[1:]):
    raise ValueError("The blocks hold different frames of " + quantity)
  return out


def _series(statistic: str, quantity: str, species: str | None, region: str,
            data: list[list[GData]], values: np.ndarray) -> RegionSeries:
  first = data[0]
  return RegionSeries(
      statistic, quantity, species, region,
      np.array([d.ctx.get("frame", -1) for d in first], int),
      np.array([d.ctx.get("time", np.nan) for d in first], float),
      np.asarray(values, float))


def _in_region(layout: MultibLayout, block: int, region: Region,
               cells: np.ndarray) -> np.ndarray:
  """A block's cell values in the region, inside-out rows, flattened."""
  return layout.rows(block, cells)[layout.region_rows(block, region)].ravel()


def region_average(quantity: str,
                   species: str | None,
                   name: str,
                   frame: str = ":",
                   *,
                   path: str = "./",
                   region: Region = "all",
                   geqdsk: str | None = None,
                   gamma_e: float | None = None,
                   gamma_i: float | None = None) -> RegionSeries:
  """The volume average of a quantity over a region of a run, per frame.

  ``int_region q J dx / int_region J dx``, ``J`` each block's
  ``geo_int_jacobgeo``: a cell contributes the cell mean of the weak product
  ``q J`` and of ``J`` times its logical size (cells where either is not
  finite are left out).

  Args:
    quantity: The quantity (see :func:`load_quantity`).
    species: Its species, or ``None`` for a species-independent quantity.
    name: The run's file prefix (without ``_b<k>``).
    frame: Frame number, comma list, or ``'start:stop[:step]'`` range
      (``':'``: all).
    path: Directory holding the run's output.
    region: ``'core'``, ``'sol'``, ``'pf'`` or ``'all'``.
    geqdsk: The run's equilibrium file, for the layout (see
      :func:`multib_layout`).

    gamma_e: The electrons' adiabatic index in the sound-speed quantities
      (``c_s_hot_i``, ``mach_hot_i``); ``None``: their default (1).
    gamma_i: The ions' adiabatic index there; ``None``: their default (3).
  Returns:
    The :class:`RegionSeries` of volume averages.
  """
  layout = multib_layout(name, path=path, geqdsk=geqdsk)
  data = _block_frames(quantity, species, layout, frame, gamma_e, gamma_i)
  num = np.zeros(len(data[0]))
  den = np.zeros(len(data[0]))
  for b, block in enumerate(data):
    jacobian = GData(layout.file_name(b, "geo_int_jacobgeo"))
    widths = [np.diff(edges) for edges in jacobian.get_grid()]
    size = np.multiply.outer(*widths)
    volume = _in_region(layout, b, region, _cell_means(jacobian) * size)
    for k, q in enumerate(block):
      content = _in_region(layout, b, region, _cell_means(q * jacobian) * size)
      ok = np.isfinite(content) & np.isfinite(volume)
      num[k] += content[ok].sum()
      den[k] += volume[ok].sum()
  with np.errstate(divide="ignore", invalid="ignore"):
    values = np.where(den != 0, num / den, np.nan)
  return _series("volume average", quantity, species, region, data, values)


def region_median_change(quantity: str,
                         species: str | None,
                         name: str,
                         frame: str = ":",
                         *,
                         path: str = "./",
                         region: Region = "all",
                         reference: int | None = None,
                         geqdsk: str | None = None,
                         gamma_e: float | None = None,
                         gamma_i: float | None = None) -> RegionSeries:
  """The median over a region's cells of each cell mean's relative change.

  Per frame, the median of ``q / q_ref - 1`` over the region's cells, with
  ``q`` and ``q_ref`` the cell means at the frame and at the reference frame
  (cells where it is not finite are left out).

  Args:
    quantity: The quantity (see :func:`load_quantity`).
    species: Its species, or ``None`` for a species-independent quantity.
    name: The run's file prefix (without ``_b<k>``).
    frame: Frame number, comma list, or ``'start:stop[:step]'`` range
      (``':'``: all).
    path: Directory holding the run's output.
    region: ``'core'``, ``'sol'``, ``'pf'`` or ``'all'``.
    reference: The reference frame; default: the run's first frame (of
      ``<block 0 prefix>-field_<frame>.gkyl``).
    geqdsk: The run's equilibrium file, for the layout (see
      :func:`multib_layout`).

    gamma_e: The electrons' adiabatic index in the sound-speed quantities
      (``c_s_hot_i``, ``mach_hot_i``); ``None``: their default (1).
    gamma_i: The ions' adiabatic index there; ``None``: their default (3).
  Returns:
    The :class:`RegionSeries` of median relative changes.
  """
  layout = multib_layout(name, path=path, geqdsk=geqdsk)
  if reference is None:
    first = block_prefixes(name, layout.path)[0]
    reference = min(discovery.available_frames(f"{first}-field_"))
  data = _block_frames(quantity, species, layout, frame, gamma_e, gamma_i)
  ref = [
      _in_region(layout, b, region, _cell_means(block[0]))
      for b, block in enumerate(
          _block_frames(quantity, species, layout, str(reference), gamma_e,
                        gamma_i))
  ]
  values = []
  for k in range(len(data[0])):
    with np.errstate(divide="ignore", invalid="ignore"):
      change = np.concatenate([np.zeros(0)] + [
          _in_region(layout, b, region, _cell_means(block[k])) / ref[b] - 1
          for b, block in enumerate(data)
      ])
    change = change[np.isfinite(change)]
    values.append(np.median(change) if change.size else np.nan)
  return _series("median relative change", quantity, species, region, data,
                 values)


def nonpositive_cells(quantity: str,
                      species: str | None,
                      name: str,
                      frame: str = ":",
                      *,
                      path: str = "./",
                      region: Region = "all",
                      geqdsk: str | None = None) -> RegionSeries:
  """How many of a region's cells have a cell mean that is not positive.

  For the density (``M0``), the cells where it went negative or to zero.

  Args:
    quantity: The quantity (see :func:`load_quantity`).
    species: Its species, or ``None`` for a species-independent quantity.
    name: The run's file prefix (without ``_b<k>``).
    frame: Frame number, comma list, or ``'start:stop[:step]'`` range
      (``':'``: all).
    path: Directory holding the run's output.
    region: ``'core'``, ``'sol'``, ``'pf'`` or ``'all'``.
    geqdsk: The run's equilibrium file, for the layout (see
      :func:`multib_layout`).

  Returns:
    The :class:`RegionSeries` of counts.
  """
  layout = multib_layout(name, path=path, geqdsk=geqdsk)
  data = _block_frames(quantity, species, layout, frame)
  counts = [
      sum(
          int(
              np.count_nonzero(
                  _in_region(layout, b, region, _cell_means(block[k])) <= 0))
          for b, block in enumerate(data)) for k in range(len(data[0]))
  ]
  return _series("non-positive cells", quantity, species, region, data, counts)
