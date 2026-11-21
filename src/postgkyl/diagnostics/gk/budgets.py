"""What crosses each target plate of a gyrokinetic tokamak run, and how well
its particle and energy budgets close between output frames.

Built on the balance traces (:func:`postgkyl.diagnostics.gk.balance`) and
the run's block layout (:func:`postgkyl.diagnostics.gk.multib_layout.
multib_layout`). Every integrated diagnostic is per radian of toroidal
angle; :func:`device_factor` turns it into the whole device.

- A target plate is the theta face of its leg block and of the private-flux
  block under it: its power and particle flux are the third and first
  components of ``<species>_bflux_<ylower|yupper>_integrated_
  HamiltonianMoments`` of both blocks, summed over the species.
- Over each interval between output frames a budget sums what the sources
  put in, what the boundaries took out, and what the content gained; its
  imbalance is the part none of them accounts for, over the largest of the
  three. Over an interval rather than at each sample: in some runs the rate
  of change swings from sample to sample while what it integrates to does
  not.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from postgkyl import io

from . import utils
from .balance import SourceMoments, balance
from .distf import resolve_frames
from .multib_layout import block_prefixes, multib_layout


def device_factor(half_domain: bool) -> float:
  """Factor from the integrated diagnostics (per radian of toroidal angle)
  to the whole device: ``2 pi``, times 2 for a half domain (its mirror image
  across the reflecting midplane)."""
  return 2 * np.pi * (2 if half_domain else 1)


# ------------------------------------------------------------ target plates
@dataclass(frozen=True, eq=False)
class PlateFlux:
  """What crosses one target plate, summed over the species.

  Attributes:
    name: The target's name (``'lower outer target'``, ...).
    blocks: The blocks whose theta face makes the plate: the leg, then the
      private-flux block under it.
    face: The boundary-flux face of the plate, ``'ylower'`` or ``'yupper'``.
    power: The Hamiltonian-moment flux through it, as the boundary-flux
      diagnostics write it (W per radian; W with ``full_torus``).
    particles: The particle flux through it, likewise (s^-1 per radian;
      s^-1 with ``full_torus``).
  """

  name: str
  blocks: tuple[int, ...]
  face: str
  power: np.ndarray
  particles: np.ndarray


@dataclass(frozen=True, eq=False)
class TargetFluxes:
  """The flux through every target plate of a run.

  Attributes:
    time: The samples' times (s), after t=0.
    plates: One :class:`PlateFlux` per target that wrote a boundary flux, in
      the layout's target order.
    species: The species summed over.
    full_torus: The fluxes are over the whole toroidal ring of each plate
      (``2 pi`` radians), else per radian.
  """

  time: np.ndarray
  plates: tuple[PlateFlux, ...]
  species: tuple[str, ...]
  full_torus: bool

  def __str__(self) -> str:
    unit = "" if self.full_torus else "/rad"
    if not self.time.size:
      return "no boundary-flux sample after t=0"
    lines = [
        f"target fluxes at t = {self.time[-1]:.6g} s "
        f"({', '.join(self.species)}):"
    ]
    for p in self.plates:
      lines.append(f"  {p.name} (blocks {', '.join(map(str, p.blocks))}, "
                   f"{p.face}): power {p.power[-1]:.4g} W{unit}, particles "
                   f"{p.particles[-1]:.4g} s^-1{unit}")
    return "\n".join(lines)


def target_fluxes(name: str,
                  species: list[str],
                  *,
                  path: str = "./",
                  geqdsk: str | None = None,
                  full_torus: bool = False) -> TargetFluxes:
  """The power and particle flux through each target plate of a run.

  A plate is the theta face (``ylower`` at a leg's theta-lower end,
  ``yupper`` at its upper end) of the leg block and of the private-flux
  block under it. Reads, per species and plate block,
  ``<block prefix>-<species>_bflux_<face>_integrated_HamiltonianMoments.gkyl``:
  the particle flux is its first component, the power its third (the
  Hamiltonian). A block that wrote no such file adds nothing; a target none
  of whose blocks wrote one is left out. Where the files' lengths differ, the
  shortest sets the samples. Like :func:`balance`, the first sample (t=0,
  before a step) is skipped.

  Args:
    name: The run's file prefix (without ``_b<k>``).
    species: The species to sum over.
    path: Directory holding the run's output.
    geqdsk: The run's equilibrium file, for the layout (see
      :func:`multib_layout`).
    full_torus: Give each plate's flux over its whole toroidal ring
      (``2 pi`` radians) instead of per radian.

  Returns:
    The :class:`TargetFluxes`.
  """
  layout = multib_layout(name, path=path, geqdsk=geqdsk)
  scale = 2 * np.pi if full_torus else 1.0
  found = []  # (target, blocks, face, [(time, values) per file])
  for t in layout.targets:
    face = "ylower" if t.theta_end == 0 else "yupper"
    blocks = (t.leg, ) + (() if t.pf is None else (t.pf, ))
    files = [
        layout.file_name(b, f"{s}_bflux_{face}_integrated_HamiltonianMoments")
        for s in species for b in blocks
    ]
    traces = [utils.read_time_trace(f) for f in files if os.path.exists(f)]
    if traces:
      found.append((t.name, blocks, face, traces))
  n = min((time.size for *_, traces in found for time, _ in traces), default=1)
  time = found[0][3][0][0][1:n] if found else np.zeros(0)
  plates = []
  for target, blocks, face, traces in found:
    total = scale * sum(values[1:n] for _, values in traces)
    plates.append(PlateFlux(target, blocks, face, total[:, 2], total[:, 0]))
  return TargetFluxes(time=time,
                      plates=tuple(plates),
                      species=tuple(species),
                      full_torus=full_torus)


# ------------------------------------------------- budgets over intervals
@dataclass(frozen=True, eq=False)
class BalanceIntervals:
  """A particle or energy budget over each interval between output frames.

  Over the ``k``-th interval, ``source - outflow - gain = residual``, and
  ``imbalance = residual / max(|source|, |outflow|, |gain|)``: 0 where the
  budget closes, +-1 where a whole term is unaccounted for.

  Attributes:
    quantity: ``'particle'`` or ``'energy'``.
    edges: The intervals' edges (s), ``n + 1`` of them.
    source: What the sources (and BGK sources) put in over each interval.
    outflow: What the boundaries took out.
    gain: What the content gained (for the energy, net of the field terms).
    residual: The balance residual integrated over the interval.
    imbalance: ``residual`` over the largest of ``|source|``, ``|outflow|``
      and ``|gain|``; ``nan`` where outflow and gain are both 0 (nothing
      recorded yet) or all three are.
    whole_device: The amounts are for the whole device (see
      :func:`device_factor`), else per radian.
  """

  quantity: str
  edges: np.ndarray
  source: np.ndarray
  outflow: np.ndarray
  gain: np.ndarray
  residual: np.ndarray
  imbalance: np.ndarray
  whole_device: bool = False

  @property
  def midpoints(self) -> np.ndarray:
    """Each interval's middle time (s)."""
    return 0.5 * (self.edges[1:] + self.edges[:-1])

  def __str__(self) -> str:
    if not self.imbalance.size:
      return f"{self.quantity} budget: fewer than two interval edges"
    ok = self.imbalance[np.isfinite(self.imbalance)]
    lines = [
        f"{self.quantity} budget over {self.imbalance.size} intervals: "
        "imbalance median " +
        (f"{np.median(ok):.3e}, largest |.| {np.abs(ok).max():.3e}"
         if ok.size else "nan")
    ]
    lines.append(f"{'t_start (s)':>12} {'t_end (s)':>12} {'source':>11} "
                 f"{'outflow':>11} {'gain':>11} {'imbalance':>11}")
    for k in range(self.imbalance.size):
      lines.append(f"{self.edges[k]:12.5e} {self.edges[k + 1]:12.5e} "
                   f"{self.source[k]:11.4e} {self.outflow[k]:11.4e} "
                   f"{self.gain[k]:11.4e} {self.imbalance[k]:11.4e}")
    return "\n".join(lines)


def interval_integrals(time: np.ndarray, rate: np.ndarray,
                       edges: np.ndarray) -> np.ndarray:
  """The integral of a sampled rate over each interval between ``edges``.

  The trapezoid rule over the samples; at an edge between two samples, the
  cumulative integral is interpolated linearly. Edges must lie within the
  samples' span.
  """
  cumulative = np.concatenate([[0.0],
                               np.cumsum(0.5 * (rate[1:] + rate[:-1]) *
                                         np.diff(time))])
  return np.diff(np.interp(edges, time, cumulative))


def imbalance(source: np.ndarray, outflow: np.ndarray,
              gain: np.ndarray) -> np.ndarray:
  """``(source - outflow - gain)`` over the largest of their magnitudes.

  ``nan`` where outflow and gain are both exactly 0 (nothing recorded yet)
  or every term is 0.
  """
  big = np.maximum(np.maximum(np.abs(source), np.abs(outflow)), np.abs(gain))
  with np.errstate(divide="ignore", invalid="ignore"):
    return np.where(((outflow == 0) & (gain == 0)) | (big == 0), np.nan,
                    (source - outflow - gain) / big)


def interval_budget(time: np.ndarray,
                    source: np.ndarray,
                    outflow: np.ndarray,
                    residual: np.ndarray,
                    frame_times: np.ndarray,
                    quantity: str = "particle") -> BalanceIntervals:
  """A budget's rate traces integrated over each interval between frames.

  The content's gain is what the source put in less what the boundaries
  took out less the residual. Leading samples where outflow and gain are
  both 0 hold nothing yet and are skipped; the frame times are clipped to
  the remaining samples' span, so the first interval starts at the first
  sample that holds something.

  Args:
    time: The samples' times (s).
    source: The source rate at each sample.
    outflow: The boundary outflow rate.
    residual: The balance residual ``source - outflow - gain`` at each
      sample.
    frame_times: The output frames' times (s): the intervals' edges.
    quantity: What the budget is of, for the record.

  Returns:
    The :class:`BalanceIntervals` (empty where fewer than two edges fall
    within the samples' span).
  """
  time, source, outflow, residual = (np.asarray(a, float)
                                     for a in (time, source, outflow, residual))
  gain = source - outflow - residual
  held = np.flatnonzero((outflow != 0) | (gain != 0))
  first = int(held[0]) if held.size else time.size
  empty = np.zeros(0)
  if time.size - first < 2:
    return BalanceIntervals(quantity, empty, empty, empty, empty, empty, empty)
  t = time[first:]
  edges = np.unique(np.clip(np.asarray(frame_times, float), t[0], t[-1]))
  if edges.size < 2:
    return BalanceIntervals(quantity, empty, empty, empty, empty, empty, empty)
  src, out, res = (interval_integrals(t, a[first:], edges)
                   for a in (source, outflow, residual))
  dot = src - out - res
  return BalanceIntervals(quantity, edges, src, out, dot, res,
                          imbalance(src, out, dot))


def frame_times(name: str,
                species: str,
                *,
                path: str = "./") -> tuple[np.ndarray, np.ndarray]:
  """The frame numbers and times of a run's output frames.

  The frames of ``<block 0 prefix>-<species>_M0_<frame>.gkyl`` and the time
  each file records.

  Args:
    name: The run's file prefix (without ``_b<k>``).
    species: The species whose ``M0`` files mark the frames.
    path: Directory holding the run's output.

  Returns:
    ``(frames, times)``: the frame numbers and their times (s).
  """
  path = path.rstrip("/") or "."
  prefixes = block_prefixes(name, path)
  base = os.path.join(path, name)
  multib = prefixes[0] != base
  frames = resolve_frames(":",
                          name=base,
                          species=species,
                          suffix="M0",
                          block_idx=0 if multib else None)
  times = [
      float(
          io.read_metadata(f"{prefixes[0]}-{species}_M0_{k}.gkyl").get(
              "time", np.nan)) for k in frames
  ]
  return np.asarray(frames, int), np.asarray(times)


def balance_intervals(name: str,
                      species: list[str],
                      *,
                      quantity: Literal["particle", "energy"] = "particle",
                      path: str = "./",
                      edges: list[float] | None = None,
                      source_moments: SourceMoments = "HamiltonianMoments",
                      whole_device: bool = False) -> BalanceIntervals:
  """A run's particle or energy budget over each interval between frames.

  The rate traces are :func:`balance`'s (summed over ``species`` and every
  block, after t=0): the source is ``src + src_bgk``, the outflow
  ``bflux_tot``, the residual ``mom_err``, and the content's gain what is
  left (``fdot``, net of the field terms for the energy). Each is integrated
  over each interval between the output frames (:func:`interval_budget`).

  Args:
    name: The run's file prefix (without ``_b<k>``).
    species: The species to sum over.
    quantity: The budget's conserved quantity.
    path: Directory holding the run's output.
    edges: The intervals' edges (s); default: the times of the output frames
      of ``species[0]``'s ``M0``.
    source_moments: What the run's older ``*_source_integrated_moms`` files
      hold (see :func:`balance`).
    whole_device: Give the amounts for the whole device (2 pi, times 2 for a
      half domain) instead of per radian.

  Returns:
    The :class:`BalanceIntervals`.
  """
  path = path.rstrip("/") or "."
  multib = block_prefixes(name, path)[0] != os.path.join(path, name)
  fig, traces = balance(name,
                        species,
                        quantity=quantity,
                        path=path,
                        source_moments=source_moments,
                        multib="-1" if multib else "-10")
  plt.close(fig)
  zero = np.zeros_like(traces["fdot"])
  source = sum(zero if traces[k] is None else traces[k]
               for k in ("src", "src_bgk"))
  outflow = zero if traces["bflux_tot"] is None else traces["bflux_tot"]
  if edges is None:
    edges = frame_times(name, species[0], path=path)[1]
  out = interval_budget(traces["time"], source, outflow, traces["mom_err"],
                        np.asarray(edges, float), quantity)
  if not whole_device:
    return out
  f = device_factor(multib_layout(name, path=path).half_domain)
  return BalanceIntervals(out.quantity,
                          out.edges,
                          f * out.source,
                          f * out.outflow,
                          f * out.gain,
                          f * out.residual,
                          out.imbalance,
                          whole_device=True)
