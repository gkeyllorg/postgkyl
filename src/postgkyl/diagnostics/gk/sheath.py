"""The sheath entrance at a gyrokinetic tokamak run's target plates.

At the frames with the sheath boundary condition's files, at each target's
plate face (the theta face of its leg block's end cell
column), per open SOL cell row:

- e (phi_se - phi_wall) / T_e, the potential drop the sheath boundary
  condition reflects electrons by: its cutoff energy m_e v_cut^2 / 2 (the
  electrons' ``<species>_bc_z{lower,upper}_vcutsq``, averaged over its cell
  and over mu, which it does not depend on) over T_e;
- the textbook floating-sheath drop -1/2 ln[2 pi (m_e / m_i) (1 + T_i / T_e)]
  (Stangeby, *The Plasma Boundary of Magnetic Fusion Devices*, ch. 2);
- the ion Mach number u_par,i / c_s toward the plate, c_s = sqrt((T_e + T_i)
  / m_i);
- T_e and T_i.

The moments M0, M1, M2 enter as their expansions averaged over the plate face
(where the boundary condition acts), not as cell averages; T and u_par are
formed from those face averages pointwise, T = m (M2 - M1^2 / M0) / (3 M0),
which for point values equals the registry's temp = (T_par + 2 T_perp) / 3.

:func:`sheath_entrance` takes a frame specification and builds the run's
layout once for all its frames.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import os
from typing import Annotated

import numpy as np

from postgkyl import dg, io
from postgkyl.cli_spec import CliType
from postgkyl.diagnostics import discovery
from postgkyl.gdata import GData

from .distf import FrameSpec, resolve_frames
from .flux_coords import layout_flux_surfaces
from .multib_layout import (MultibLayout, Target, block_prefixes, multib_layout,
                            psi_cells)

_FACE = {0: "lower", -1: "upper"}  # a target's theta end -> its face
_MOMENTS = ("M0", "M1", "M2")


# ------------------------------------------------------------- pure cores
def face_weights(basis_type: str, ndim: int, poly_order: int, direction: int,
                 side: int) -> np.ndarray:
  """``(num_basis,)`` weights: a cell's coefficients times them is the mean
  of its expansion over one face.

  The expansion on the face is a polynomial of degree ``poly_order`` in each
  tangential direction, so ``poly_order + 1`` Gauss-Legendre points per
  direction average it exactly.

  Args:
    basis_type: The field's basis.
    ndim: Its dimension.
    poly_order: Its polynomial order.
    direction: The face's normal direction.
    side: ``-1`` for the lower face, ``+1`` for the upper.
  """
  x, w = np.polynomial.legendre.leggauss(poly_order + 1)
  tangential = ndim - 1
  points = np.insert(np.array(list(itertools.product(x, repeat=tangential))),
                     direction,
                     float(side),
                     axis=1)
  weights = np.prod(list(itertools.product(w, repeat=tangential)),
                    axis=1) / 2**tangential
  return weights @ dg.basis_matrix(basis_type, ndim, poly_order, points)


def temperature(m0: np.ndarray, m1: np.ndarray, m2: np.ndarray,
                mass: float) -> np.ndarray:
  """T (J) from point values of gyrokinetic moments: m (M2 - M1^2 / M0) /
  (3 M0), M2 holding v_par^2 + 2 mu B / m (three degrees of freedom)."""
  return mass * (m2 - m1**2 / m0) / (3 * m0)


def floating_drop(te: np.ndarray, ti: np.ndarray, elc_mass: float,
                  ion_mass: float) -> np.ndarray:
  """The floating sheath's e (phi_se - phi_wall) / T_e:
  -1/2 ln[2 pi (m_e / m_i) (1 + T_i / T_e)]."""
  return -0.5 * np.log(2 * np.pi * elc_mass / ion_mass * (1 + ti / te))


def sheath_terms(
    vcut_sq: np.ndarray, elc: tuple[np.ndarray, np.ndarray, np.ndarray],
    ion: tuple[np.ndarray, np.ndarray, np.ndarray], *, elc_mass: float,
    ion_mass: float, elementary_charge: float, toward: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """The sheath entrance from face values.

  Args:
    vcut_sq: The electrons' cutoff velocity squared (m^2/s^2).
    elc: The electrons' face (M0, M1, M2).
    ion: The ions' face (M0, M1, M2).
    elc_mass: The electron mass (kg).
    ion_mass: The ion mass (kg).
    elementary_charge: e (C), for temperatures in eV.
    toward: ``+1`` where the plate lies at the face's upper theta end, ``-1``
      at its lower: the sign of a flow toward it.

  Returns:
    ``(drop, floating, mach, te, ti)``: e (phi_se - phi_wall) / T_e =
    (m_e v_cut^2 / 2) / T_e, the floating sheath's (:func:`floating_drop`),
    u_par,i / c_s toward the plate, and T_e and T_i in eV; ``nan`` or
    ``inf`` where a moment vanishes.
  """
  with np.errstate(divide="ignore", invalid="ignore"):
    te = temperature(*elc, elc_mass)
    ti = temperature(*ion, ion_mass)
    drop = 0.5 * elc_mass * vcut_sq / te
    floating = floating_drop(te, ti, elc_mass, ion_mass)
    mach = toward * (ion[1] / ion[0]) / np.sqrt((te + ti) / ion_mass)
  return drop, floating, mach, te / elementary_charge, ti / elementary_charge


# ------------------------------------------------------------- the records
@dataclass(frozen=True, eq=False)
class SheathProfile:
  """The sheath entrance along one target plate at a run's frames, per open
  SOL cell row.

  The per-frame fields are ``(n_f, n_rows)``, one row per frame of the
  :class:`SheathEntrance`; nan or inf at a frame where a moment vanishes.

  Attributes:
    target: The target's name.
    block: Its leg block.
    rows: The leg's cell rows (inside out) the values are for: its open rows
      where the sheath files give a finite drop at some frame.
    psi_n: psi_N of those rows, or ``None`` without a geqdsk.
    drop: e (phi_se - phi_wall) / T_e from the boundary condition's cutoff.
    floating: The textbook floating-sheath drop for the same T_e, T_i.
    mach: The ion Mach number u_par,i / c_s toward the plate.
    te: T_e (eV) at the plate face.
    ti: T_i (eV) at the plate face.
  """

  target: str
  block: int
  rows: np.ndarray
  psi_n: np.ndarray | None
  drop: np.ndarray
  floating: np.ndarray
  mach: np.ndarray
  te: np.ndarray
  ti: np.ndarray

  def __post_init__(self):
    n = len(self.rows)
    shapes = {
        np.shape(v)
        for v in (self.drop, self.floating, self.mach, self.te, self.ti)
    }
    shape = shapes.pop() if len(shapes) == 1 else ()
    if len(shape) != 2 or shape[1] != n or (self.psi_n is not None
                                            and len(self.psi_n) != n):
      raise ValueError(f"SheathProfile {self.target}: drop, floating, mach, "
                       f"te and ti must share one shape (frames, {n} rows) "
                       f"and psi_n have {n} rows")


def _span(v: np.ndarray) -> str:
  """The range of the finite values."""
  v = v[np.isfinite(v)]
  return f"{np.min(v):.4g}..{np.max(v):.4g}" if len(v) else "-"


@dataclass(frozen=True, eq=False)
class SheathEntrance:
  """The sheath entrance at every target plate of a run at its frames.

  Attributes:
    name: The run's file prefix.
    path: Its directory.
    frames: The frames ``(n_f,)``.
    time: Each frame's time (s) ``(n_f,)``.
    electron: The electron species.
    ion: The ion species.
    targets: Each target whose leg wrote the electrons' sheath file at the
      frames, in the layout's target order.
  """

  name: str
  path: str
  frames: np.ndarray
  time: np.ndarray
  electron: str
  ion: str
  targets: tuple[SheathProfile, ...]

  def __post_init__(self):
    n = len(self.frames)
    if len(self.time) != n or any(len(p.drop) != n for p in self.targets):
      raise ValueError(f"SheathEntrance: time and each target's values must "
                       f"cover the {n} frames")

  def __str__(self) -> str:
    n = len(self.frames)
    lines = [
        f"{self.name}: sheath entrance ({self.electron}, {self.ion}), " +
        (f"frame {self.frames[0]}"
         if n == 1 else f"{n} frames ({self.frames[0]} to {self.frames[-1]})")
    ]
    for p in self.targets:
      lines.append(f"  {p.target} (block {p.block}, {len(p.rows)} rows" +
                   ("" if p.psi_n is None else f", psi_N {_span(p.psi_n)}") +
                   "):")
      for i, (k, t) in enumerate(zip(self.frames, self.time)):
        lines.append(
            f"    frame {k} (t = {t:.5e} s): e dphi/T_e {_span(p.drop[i])} "
            f"(floating {_span(p.floating[i])}), Mach {_span(p.mach[i])}, "
            f"T_e {_span(p.te[i])} eV, T_i {_span(p.ti[i])} eV")
    return "\n".join(lines)


# ---------------------------------------------------------------- reading
def _modal(file_name: str) -> tuple[np.ndarray, dict]:
  """A modal file's coefficients (unsqueezed) and its metadata."""
  data = GData(file_name)
  return np.asarray(data.get_values()), data.ctx


def _face_values(file_name: str, theta_end: int,
                 side: int) -> tuple[np.ndarray, dict]:
  """A 2D modal field's mean over the theta face ``side`` of its cell column
  ``theta_end`` (one value per psi cell), and its metadata."""
  coeffs, ctx = _modal(file_name)
  w = face_weights(ctx["basis_type"], coeffs.ndim - 1, ctx["poly_order"], 1,
                   side)
  return coeffs[:, theta_end] @ w, ctx


def sheath_frames(name: str,
                  *,
                  path: str = "./",
                  electron: str = "elc") -> tuple[int, ...]:
  """The frames at which a run wrote its sheath boundary condition's files
  (the electrons' ``_bc_z{lower,upper}_vcutsq``, with its phase-space
  output), from its first target's leg block.

  Args:
    name: The run's file prefix (without ``_b<k>``).
    path: Directory holding the run's output.
    electron: The electron species.
  """
  layout = multib_layout(name, path=path)
  if not layout.targets:
    return ()
  t = layout.targets[0]
  prefix = block_prefixes(name, path)[t.leg]
  return tuple(
      sorted(
          discovery.available_frames(
              f"{prefix}-{electron}_bc_z{_FACE[t.theta_end]}_vcutsq_")))


def _vcut_file(layout: MultibLayout, target: Target, electron: str,
               frame: int) -> str:
  """A target's sheath file: its leg's electron cutoff at the plate face."""
  return layout.file_name(target.leg,
                          f"{electron}_bc_z{_FACE[target.theta_end]}_vcutsq",
                          frame)


def _plate_terms(layout: MultibLayout, target: Target, frame: int,
                 electron: str, ion: str,
                 meta: dict[str, dict]) -> tuple[np.ndarray, ...]:
  """:func:`sheath_terms` at one frame over a target leg's open rows, inside
  out (nan where its sheath file does not reach); ``meta`` holds each
  species' mass and charge."""
  b, side = target.leg, -1 if target.theta_end == 0 else 1
  vcut, vctx = _modal(_vcut_file(layout, target, electron, frame))
  nb = dg.num_basis(vcut.ndim - 1, vctx["poly_order"], vctx["basis_type"])
  vcut_sq = dg.modal.cell_means(vcut, vcut.ndim - 1, nb)[..., 0].mean(axis=1)
  face = {
      (s, m):
      _face_values(layout.file_name(b, f"{s}_{m}", frame), target.theta_end,
                   side)
      for s in (electron, ion)
      for m in _MOMENTS
  }
  r0 = psi_cells(face[electron, "M0"][1], vctx).start
  covered = slice(r0, r0 + len(vcut_sq))
  values = sheath_terms(vcut_sq,
                        tuple(face[electron, m][0][covered] for m in _MOMENTS),
                        tuple(face[ion, m][0][covered] for m in _MOMENTS),
                        elc_mass=meta[electron]["mass"],
                        ion_mass=meta[ion]["mass"],
                        elementary_charge=-meta[electron]["charge"],
                        toward=side)
  nr = layout.nodes[b].shape[0] - 1
  rows = []
  for v in values:
    full = np.full(nr, np.nan)
    full[covered] = v
    rows.append(layout.rows(b, full)[target.open_row:])
  return tuple(rows)


def sheath_entrance(name: str,
                    frame: Annotated[FrameSpec, CliType(str)] = ":",
                    *,
                    path: str = "./",
                    geqdsk: str | None = None,
                    electron: str = "elc",
                    ion: str = "ion") -> SheathEntrance:
  """The sheath entrance at every target plate: e (phi_se - phi_wall) / T_e
  from the sheath boundary condition's electron cutoff, the textbook
  floating-sheath drop, the ion Mach number toward the plate, T_e and T_i, per
  open SOL cell row of each target's leg, at the run's frames with the
  sheath files.

  The frames are those of ``frame`` at which the run wrote the sheath files
  (:func:`sheath_frames`); the others are skipped. The layout is built once,
  from the run's first frame (the sheath boundary condition's rows do not
  change in time).

  Args:
    name: The run's file prefix (without ``_b<k>``).
    frame: Frame number, comma list, or ``'start:stop[:step]'`` range
      (``':'``: all the sheath frames).
    path: Directory holding the run's output.
    geqdsk: The run's equilibrium file, for psi_N of the rows (and the
      layout's orientation where the topology does not fix it).
    electron: The electron species (negative charge).
    ion: The ion species (positive charge).

  Returns:
    The :class:`SheathEntrance`.

  Raises:
    ValueError: No frame of ``frame`` has the sheath files, the species'
      charges have the wrong signs, or a target's leg wrote its sheath file
      at some of the frames but not all.
  """
  layout = multib_layout(name, path=path, geqdsk=geqdsk)
  if not layout.targets:
    raise ValueError(f"{name}: the layout has no target plate")
  first = layout.targets[0]
  frames = [
      k for k in resolve_frames(
          frame,
          name=os.path.join(layout.path, layout.name),
          species=electron,
          suffix=f"bc_z{_FACE[first.theta_end]}_vcutsq",
          block_idx=None if layout.single_block else first.leg)
      if os.path.exists(_vcut_file(layout, first, electron, k))
  ]
  if not frames:
    raise ValueError(f"{name}: the run wrote no '{electron}' sheath files at "
                     f"frame {frame!r}")
  meta = {
      s: io.read_metadata(layout.file_name(0, f"{s}_M0", frames[0]))
      for s in (electron, ion)
  }
  if not meta[electron]["charge"] < 0 < meta[ion]["charge"]:
    raise ValueError(f"'{electron}' must be negatively charged and '{ion}' "
                     "positively")
  time = [
      io.read_metadata(layout.file_name(0, f"{electron}_M0",
                                        k)).get("time", np.nan) for k in frames
  ]
  psi_n = layout_flux_surfaces(layout, geqdsk).cells if geqdsk else None
  profiles = []
  for t in layout.targets:
    missing = [
        k for k in frames
        if not os.path.exists(_vcut_file(layout, t, electron, k))
    ]
    if len(missing) == len(frames):
      continue
    if missing:
      raise ValueError(f"{name}: {t.name}'s leg (block {t.leg}) wrote no "
                       f"'{electron}' sheath file at frames {missing}")
    values = [
        np.array(v)
        for v in zip(*(_plate_terms(layout, t, k, electron, ion, meta)
                       for k in frames))
    ]
    ok = np.isfinite(values[0]).any(axis=0)
    nr = layout.nodes[t.leg].shape[0] - 1
    profiles.append(
        SheathProfile(t.name, t.leg,
                      np.arange(t.open_row, nr)[ok],
                      None if psi_n is None else psi_n[t.leg][t.open_row:][ok],
                      *(v[:, ok] for v in values)))
  return SheathEntrance(layout.name, layout.path, np.array(frames),
                        np.array(time), electron, ion, tuple(profiles))


__all__ = [
    "SheathEntrance", "SheathProfile", "face_weights", "floating_drop",
    "sheath_entrance", "sheath_frames", "sheath_terms", "temperature"
]
