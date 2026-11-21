"""The sheath entrance at a gyrokinetic tokamak run's target plates.

At each target's plate face (the theta face of its leg block's end cell
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
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import os

import numpy as np

from postgkyl import dg, io
from postgkyl.diagnostics import discovery
from postgkyl.gdata import GData

from .flux_coords import layout_flux_surfaces
from .multib_layout import block_prefixes, multib_layout, psi_cells

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
  """The sheath entrance along one target plate, per open SOL cell row.

  Attributes:
    target: The target's name.
    block: Its leg block.
    rows: The leg's cell rows (inside out) the values are for: its open rows
      where the sheath files give a finite drop.
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


def _span(v: np.ndarray) -> str:
  return f"{np.min(v):.4g}..{np.max(v):.4g}" if len(v) else "-"


@dataclass(frozen=True, eq=False)
class SheathEntrance:
  """The sheath entrance at every target plate of a run at one frame.

  Attributes:
    name: The run's file prefix.
    path: Its directory.
    frame: The frame.
    electron: The electron species.
    ion: The ion species.
    targets: Each target whose leg wrote the electrons' sheath file at the
      frame, in the layout's target order.
  """

  name: str
  path: str
  frame: int
  electron: str
  ion: str
  targets: tuple[SheathProfile, ...]

  def __str__(self) -> str:
    lines = [
        f"{self.name} frame {self.frame}: sheath entrance "
        f"({self.electron}, {self.ion})"
    ]
    for p in self.targets:
      lines.append(
          f"  {p.target} (block {p.block}, {len(p.rows)} rows): "
          f"e dphi/T_e {_span(p.drop)} (floating {_span(p.floating)}), "
          f"Mach {_span(p.mach)}, T_e {_span(p.te)} eV, T_i {_span(p.ti)} eV" +
          ("" if p.psi_n is None else f", psi_N {_span(p.psi_n)}"))
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


def sheath_entrance(name: str,
                    frame: int,
                    *,
                    path: str = "./",
                    geqdsk: str | None = None,
                    electron: str = "elc",
                    ion: str = "ion") -> SheathEntrance:
  """The sheath entrance at every target plate: e (phi_se - phi_wall) / T_e
  from the sheath boundary condition's electron cutoff, the textbook
  floating-sheath drop, the ion Mach number toward the plate, T_e and T_i, per
  open SOL cell row of each target's leg.

  Args:
    name: The run's file prefix (without ``_b<k>``).
    frame: A frame with the sheath files (:func:`sheath_frames`).
    path: Directory holding the run's output.
    geqdsk: The run's equilibrium file, for psi_N of the rows (and the
      layout's orientation where the topology does not fix it).
    electron: The electron species (negative charge).
    ion: The ion species (positive charge).

  Returns:
    The :class:`SheathEntrance`.

  Raises:
    ValueError: The species' charges have the wrong signs, or no target's
      leg wrote the electrons' sheath file at ``frame``.
  """
  layout = multib_layout(name, path=path, geqdsk=geqdsk, frame=frame)
  meta = {
      s: io.read_metadata(layout.file_name(0, f"{s}_M0", frame))
      for s in (electron, ion)
  }
  if not meta[electron]["charge"] < 0 < meta[ion]["charge"]:
    raise ValueError(f"'{electron}' must be negatively charged and '{ion}' "
                     "positively")
  psi_n = layout_flux_surfaces(layout, geqdsk).cells if geqdsk else None
  profiles = []
  for t in layout.targets:
    b, side = t.leg, -1 if t.theta_end == 0 else 1
    vfile = layout.file_name(b, f"{electron}_bc_z{_FACE[t.theta_end]}_vcutsq",
                             frame)
    if not os.path.exists(vfile):
      continue
    vcut, vctx = _modal(vfile)
    nb = dg.num_basis(vcut.ndim - 1, vctx["poly_order"], vctx["basis_type"])
    vcut_sq = dg.modal.cell_means(vcut, vcut.ndim - 1, nb)[..., 0].mean(axis=1)
    face = {
        (s, m):
        _face_values(layout.file_name(b, f"{s}_{m}", frame), t.theta_end, side)
        for s in (electron, ion)
        for m in _MOMENTS
    }
    r0 = psi_cells(face[electron, "M0"][1], vctx).start
    covered = slice(r0, r0 + len(vcut_sq))
    values = sheath_terms(vcut_sq,
                          tuple(face[electron, m][0][covered]
                                for m in _MOMENTS),
                          tuple(face[ion, m][0][covered] for m in _MOMENTS),
                          elc_mass=meta[electron]["mass"],
                          ion_mass=meta[ion]["mass"],
                          elementary_charge=-meta[electron]["charge"],
                          toward=side)
    nr = layout.nodes[b].shape[0] - 1
    rows = []
    for v in values:  # the leg's open rows, inside out
      full = np.full(nr, np.nan)
      full[covered] = v
      rows.append(layout.rows(b, full)[t.open_row:])
    ok = np.isfinite(rows[0])
    profiles.append(
        SheathProfile(t.name, b,
                      np.arange(t.open_row, nr)[ok],
                      None if psi_n is None else psi_n[b][t.open_row:][ok],
                      *(v[ok] for v in rows)))
  if not profiles:
    raise ValueError(f"{name}: no target's leg wrote '{electron}' sheath files "
                     f"at frame {frame}")
  return SheathEntrance(layout.name, layout.path, frame, electron, ion,
                        tuple(profiles))


__all__ = [
    "SheathEntrance", "SheathProfile", "face_weights", "floating_drop",
    "sheath_entrance", "sheath_frames", "sheath_terms", "temperature"
]
