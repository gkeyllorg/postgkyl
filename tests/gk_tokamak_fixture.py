"""A synthetic three-block half-domain tokamak run, for the budget, region
and diffusivity diagnostics.

Geometry (circular flux surfaces about (R0, 0), polar angle th, Z = r sin th,
both on node rows and columns):

- block 0, core: r in [0.5, 1], th in [0, pi]; its theta faces lie on Z = 0
  (the reflecting midplane of a half domain), its psi-lower face physical.
- block 1, SOL over block 0: r in [1, 1.5], th in [0, pi]; its psi-upper and
  theta-lower faces physical, its theta-upper face joined to block 2.
- block 2, leg: r in [1, 1.5], th in [pi, 5 pi / 4]; its psi faces and
  theta-upper face physical: the target plate (``'lower inner target'``,
  face ``yupper``).

Block 1 crosses Z = 0 at both theta ends, so the run has an outboard and an
inboard midplane, each the core block under the SOL block.

Every block's logical grid differs (``LOWER``/``UPPER``). Modal fields are
bilinear ``a + b x + c y + e x y`` with closed-form coefficients on the
orthonormal 2D p = 1 serendipity basis ``1/2, sqrt(3) xi / 2,
sqrt(3) eta / 2, 3 xi eta / 2``. Dynvectors are written with their full time
series. No production code computes any of this.

:func:`write_plate_files` adds, on the leg, the moments of the sheath
entrance and the heat flux onto the plate (:func:`write_run` leaves them out).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from generate_test_data import write_gkyl_dynvector, write_gkyl_field

NAME = "tok"
R0 = 3.0
PHYSICAL = 5
LOWER_POS, UPPER_POS = 1, 3
# Per block: (r range, theta range, cells (nr, nt), logical lower, upper).
BLOCKS = (
    ((0.5, 1.0), (0.0, np.pi), (2, 4), (0.0, 0.0), (1.0, 2.0)),
    ((1.0, 1.5), (0.0, np.pi), (2, 4), (1.0, 0.0), (2.0, 2.0)),
    ((1.0, 1.5), (np.pi, 1.25 * np.pi), (2, 2), (1.0, 2.0), (2.0, 3.0)),
)
MASS = {"elc": 9.0e-31, "ion": 3.0e-27}
CHARGE = {"elc": -1.6e-19, "ion": 1.6e-19}
FRAME_TIMES = (0.0, 0.5, 1.0)
SAMPLES = np.linspace(0.0, 1.0, 11)

# Jacobian J = 1 + x in every block.
JACOBIAN = (1.0, 1.0, 0.0, 0.0)
# g^xx J = 2 + y / 4 (positive at every corner).
GXXJ = (2.0, 0.0, 0.25, 0.0)


def density(block: int, frame: int) -> tuple[float, float, float, float]:
  """``(a, b, c, e)`` of the electron density ``a + b x``; block 2 at frame
  2 is ``-4 + 2.5 x``: negative in its x cell [1, 1.5]."""
  if block == 2 and frame == 2:
    return (-4.0, 2.5, 0.0, 0.0)
  return (1.0 + block + frame, 0.5 + 0.25 * block, 0.0, 0.0)


def diffusivity(block: int) -> tuple[float, float, float, float]:
  """``(a, b, c, e)`` of the diffusivity ``D = (1 + block) + x / 2``."""
  return (1.0 + block, 0.5, 0.0, 0.0)


def cell_edges(block: int) -> tuple[np.ndarray, np.ndarray]:
  _, _, (nr, nt), lo, up = BLOCKS[block]
  return np.linspace(lo[0], up[0], nr + 1), np.linspace(lo[1], up[1], nt + 1)


def bilinear_coefficients(block: int, poly) -> np.ndarray:
  """Modal coefficients ``(nr, nt, 4)`` of ``a + b x + c y + e x y``."""
  a, b, c, e = poly
  xe, ye = cell_edges(block)
  xc, hx = 0.5 * (xe[1:] + xe[:-1]), 0.5 * np.diff(xe)
  yc, hy = 0.5 * (ye[1:] + ye[:-1]), 0.5 * np.diff(ye)
  X, Y = np.meshgrid(xc, yc, indexing="ij")
  HX, HY = np.meshgrid(hx, hy, indexing="ij")
  return np.stack([
      2 * (a + b * X + c * Y + e * X * Y), 2 / np.sqrt(3) *
      (b + e * Y) * HX, 2 / np.sqrt(3) * (c + e * X) * HY, 2 / 3 * e * HX * HY
  ],
                  axis=-1)


def product(p, q) -> tuple[float, float, float, float]:
  """The bilinear product of ``a + b x`` (``p``) and ``c0 + c y`` (``q``)."""
  return (p[0] * q[0], p[1] * q[0], p[0] * q[2], p[1] * q[2])


def nodes(block: int) -> np.ndarray:
  """Corner nodes ``(nr + 1, nt + 1, 2)`` in (R, Z); Z on the midplane is
  exactly 0."""
  (r0, r1), (t0, t1), (nr, nt), _, _ = BLOCKS[block]
  r, th = np.meshgrid(np.linspace(r0, r1, nr + 1),
                      np.linspace(t0, t1, nt + 1),
                      indexing="ij")
  Z = r * np.sin(th)
  Z[np.abs(Z) < 1e-12] = 0.0
  return np.stack([R0 + r * np.cos(th), Z], axis=-1)


def connections() -> list[int]:
  """The block topology, ``(block, dir, edge)`` per (block, dir, side)."""
  phys = (0, 0, PHYSICAL)
  conn = [
      # block 0: psi lower physical, psi upper -> block 1; theta physical
      [[phys, (1, 0, LOWER_POS)], [phys, phys]],
      # block 1: psi lower -> block 0, psi upper physical; theta upper -> 2
      [[(0, 0, UPPER_POS), phys], [phys, (2, 1, LOWER_POS)]],
      # block 2: psi physical; theta lower -> block 1, theta upper physical
      [[phys, phys], [(1, 1, UPPER_POS), phys]],
  ]
  return [int(v) for v in np.asarray(conn).ravel()]


def _field(path: Path, block: int, values: np.ndarray, **meta) -> None:
  _, _, cells, lo, up = BLOCKS[block]
  write_gkyl_field(path,
                   list(cells),
                   list(lo),
                   list(up),
                   values,
                   poly_order=1,
                   basis_type="serendipity",
                   metadata={
                       "value_form": "modal",
                       **meta
                   })


def write_run(path: Path, rates: dict[str, np.ndarray] | None = None) -> Path:
  """Write the run under ``path``; ``rates`` maps a block-file name (e.g.
  ``'b0-elc_fdot_integrated_HamiltonianMoments'``) to its samples'
  ``(n, 3)`` values. Returns the directory."""
  path.mkdir(parents=True, exist_ok=True)
  dummy = np.zeros((1, 1, 1))
  write_gkyl_field(path / f"{NAME}-block_topo.gkyl", [1, 1], [0.0, 0.0],
                   [1.0, 1.0],
                   dummy,
                   poly_order=0,
                   basis_type="serendipity",
                   metadata={
                       "num_blocks": len(BLOCKS),
                       "ndim": 2,
                       "connections": connections()
                   })
  for b in range(len(BLOCKS)):
    pre = path / f"{NAME}_b{b}"
    nd = nodes(b)
    cart = np.stack([nd[..., 0], np.zeros_like(nd[..., 0]), nd[..., 1]], -1)
    write_gkyl_field(Path(f"{pre}-geo_corn_nodes.gkyl"),
                     list(nd.shape[:2]), [0.0, 0.0], [1.0, 1.0],
                     cart,
                     poly_order=0,
                     basis_type="serendipity",
                     metadata={"geometry_type": 3})
    _field(Path(f"{pre}-geo_int_jacobgeo.gkyl"), b,
           bilinear_coefficients(b, JACOBIAN))
    _field(Path(f"{pre}-geo_int_gxxj.gkyl"), b, bilinear_coefficients(b, GXXJ))
    for s in MASS:
      _field(Path(f"{pre}-{s}_anom_diff_0.gkyl"),
             b,
             bilinear_coefficients(b, product(diffusivity(b), GXXJ)),
             mass=MASS[s],
             charge=CHARGE[s])
    for k, t in enumerate(FRAME_TIMES):
      _field(Path(f"{pre}-field_{k}.gkyl"),
             b,
             np.zeros(BLOCKS[b][2] + (4, )),
             time=t,
             frame=k)
      for s in MASS:
        _field(Path(f"{pre}-{s}_M0_{k}.gkyl"),
               b,
               bilinear_coefficients(b, density(b, k)),
               time=t,
               frame=k,
               mass=MASS[s],
               charge=CHARGE[s])
  for what, values in (rates or {}).items():
    write_gkyl_dynvector(path / f"{NAME}_{what}.gkyl", SAMPLES, values,
                         {"mass": 1.0})
  return path


# ---------------------------------------------------- the plate (block 2)
LEG = 2
# The sheath boundary condition writes its electron cutoff with the
# phase-space output: every other frame.
SHEATH_FRAMES = (0, 2)
ELEMENTARY_CHARGE = -CHARGE["elc"]
# Per frame: uniform T_e, T_i (eV), the ions' flow toward the plate in units
# of c_s, and e (phi_se - phi_wall) / T_e per cell row of the leg.
TE_EV = (20.0, 30.0, 40.0)
TI_EV = (30.0, 35.0, 40.0)
MACH = (1.0, 1.1, 1.2)
DROP = ((2.5, 3.0), (2.6, 3.1), (2.7, 3.2))
# Per frame: (h0, h1) of the parallel flux of kinetic energy summed over the
# species in the plate cell column, h0 + h1 x (W/m^2); the other column
# differs by HEAT_FLUX_Y (y - Y_PLATE), Y_PLATE its cell centre y.
HEAT_FLUX = ((6.0e6, -2.0e6), (3.0e6, -4.0e6), (3.0e6, 0.0))
HEAT_FLUX_Y, Y_PLATE = 1.0e6, 2.75
# Each species' share of HEAT_FLUX: (par, perp).
HEAT_FLUX_SHARE = {"elc": (0.25, 0.25), "ion": (0.5, 0.0)}


def temperature(species: str, frame: int) -> float:
  """T (J), uniform."""
  return (TE_EV if species == "elc" else TI_EV)[frame] * ELEMENTARY_CHARGE


def flow(species: str, frame: int) -> float:
  """u_par (m/s), uniform: the electrons at rest, the ions toward the plate
  (+theta) at MACH c_s, c_s^2 = (T_e + T_i) / m_i."""
  if species == "elc":
    return 0.0
  return MACH[frame] * np.sqrt(
      (temperature("elc", frame) + temperature("ion", frame)) / MASS["ion"])


def _scaled(poly, s) -> tuple[float, ...]:
  return tuple(s * c for c in poly)


def write_plate_files(path: Path) -> Path:
  """Add to a run written by :func:`write_run`, on the leg: each species' M1
  and M2 of a drifting Maxwellian over its M0 (uniform u = :func:`flow`,
  T = :func:`temperature`: M1 = u M0, M2 = (u^2 + 3 T / m) M0), M3par and
  M3perp with (m/2)(M3par + M3perp) summed over the species = HEAT_FLUX, and
  at SHEATH_FRAMES the electrons' cutoff velocity squared at the plate face,
  m_e v_cut^2 / 2 = DROP T_e per cell row. Returns the directory."""
  pre = path / f"{NAME}_b{LEG}"
  (nr, _), lo, up = BLOCKS[LEG][2:]
  for k, t in enumerate(FRAME_TIMES):
    meta = {"time": t, "frame": k}
    h0, h1 = HEAT_FLUX[k]
    heat = (h0 - HEAT_FLUX_Y * Y_PLATE, h1, HEAT_FLUX_Y, 0.0)
    for s in MASS:
      species = {**meta, "mass": MASS[s], "charge": CHARGE[s]}
      n, u = density(LEG, k), flow(s, k)
      moments = {
          "M1": _scaled(n, u),
          "M2": _scaled(n, u**2 + 3 * temperature(s, k) / MASS[s]),
          "M3par": _scaled(heat, 2 / MASS[s] * HEAT_FLUX_SHARE[s][0]),
          "M3perp": _scaled(heat, 2 / MASS[s] * HEAT_FLUX_SHARE[s][1]),
      }
      for m, poly in moments.items():
        _field(Path(f"{pre}-{s}_{m}_{k}.gkyl"), LEG,
               bilinear_coefficients(LEG, poly), **species)
    if k in SHEATH_FRAMES:
      vcut_sq = 2 * np.array(DROP[k]) * temperature("elc", k) / MASS["elc"]
      values = np.zeros((nr, 1, 4))
      values[:, 0, 0] = 2 * vcut_sq  # the cell mean is c0 / 2
      write_gkyl_field(Path(f"{pre}-elc_bc_zupper_vcutsq_{k}.gkyl"), [nr, 1],
                       [lo[0], up[1]], [up[0], up[1] + 1.0],
                       values,
                       poly_order=1,
                       basis_type="serendipity",
                       time=t,
                       frame=k,
                       metadata={"value_form": "modal"})
  return path
