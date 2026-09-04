"""2-D/3-D five-moment enstrophy diagnostic.

Ported from ``src_bak/postgkyl/tools/calc_enstrophy.py``. Sweeps a family of
five-moment output frames (density + momentum, ``rho, px, py, pz``) and
computes, per frame, the enstrophy in its general form (integral of the
squared magnitude of the curl of the velocity over the volume) and its
incompressible form (integral of a velocity-gradient invariant, weighted by
density).

Fixes one bug present in ``src_bak``: ``incom_enstrophy = enstrophy`` aliased
the very array the general-form result was written into, so both returned
traces ended up identical (equal to whichever form was written last in the
frame loop) instead of being the two distinct quantities the function's own
docstring and return statement promised -- doctrine #21 requires fixing an
unambiguous bug rather than silently porting it forward. The per-cell nested
loop's ``range(len(axis) - 1)`` bound (leaving the last plane along every
axis at zero) is preserved verbatim: unlike the aliasing, it is not
unambiguously a bug (it could be deliberate avoidance of a less-accurate
``np.gradient`` edge-order boundary), so changing it would be a silent
numerical-behavior change doctrine #21 forbids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from postgkyl.gdata import GData


@dataclass(frozen=True)
class EnstrophyTraces:
  """Per-frame enstrophy traces, one entry per swept frame.

  Attributes:
    enstrophy: General-form enstrophy (integral of the squared curl
      magnitude).
    incompressible_enstrophy: Incompressible-form enstrophy (integral of a
      density-weighted velocity-gradient invariant).
  """

  enstrophy: np.ndarray
  incompressible_enstrophy: np.ndarray


def _enstrophy_terms(rho: np.ndarray, px: np.ndarray, py: np.ndarray,
                     pz: np.ndarray, dx: float, dy: float,
                     dz: float) -> tuple[float, float]:
  """Pure array math: the general and incompressible enstrophy integrals
  for one frame of five-moment (density + momentum) data.

  Args:
    rho, px, py, pz: 3-D density and momentum-component arrays (same shape).
    dx, dy, dz: Grid spacing along each axis.

  Returns:
    ``(enstrophy, incompressible_enstrophy)``: the two scalar integrals for
    this frame.
  """
  u = px / rho
  v = py / rho
  w = pz / rho

  u_grad = np.gradient(u, dx, dy, dz, edge_order=2)
  v_grad = np.gradient(v, dx, dy, dz, edge_order=2)
  w_grad = np.gradient(w, dx, dy, dz, edge_order=2)
  grad_tensor = np.array([u_grad, v_grad, w_grad])

  u_x, u_y, u_z = u_grad
  v_x, v_y, v_z = v_grad
  w_x, w_y, w_z = w_grad

  curl_mag = (w_y - v_z)**2 + (u_z - w_x)**2 + (v_x - u_y)**2
  enstrophy = np.sum(curl_mag, axis=(0, 1, 2)) * dx * dy * dz

  nx, ny, nz = rho.shape
  incom_mag = np.zeros((nx, ny, nz))
  for c in range(nx - 1):
    for j in range(ny - 1):
      for k in range(nz - 1):
        cell = grad_tensor[:, :, c, j, k]
        incom_mag[c, j, k] = np.trace(np.transpose(cell) * cell) * rho[c, j, k]
  incompressible_enstrophy = np.sum(incom_mag, axis=(0, 1, 2)) * dx * dy * dz

  return enstrophy, incompressible_enstrophy


def enstrophy(
    stem: str,
    init_frame: int,
    final_frame: int,
    *,
    extension: str = "gkyl",
) -> EnstrophyTraces:
  """Sweep a frame family and compute the enstrophy in 2 forms.

  Args:
    stem: File-name stem before the frame number, e.g. ``"sim-fluid_"``.
    init_frame: First frame (inclusive).
    final_frame: Last frame (inclusive).
    extension: File extension of the frame files (defaults to the native
      ``gkyl`` format).

  Returns:
    :class:`EnstrophyTraces`, one entry per swept frame.
  """
  num_frames = final_frame - init_frame + 1

  first = GData(f"{stem}{init_frame}.{extension}")
  grid = first.grid
  dx = grid[0][1] - grid[0][0]
  dy = grid[1][1] - grid[1][0]
  dz = grid[2][1] - grid[2][0]

  enstrophy_trace = np.empty(num_frames)
  incompressible_trace = np.empty(num_frames)
  for r, frame_idx in enumerate(range(init_frame, final_frame + 1)):
    data = GData(f"{stem}{frame_idx}.{extension}")
    values = data.values
    rho, px, py, pz = (values[..., c] for c in range(4))
    enstrophy_trace[r], incompressible_trace[r] = _enstrophy_terms(
        rho, px, py, pz, dx, dy, dz)

  return EnstrophyTraces(enstrophy=enstrophy_trace,
                         incompressible_enstrophy=incompressible_trace)
