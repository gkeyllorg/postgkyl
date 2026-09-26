"""Two- and three-dimensional fluid enstrophy from physical velocity fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from postgkyl.gdata import GData


@dataclass(frozen=True)
class EnstrophyTraces:
  """Frame integrals of ``|curl(u)|**2`` and ``rho*sum_ij(du_i/dx_j)**2``.

  The second quantity is the density-weighted squared velocity-gradient
  norm, conventionally used for incompressible viscous dissipation. These
  definitions omit an optional factor of one half.
  """

  enstrophy: np.ndarray
  incompressible_enstrophy: np.ndarray


def _enstrophy_terms(data: GData) -> tuple[float, float]:
  """Compose velocity, derivatives and integration with their DG semantics."""
  if data.num_dims not in (2, 3):
    raise ValueError("enstrophy requires two or three spatial dimensions")
  if data.backend == "gkyl" and data.ctx.get("value_form") != "modal":
    data = data.represent(to="modal")
  rho = data.select(comp=0)
  velocity = [data.select(comp=i) / rho for i in (1, 2, 3)]
  gradient = [[
      component.differentiate(direction=d) for d in range(data.num_dims)
  ] for component in velocity]
  if data.num_dims == 2:
    for row in gradient:
      row.append(rho * 0.0)
  curl = [
      gradient[2][1] - gradient[1][2], gradient[0][2] - gradient[2][0],
      gradient[1][0] - gradient[0][1]
  ]
  squared_curl = sum(component**2 for component in curl)
  squared_gradient = sum(entry**2 for row in gradient for entry in row)
  return (float(squared_curl.integrate()),
          float((rho * squared_gradient).integrate()))


def enstrophy(
    stem: str,
    init_frame: int,
    final_frame: int,
    *,
    extension: str = "gkyl",
) -> EnstrophyTraces:
  """Integrate squared curl and density-weighted velocity gradients per frame.

  Native DG data uses weak velocity division, local polynomial derivatives
  (excluding inter-cell jumps), weak products, and DG integration. Native
  nodal/quadrature data is first represented as modal. Point samples use
  numerical derivatives and integration at their physical coordinates. All
  cells, including boundary cells, contribute. For two spatial dimensions,
  derivatives in the absent third direction are zero; all three velocity
  components still contribute.

  Args:
    stem: File-name stem before the frame number, e.g. ``"sim-fluid_"``.
    init_frame: First frame (inclusive).
    final_frame: Last frame (inclusive).
    extension: Frame-file extension.

  Returns:
    One value of each integral per frame.
  """
  if final_frame < init_frame:
    raise ValueError("final_frame must be at least init_frame")
  num_frames = final_frame - init_frame + 1
  curl_trace = np.empty(num_frames)
  gradient_trace = np.empty(num_frames)
  for index, frame in enumerate(range(init_frame, final_frame + 1)):
    data = GData(f"{stem}{frame}.{extension}")
    curl_trace[index], gradient_trace[index] = _enstrophy_terms(data)
  return EnstrophyTraces(enstrophy=curl_trace,
                         incompressible_enstrophy=gradient_trace)
