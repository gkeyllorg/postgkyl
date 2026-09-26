"""Frame-integrated fluid kinetic energy and its physical-time differences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from postgkyl.gdata import GData
from postgkyl.operations import integrate

from .five_moment import ke as kinetic_energy


@dataclass(frozen=True)
class KineticEnergyTraces:
  """Integrated kinetic energy and negative consecutive time differences.

  ``ke`` has one entry per frame. ``dke`` has one entry per frame interval,
  with ``dke[i] = -(ke[i+1] - ke[i]) / (time[i+1] - time[i])``.
  """

  ke: np.ndarray
  dke: np.ndarray


def _dissipation_rate(ke: np.ndarray, times: np.ndarray) -> np.ndarray:
  """Negative energy slopes over strictly increasing physical timestamps."""
  times = np.asarray(times, dtype=float)
  if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
    raise ValueError("frame timestamps must be finite and strictly increasing")
  return -np.diff(ke) / np.diff(times)


def ke_dke(
    root_file_name: str,
    init_frame: int,
    final_frame: int,
    dim: int,
    vol: float,
    init_time: float,
    final_time: float,
    *,
    extension: str = "gkyl",
) -> KineticEnergyTraces:
  """Integrate kinetic energy and compute its negative rate of change.

  The first four fluid components are ``rho, px, py, pz``; total energy
  need not be present. The energy density is ``|momentum|**2 / (2*rho)``.
  Modal inputs use
  native weak multiplication/division and DG integration; point-value inputs
  use the matching numerical integration. Each frame's own grid is used.

  Args:
    root_file_name: File-name stem before the frame number.
    init_frame: First frame (inclusive).
    final_frame: Last frame (inclusive).
    dim: Spatial dimension (2 or 3), checked against every frame.
    vol: Additional volume multiplier, for example an omitted physical
      thickness in a 2D simulation. Grid cell volumes are already integrated.
    init_time: First-frame time used only when every frame lacks timestamps.
    final_time: Last-frame time used only when every frame lacks timestamps;
      the fallback assumes uniform spacing across the frame intervals.
    extension: Frame-file extension.

  Returns:
    One kinetic energy per frame and one dissipation rate per interval.

  Raises:
    ValueError: If dimensions, frame range, or timestamps are invalid. A
      mixture of present and absent timestamps is not sufficient to define
      the intervening physical time intervals.
  """
  if dim not in (2, 3):
    raise ValueError("ke_dke requires dim=2 or dim=3")
  if final_frame < init_frame:
    raise ValueError("final_frame must be at least init_frame")
  num_frames = final_frame - init_frame + 1
  energy = np.empty(num_frames)
  times = []
  for index, frame in enumerate(range(init_frame, final_frame + 1)):
    data = GData(f"{root_file_name}{frame}.{extension}")
    if data.num_dims != dim:
      raise ValueError(
          f"frame {frame} has {data.num_dims} dimensions, expected {dim}")
    energy[index] = float(integrate(kinetic_energy(data, num_moms=5))) * vol
    times.append(data.ctx.get("time"))
  if all(time is None for time in times):
    times = np.linspace(init_time, final_time, num_frames)
  elif any(time is None for time in times):
    raise ValueError("frame timestamps must be present for every frame or none")
  return KineticEnergyTraces(ke=energy, dke=_dissipation_rate(energy, times))
