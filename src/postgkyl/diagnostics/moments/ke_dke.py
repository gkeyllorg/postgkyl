"""Five-moment kinetic-energy / dissipation-rate diagnostic.

Ported from ``src_bak/postgkyl/tools/calc_ke_dke.py``. Sweeps a family of
five-moment output frames (density + momentum, ``rho, px, py, pz``),
integrates the kinetic energy over the grid for each frame, and estimates
its dissipation rate by backward finite difference between consecutive
frames.

Fixes three bugs present in ``src_bak`` (doctrine #21: fix an unambiguous
bug rather than silently port it forward):

  - the per-frame file name inside the sweep loop was built as
    ``f"root_file_name{c:d}.gkyl"`` -- a literal string containing the
    parameter's *name*, not an f-string interpolating its *value*
    (``f"{root_file_name}{c:d}.gkyl"``); only the *first* frame, read once
    before the loop to get the grid spacing, used the correct spelling;
  - ``dEk = ke`` aliased the very array the kinetic-energy trace was
    written into (instead of allocating its own array), so writing the
    dissipation-rate trace corrupted not-yet-read kinetic-energy values;
  - the difference loop's ``range(init_frame, final_frame - 1)`` is off by
    one frame short of every valid backward difference (it should run
    through ``final_frame - 1`` inclusive, i.e. ``range(init_frame,
    final_frame)``).

Combined, no variant of the original code could ever have produced a
meaningful trace, so this ports the clearly-intended calculation (every
consecutive-frame backward difference) rather than reproducing undefined
behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from postgkyl.gdata import GData


@dataclass(frozen=True)
class KineticEnergyTraces:
  """Per-frame kinetic-energy traces.

  Attributes:
    ke: Integrated kinetic energy, one entry per swept frame.
    dke: Dissipation rate (backward difference of ``ke``), one entry per
      consecutive frame pair -- one shorter than ``ke``.
  """

  ke: np.ndarray
  dke: np.ndarray
# end


def _kinetic_energy(rho: np.ndarray, px: np.ndarray, py: np.ndarray,
    pz: np.ndarray, dx: float, dy: float, dz: float, vol: float) -> float:
  """Pure array math: the integrated kinetic energy for one frame."""
  u = px / rho
  v = py / rho
  w = pz / rho
  e = rho * (u ** 2 + v ** 2 + w ** 2)
  return np.sum(e, axis=(0, 1, 2)) * dx * dy * dz * vol
# end


def _dissipation_rate(ke: np.ndarray, dt: float) -> np.ndarray:
  """Backward-difference dissipation rate between every consecutive pair:
  ``dke[i] = -(ke[i + 1] - ke[i]) / dt``."""
  return -(ke[1:] - ke[:-1]) / dt
# end


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
  """Sweep a frame family and compute the kinetic energy and dissipation rate.

  Args:
    root_file_name: File-name stem before the frame number.
    init_frame: First frame (inclusive).
    final_frame: Last frame (inclusive).
    dim: Simulation dimensionality (2 or 3); the z grid spacing is taken as
      1 when ``dim != 3``.
    vol: Grid cell volume factor.
    init_time: Simulation start time.
    final_time: Simulation end time; used with ``init_time`` to derive a
      uniform ``dt`` for the dissipation-rate estimate.
    extension: File extension of the frame files (defaults to the native
      ``gkyl`` format).

  Returns:
    :class:`KineticEnergyTraces`.
  """
  num_frames = final_frame - init_frame + 1
  dt = (final_time - init_time + 1) / num_frames

  first = GData(f"{root_file_name}{init_frame}.{extension}")
  grid = first.grid
  dx = grid[0][1] - grid[0][0]
  dy = grid[1][1] - grid[1][0]
  dz = grid[2][1] - grid[2][0] if dim == 3 else 1

  ke = np.empty(num_frames)
  for r, frame_idx in enumerate(range(init_frame, final_frame + 1)):
    data = GData(f"{root_file_name}{frame_idx}.{extension}")
    values = data.values
    rho, px, py, pz = (values[..., c] for c in range(4))
    ke[r] = _kinetic_energy(rho, px, py, pz, dx, dy, dz, vol)
  # end

  dke = _dissipation_rate(ke, dt)
  return KineticEnergyTraces(ke=ke, dke=dke)
# end
