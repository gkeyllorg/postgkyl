"""Ten-moment diagnostics -- the five-moment set (fixed to 10-moment data)
plus the pressure tensor, field-aligned pressure, and agyrotropy.

10-moment fluid data is laid out ``[rho, mx, my, mz, Pxx, Pxy, Pxz, Pyy, Pyz,
Pzz]``; the pressure tensor components below subtract the bulk-flow (ram)
contribution from the raw second moments. ``p_par``/``p_perp``/``agyro``
then take an already-built 6-component pressure tensor (``P_xx, P_xy, P_xz,
P_yy, P_yz, P_zz``) and a 3-component magnetic field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ... import numerics
from ...gdatastate.guards import require_field_domain as _require_field_domain
from .five_moment import (
    _get_density,
    _get_vx,
    _get_vy,
    _get_vz,
    _get_p,
    _get_ke,
    _get_temp,
    _get_sound,
    _get_mach,
    density,
    xvel,
    yvel,
    zvel,
    vel,
)

if TYPE_CHECKING:
  from ...gdatastate.gdatastate import GDataState

_REASON = ("extracting primitive variables from raw DG coefficients would "
           "mix basis functions")
_AGYRO_REASON = ("computing agyrotropy from raw DG coefficients would mix "
                 "basis functions")


# --------------------------------------------------------- array-level math
def _get_pxx(grid: list[np.ndarray],
             values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """``P_xx = M_xx - rho * vx * vx`` (component 4 of 10-moment data)."""
  _, rho = _get_density(grid, values)
  _, vx = _get_vx(grid, values)
  return list(grid), values[..., 4, np.newaxis] - rho * vx * vx


def _get_pxy(grid: list[np.ndarray],
             values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """``P_xy = M_xy - rho * vx * vy`` (component 5 of 10-moment data)."""
  _, rho = _get_density(grid, values)
  _, vx = _get_vx(grid, values)
  _, vy = _get_vy(grid, values)
  return list(grid), values[..., 5, np.newaxis] - rho * vx * vy


def _get_pxz(grid: list[np.ndarray],
             values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """``P_xz = M_xz - rho * vx * vz`` (component 6 of 10-moment data)."""
  _, rho = _get_density(grid, values)
  _, vx = _get_vx(grid, values)
  _, vz = _get_vz(grid, values)
  return list(grid), values[..., 6, np.newaxis] - rho * vx * vz


def _get_pyy(grid: list[np.ndarray],
             values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """``P_yy = M_yy - rho * vy * vy`` (component 7 of 10-moment data)."""
  _, rho = _get_density(grid, values)
  _, vy = _get_vy(grid, values)
  return list(grid), values[..., 7, np.newaxis] - rho * vy * vy


def _get_pyz(grid: list[np.ndarray],
             values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """``P_yz = M_yz - rho * vy * vz`` (component 8 of 10-moment data)."""
  _, rho = _get_density(grid, values)
  _, vy = _get_vy(grid, values)
  _, vz = _get_vz(grid, values)
  return list(grid), values[..., 8, np.newaxis] - rho * vy * vz


def _get_pzz(grid: list[np.ndarray],
             values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """``P_zz = M_zz - rho * vz * vz`` (component 9 of 10-moment data)."""
  _, rho = _get_density(grid, values)
  _, vz = _get_vz(grid, values)
  return list(grid), values[..., 9, np.newaxis] - rho * vz * vz


def _get_pij(grid: list[np.ndarray],
             values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Full symmetric pressure tensor, packed
  ``(P_xx, P_xy, P_xz, P_yy, P_yz, P_zz)``."""
  out_values = np.zeros(values[..., 4:10].shape)
  _, pxx = _get_pxx(grid, values)
  _, pxy = _get_pxy(grid, values)
  _, pxz = _get_pxz(grid, values)
  _, pyy = _get_pyy(grid, values)
  _, pyz = _get_pyz(grid, values)
  _, pzz = _get_pzz(grid, values)

  out_values[..., 0] = np.squeeze(pxx)
  out_values[..., 1] = np.squeeze(pxy)
  out_values[..., 2] = np.squeeze(pxz)
  out_values[..., 3] = np.squeeze(pyy)
  out_values[..., 4] = np.squeeze(pyz)
  out_values[..., 5] = np.squeeze(pzz)

  return list(grid), out_values


def _get_p_par(
    p_grid: list[np.ndarray],
    p_values: np.ndarray,
    b_grid: list[np.ndarray],
    b_values: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the pressure parallel to the magnetic field.

  Projects the pressure tensor onto the magnetic-field direction:
  ``p_par = (b . P . b) / |B|**2``.

  Args:
    p_grid: Pressure-tensor grid.
    p_values: 6-component pressure tensor
      ``(P_xx, P_xy, P_xz, P_yy, P_yz, P_zz)``.
    b_grid: Magnetic-field grid.
    b_values: 3-component magnetic field ``(Bx, By, Bz)``.

  Returns:
    ``(grid, values)`` holding the parallel pressure field.
  """
  p_xx = p_values[..., 0, np.newaxis]
  p_xy = p_values[..., 1, np.newaxis]
  p_xz = p_values[..., 2, np.newaxis]
  p_yy = p_values[..., 3, np.newaxis]
  p_yz = p_values[..., 4, np.newaxis]
  p_zz = p_values[..., 5, np.newaxis]

  b_x = b_values[..., 0, np.newaxis]
  b_y = b_values[..., 1, np.newaxis]
  b_z = b_values[..., 2, np.newaxis]

  grid, mag_b_sq = numerics.mag_sq(b_grid, b_values)

  out = (b_x * b_x * p_xx + b_y * b_y * p_yy + b_z * b_z * p_zz + 2.0 *
         (b_x * b_y * p_xy + b_x * b_z * p_xz + b_y * b_z * p_yz)) / mag_b_sq
  return grid, out


def _get_gkyl_10m_p_par(
    species_grid: list[np.ndarray],
    species_values: np.ndarray,
    field_grid: list[np.ndarray],
    field_values: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the parallel pressure directly from raw 10-moment species and
  EM field data (whose components 3:6 are ``(Bx, By, Bz)``)."""
  p_grid, p_values = _get_pij(species_grid, species_values)
  b_values = field_values[..., 3:6]
  return _get_p_par(p_grid, p_values, field_grid, b_values)


def _get_p_perp(
    p_grid: list[np.ndarray],
    p_values: np.ndarray,
    b_grid: list[np.ndarray],
    b_values: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the pressure perpendicular to the magnetic field.

  Uses the trace of the pressure tensor and the parallel pressure:
  ``p_perp = (P_xx + P_yy + P_zz - p_par) / 2``.
  """
  p_xx = p_values[..., 0, np.newaxis]
  p_yy = p_values[..., 3, np.newaxis]
  p_zz = p_values[..., 5, np.newaxis]

  grid, p_par = _get_p_par(p_grid, p_values, b_grid, b_values)

  return grid, (p_xx + p_yy + p_zz - p_par) / 2.0


def _get_gkyl_10m_p_perp(
    species_grid: list[np.ndarray],
    species_values: np.ndarray,
    field_grid: list[np.ndarray],
    field_values: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the perpendicular pressure directly from raw 10-moment species
  and EM field data (whose components 3:6 are ``(Bx, By, Bz)``)."""
  p_grid, p_values = _get_pij(species_grid, species_values)
  b_values = field_values[..., 3:6]
  return _get_p_perp(p_grid, p_values, field_grid, b_values)


def _get_agyro(p_grid: list[np.ndarray],
               p_values: np.ndarray,
               b_grid: list[np.ndarray],
               b_values: np.ndarray,
               *,
               measure: str = "swisdak") -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the agyrotropy of the pressure tensor.

  The ``'swisdak'`` measure uses the tensor invariants and parallel pressure
  as in Appendix A of Swisdak (2015). The ``'frobenius'`` measure is the
  Frobenius norm of the non-gyrotropic part of the pressure tensor,
  normalized by the gyrotropic part.

  Args:
    p_grid: Pressure-tensor grid.
    p_values: 6-component pressure tensor
      ``(P_xx, P_xy, P_xz, P_yy, P_yz, P_zz)``.
    b_grid: Magnetic-field grid.
    b_values: 3-component magnetic field ``(Bx, By, Bz)``.
    measure: ``'swisdak'`` (default) or ``'frobenius'`` (case-insensitive).

  Returns:
    ``(grid, values)`` holding the agyrotropy field.

  Raises:
    ValueError: If ``measure`` is neither ``'swisdak'`` nor ``'frobenius'``.
  """
  p_xx = p_values[..., 0, np.newaxis]
  p_xy = p_values[..., 1, np.newaxis]
  p_xz = p_values[..., 2, np.newaxis]
  p_yy = p_values[..., 3, np.newaxis]
  p_yz = p_values[..., 4, np.newaxis]
  p_zz = p_values[..., 5, np.newaxis]

  b_x = b_values[..., 0, np.newaxis]
  b_y = b_values[..., 1, np.newaxis]
  b_z = b_values[..., 2, np.newaxis]

  grid, mag_b_sq = numerics.mag_sq(b_grid, b_values)
  _, p_par = _get_p_par(p_grid, p_values, b_grid, b_values)
  _, p_perp = _get_p_perp(p_grid, p_values, b_grid, b_values)

  measure_lower = measure.lower()
  if measure_lower == "swisdak":
    I1 = p_xx + p_yy + p_zz
    I2 = (p_xx * p_yy + p_xx * p_zz + p_yy * p_zz -
          (p_xy * p_xy + p_xz * p_xz + p_yz * p_yz))
    # Tensor algebra of Appendix A of Swisdak 2015.
    out = np.sqrt(1 - 4 * I2 / ((I1 - p_par) * (I1 + 3 * p_par)))
  elif measure_lower == "frobenius":
    p_ixx = p_xx - (p_par * b_x * b_x / mag_b_sq + p_perp *
                    (1 - b_x * b_x / mag_b_sq))
    p_ixy = p_xy - (p_par * b_x * b_y / mag_b_sq + p_perp *
                    (0 - b_x * b_y / mag_b_sq))
    p_ixz = p_xz - (p_par * b_x * b_z / mag_b_sq + p_perp *
                    (0 - b_x * b_z / mag_b_sq))
    p_iyy = p_yy - (p_par * b_y * b_y / mag_b_sq + p_perp *
                    (1 - b_y * b_y / mag_b_sq))
    p_iyz = p_yz - (p_par * b_y * b_z / mag_b_sq + p_perp *
                    (0 - b_y * b_z / mag_b_sq))
    p_izz = p_zz - (p_par * b_z * b_z / mag_b_sq + p_perp *
                    (1 - b_z * b_z / mag_b_sq))
    out = (np.sqrt(p_ixx**2 + 2 * p_ixy**2 + 2 * p_ixz**2 + p_iyy**2 +
                   2 * p_iyz**2 + p_izz**2) /
           np.sqrt(2 * p_perp**2 + 4 * p_par * p_perp))
  else:
    raise ValueError(
        f"Measure specified is {measure_lower:s}; it needs to be either "
        "'swisdak' or 'frobenius'")

  return grid, out


def _get_gkyl_10m_agyro(
    species_grid: list[np.ndarray],
    species_values: np.ndarray,
    field_grid: list[np.ndarray],
    field_values: np.ndarray,
    *,
    measure: str = "swisdak") -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the agyrotropy directly from raw 10-moment species and EM field
  data (whose components 3:6 are ``(Bx, By, Bz)``)."""
  p_grid, p_values = _get_pij(species_grid, species_values)
  b_values = field_values[..., 3:6]
  return _get_agyro(p_grid, p_values, field_grid, b_values, measure=measure)


# ---------------------------------------------------------------- GData verbs
def pressure(data: "GDataState",
             *,
             gas_gamma: float = 5.0 / 3,
             inplace: bool = False,
             tag: str | None = None,
             label: str | None = None) -> "GDataState":
  """Scalar pressure (trace of the pressure tensor over three) from
  10-moment fluid data.

  Args:
    data: Ten-moment fluid data; must be NumPy-backed.
    gas_gamma: Unused compatibility parameter.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the scalar pressure.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "pressure", _REASON)
  grid, values = _get_p(data.grid,
                        data.values,
                        gas_gamma=gas_gamma,
                        num_moms=10)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)


def ke(data: "GDataState",
       *,
       gas_gamma: float = 5.0 / 3,
       inplace: bool = False,
       tag: str | None = None,
       label: str | None = None) -> "GDataState":
  """Kinetic (bulk-flow) energy density from 10-moment fluid data.

  Args:
    data: Ten-moment fluid data; must be NumPy-backed.
    gas_gamma: Unused compatibility parameter.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the bulk-flow energy density.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "ke", _REASON)
  grid, values = _get_ke(data.grid,
                         data.values,
                         gas_gamma=gas_gamma,
                         num_moms=10)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)


def temp(data: "GDataState",
         *,
         gas_gamma: float = 5.0 / 3,
         inplace: bool = False,
         tag: str | None = None,
         label: str | None = None) -> "GDataState":
  """Temperature ``T = p / rho`` from 10-moment fluid data.

  Args:
    data: Ten-moment fluid data; must be NumPy-backed.
    gas_gamma: Unused compatibility parameter.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the temperature.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "temp", _REASON)
  grid, values = _get_temp(data.grid,
                           data.values,
                           gas_gamma=gas_gamma,
                           num_moms=10)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)


def sound(data: "GDataState",
          *,
          gas_gamma: float = 5.0 / 3,
          inplace: bool = False,
          tag: str | None = None,
          label: str | None = None) -> "GDataState":
  """Sound speed ``c_s = sqrt(gas_gamma * p / rho)`` from 10-moment data.

  Args:
    data: Ten-moment fluid data; must be NumPy-backed.
    gas_gamma: Adiabatic index used in the sound speed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the sound speed.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "sound", _REASON)
  grid, values = _get_sound(data.grid,
                            data.values,
                            gas_gamma=gas_gamma,
                            num_moms=10)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)


def mach(data: "GDataState",
         *,
         gas_gamma: float = 5.0 / 3,
         inplace: bool = False,
         tag: str | None = None,
         label: str | None = None) -> "GDataState":
  """Sonic Mach number ``M = |v| / c_s`` from 10-moment data.

  Args:
    data: Ten-moment fluid data; must be NumPy-backed.
    gas_gamma: Adiabatic index used in the sound speed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the Mach number.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "mach", _REASON)
  grid, values = _get_mach(data.grid,
                           data.values,
                           gas_gamma=gas_gamma,
                           num_moms=10)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)


def pxx(data: "GDataState",
        *,
        inplace: bool = False,
        tag: str | None = None,
        label: str | None = None) -> "GDataState":
  """``P_xx`` pressure-tensor component.

  Args:
    data: Ten-moment fluid data; must be NumPy-backed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of ``P_xx``.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "pxx", _REASON)
  grid, values = _get_pxx(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)


def pxy(data: "GDataState",
        *,
        inplace: bool = False,
        tag: str | None = None,
        label: str | None = None) -> "GDataState":
  """``P_xy`` pressure-tensor component.

  Args:
    data: Ten-moment fluid data; must be NumPy-backed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of ``P_xy``.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "pxy", _REASON)
  grid, values = _get_pxy(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)


def pxz(data: "GDataState",
        *,
        inplace: bool = False,
        tag: str | None = None,
        label: str | None = None) -> "GDataState":
  """``P_xz`` pressure-tensor component.

  Args:
    data: Ten-moment fluid data; must be NumPy-backed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of ``P_xz``.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "pxz", _REASON)
  grid, values = _get_pxz(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)


def pyy(data: "GDataState",
        *,
        inplace: bool = False,
        tag: str | None = None,
        label: str | None = None) -> "GDataState":
  """``P_yy`` pressure-tensor component.

  Args:
    data: Ten-moment fluid data; must be NumPy-backed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of ``P_yy``.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "pyy", _REASON)
  grid, values = _get_pyy(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)


def pyz(data: "GDataState",
        *,
        inplace: bool = False,
        tag: str | None = None,
        label: str | None = None) -> "GDataState":
  """``P_yz`` pressure-tensor component.

  Args:
    data: Ten-moment fluid data; must be NumPy-backed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of ``P_yz``.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "pyz", _REASON)
  grid, values = _get_pyz(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)


def pzz(data: "GDataState",
        *,
        inplace: bool = False,
        tag: str | None = None,
        label: str | None = None) -> "GDataState":
  """``P_zz`` pressure-tensor component.

  Args:
    data: Ten-moment fluid data; must be NumPy-backed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of ``P_zz``.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "pzz", _REASON)
  grid, values = _get_pzz(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)


def pressure_tensor(data: "GDataState",
                    *,
                    inplace: bool = False,
                    tag: str | None = None,
                    label: str | None = None) -> "GDataState":
  """Full symmetric pressure tensor
  ``(P_xx, P_xy, P_xz, P_yy, P_yz, P_zz)``.

  Args:
    data: Ten-moment fluid data; must be NumPy-backed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A six-component dataset of the symmetric pressure tensor.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "pressure_tensor", _REASON)
  grid, values = _get_pij(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)


def p_par(ptensor: "GDataState",
          bfield: "GDataState",
          *,
          inplace: bool = False,
          tag: str | None = None,
          label: str | None = None) -> "GDataState":
  """Pressure parallel to the magnetic field: ``(b . P . b) / |B|**2``.

  Args:
    ptensor: Six-component symmetric pressure tensor (Pxx, Pxy, Pxz, Pyy,
      Pyz, Pzz); must be NumPy-backed.
    bfield: Magnetic field whose first three components are (Bx, By, Bz);
      must be NumPy-backed.
    inplace: mutate and return ``ptensor`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A single-component dataset of the parallel pressure.

  Raises:
    ValueError: if either input is native modal (gkyl-backed).
  """
  _require_field_domain(ptensor, "p_par", _REASON)
  _require_field_domain(bfield, "p_par", _REASON)
  grid, values = _get_p_par(ptensor.grid, ptensor.values, bfield.grid,
                            bfield.values)
  return ptensor._result(grid, values, inplace=inplace, tag=tag, label=label)


def p_perp(ptensor: "GDataState",
           bfield: "GDataState",
           *,
           inplace: bool = False,
           tag: str | None = None,
           label: str | None = None) -> "GDataState":
  """Pressure perpendicular to the magnetic field:
  ``(P_xx + P_yy + P_zz - p_par) / 2``.

  Args:
    ptensor: Six-component symmetric pressure tensor (Pxx, Pxy, Pxz, Pyy,
      Pyz, Pzz); must be NumPy-backed.
    bfield: Magnetic field whose first three components are (Bx, By, Bz);
      must be NumPy-backed.
    inplace: mutate and return ``ptensor`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A single-component dataset of the perpendicular pressure.

  Raises:
    ValueError: if either input is native modal (gkyl-backed).
  """
  _require_field_domain(ptensor, "p_perp", _REASON)
  _require_field_domain(bfield, "p_perp", _REASON)
  grid, values = _get_p_perp(ptensor.grid, ptensor.values, bfield.grid,
                             bfield.values)
  return ptensor._result(grid, values, inplace=inplace, tag=tag, label=label)


def agyro(ptensor: "GDataState",
          bfield: "GDataState",
          *,
          measure: str = "frobenius",
          inplace: bool = False,
          tag: str | None = None,
          label: str | None = None) -> "GDataState":
  """Agyrotropy from a pressure tensor and an EM field.

  Measures how far the pressure tensor departs from gyrotropy about the
  local magnetic field. The field's first three components are used as the
  magnetic field direction.

  Args:
    ptensor: Six-component symmetric pressure tensor (Pxx, Pxy, Pxz, Pyy,
      Pyz, Pzz); must be NumPy-backed.
    bfield: Magnetic field whose first three components are (Bx, By, Bz);
      must be NumPy-backed.
    measure: 'frobenius' (Frobenius norm of the agyrotropic part of the
      pressure tensor) or 'swisdak' (the Q measure of Swisdak 2015).
      Case-insensitive.
    inplace: mutate and return ``ptensor`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A single-component dataset of the agyrotropy.

  Raises:
    ValueError: if either input is native modal (gkyl-backed), or
      ``measure`` is not 'frobenius' or 'swisdak'.
  """
  _require_field_domain(ptensor, "agyro", _AGYRO_REASON)
  _require_field_domain(bfield, "agyro", _AGYRO_REASON)
  grid, values = _get_agyro(ptensor.grid,
                            ptensor.values,
                            bfield.grid,
                            bfield.values,
                            measure=measure)
  return ptensor._result(grid, values, inplace=inplace, tag=tag, label=label)


def mom_agyro(species: "GDataState",
              field: "GDataState",
              *,
              measure: str = "frobenius",
              inplace: bool = False,
              tag: str | None = None,
              label: str | None = None) -> "GDataState":
  """Agyrotropy from raw 10-moment species data and an EM field.

  Convenience wrapper that first forms the pressure tensor from raw
  10-moment species data and extracts the magnetic field (components 3:6)
  from a Gkeyll EM field, then computes the agyrotropy.

  Args:
    species: Raw 10-moment fluid data for a single species (density,
      momentum, and the six pressure-tensor moments); must be NumPy-backed.
    field: Gkeyll EM field whose components 3:6 are the magnetic field (Bx,
      By, Bz); must be NumPy-backed.
    measure: 'frobenius' (Frobenius norm of the agyrotropic part of the
      pressure tensor) or 'swisdak' (the Q measure of Swisdak 2015).
      Case-insensitive.
    inplace: mutate and return ``species`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A single-component dataset of the agyrotropy.

  Raises:
    ValueError: if either input is native modal (gkyl-backed), or
      ``measure`` is not 'frobenius' or 'swisdak'.
  """
  _require_field_domain(species, "mom_agyro", _AGYRO_REASON)
  _require_field_domain(field, "mom_agyro", _AGYRO_REASON)
  grid, values = _get_gkyl_10m_agyro(species.grid,
                                     species.values,
                                     field.grid,
                                     field.values,
                                     measure=measure)
  return species._result(grid, values, inplace=inplace, tag=tag, label=label)


VARIABLES = {
    "density": density,
    "xvel": xvel,
    "yvel": yvel,
    "zvel": zvel,
    "vel": vel,
    "pressure": pressure,
    "ke": ke,
    "temp": temp,
    "sound": sound,
    "mach": mach,
    "pressureTensor": pressure_tensor,
    "pxx": pxx,
    "pxy": pxy,
    "pxz": pxz,
    "pyy": pyy,
    "pyz": pyz,
    "pzz": pzz,
}
