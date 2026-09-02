"""Ideal-MHD diagnostics -- the five-moment set (density/velocity) plus the
magnetic field, magnetic pressure, thermal pressure, temperature, sound
speed, and Mach number.

MHD moment data is laid out ``[rho, mx, my, mz, E, Bx, By, Bz]``: components
0:4 are shared with the 5-moment layout (density and momentum), so density
and velocity are reused from :mod:`postgkyl.diagnostics.moments.five_moment`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...gdatastate.guards import require_field_domain as _require_field_domain
from .five_moment import _get_density, _get_vx, _get_vy, _get_vz
from .five_moment import density, xvel, yvel, zvel, vel

if TYPE_CHECKING:
  from ...gdatastate.gdatastate import GDataState
# end

_REASON = ("extracting primitive variables from raw DG coefficients would "
    "mix basis functions")


# --------------------------------------------------------- array-level math
def _get_mhd_Bx(grid: list[np.ndarray],
    values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Extract the x magnetic-field component (component 5 of MHD data)."""
  return list(grid), values[..., 5, np.newaxis]
# end


def _get_mhd_By(grid: list[np.ndarray],
    values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Extract the y magnetic-field component (component 6 of MHD data)."""
  return list(grid), values[..., 6, np.newaxis]
# end


def _get_mhd_Bz(grid: list[np.ndarray],
    values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Extract the z magnetic-field component (component 7 of MHD data)."""
  return list(grid), values[..., 7, np.newaxis]
# end


def _get_mhd_Bi(grid: list[np.ndarray],
    values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Extract the magnetic-field vector ``(Bx, By, Bz)`` (components 5:8)."""
  return list(grid), values[..., 5:8]
# end


def _get_mhd_mag_p(grid: list[np.ndarray], values: np.ndarray, *,
    mu_0: float = 1.0) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the magnetic pressure
  ``p_B = 0.5 * (Bx**2 + By**2 + Bz**2) / mu_0``."""
  _, Bx = _get_mhd_Bx(grid, values)
  _, By = _get_mhd_By(grid, values)
  _, Bz = _get_mhd_Bz(grid, values)
  return list(grid), 0.5 * (Bx**2 + By**2 + Bz**2) / mu_0
# end


def _get_mhd_p(grid: list[np.ndarray], values: np.ndarray, *,
    gas_gamma: float = 5.0 / 3, mu_0: float = 1.0,
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the thermal (gas) pressure.

  ``p = (gas_gamma - 1) * (E - 0.5*rho*|v|**2 - p_B)``.
  """
  _, rho = _get_density(grid, values)
  _, vx = _get_vx(grid, values)
  _, vy = _get_vy(grid, values)
  _, vz = _get_vz(grid, values)
  _, mag_p = _get_mhd_mag_p(grid, values, mu_0=mu_0)

  out_values = (gas_gamma - 1) * (
      values[..., 4, np.newaxis] - 0.5 * rho * (vx**2 + vy**2 + vz**2) - mag_p)
  return list(grid), out_values
# end


def _get_mhd_temp(grid: list[np.ndarray], values: np.ndarray, *,
    gas_gamma: float = 5.0 / 3, mu_0: float = 1.0,
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the temperature ``T = p / rho``."""
  _, rho = _get_density(grid, values)
  _, pr = _get_mhd_p(grid, values, gas_gamma=gas_gamma, mu_0=mu_0)
  return list(grid), pr / rho
# end


def _get_mhd_sound(grid: list[np.ndarray], values: np.ndarray, *,
    gas_gamma: float = 5.0 / 3, mu_0: float = 1.0,
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the sound speed ``c_s = sqrt(gas_gamma * p / rho)``."""
  _, rho = _get_density(grid, values)
  _, pr = _get_mhd_p(grid, values, gas_gamma=gas_gamma, mu_0=mu_0)
  return list(grid), np.sqrt(gas_gamma * pr / rho)
# end


def _get_mhd_mach(grid: list[np.ndarray], values: np.ndarray, *,
    gas_gamma: float = 5.0 / 3, mu_0: float = 1.0,
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the sonic Mach number ``M = |v| / c_s``."""
  _, vx = _get_vx(grid, values)
  _, vy = _get_vy(grid, values)
  _, vz = _get_vz(grid, values)
  _, cs = _get_mhd_sound(grid, values, gas_gamma=gas_gamma, mu_0=mu_0)
  return list(grid), np.sqrt(vx**2 + vy**2 + vz**2) / cs
# end


# ---------------------------------------------------------------- GData verbs
def bx(data: "GDataState", *, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """x magnetic-field component (component 5 of MHD data).

  Args:
    data: MHD conserved variables; must be NumPy-backed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of ``Bx``.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "bx", _REASON)
  grid, values = _get_mhd_Bx(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def by(data: "GDataState", *, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """y magnetic-field component (component 6 of MHD data).

  Args:
    data: MHD conserved variables; must be NumPy-backed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of ``By``.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "by", _REASON)
  grid, values = _get_mhd_By(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def bz(data: "GDataState", *, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """z magnetic-field component (component 7 of MHD data).

  Args:
    data: MHD conserved variables; must be NumPy-backed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of ``Bz``.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "bz", _REASON)
  grid, values = _get_mhd_Bz(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def bi(data: "GDataState", *, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """Magnetic-field vector ``(Bx, By, Bz)`` (components 5:8).

  Args:
    data: MHD conserved variables; must be NumPy-backed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A three-component dataset of the magnetic field.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "bi", _REASON)
  grid, values = _get_mhd_Bi(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def mag_pressure(data: "GDataState", *, mu_0: float = 1.0,
    inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Magnetic pressure ``p_B = 0.5 * (Bx**2 + By**2 + Bz**2) / mu_0``.

  Args:
    data: MHD conserved variables; must be NumPy-backed.
    mu_0: Vacuum permeability.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the magnetic pressure.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "mag_pressure", _REASON)
  grid, values = _get_mhd_mag_p(data.grid, data.values, mu_0=mu_0)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def pressure(data: "GDataState", *, gas_gamma: float = 5.0 / 3,
    mu_0: float = 1.0, inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Thermal (gas) pressure
  ``p = (gas_gamma - 1) * (E - 0.5*rho*|v|**2 - p_B)``.

  Args:
    data: MHD conserved variables; must be NumPy-backed.
    gas_gamma: Adiabatic index.
    mu_0: Vacuum permeability used in the magnetic energy.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the thermal pressure.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "pressure", _REASON)
  grid, values = _get_mhd_p(data.grid, data.values, gas_gamma=gas_gamma,
      mu_0=mu_0)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def temp(data: "GDataState", *, gas_gamma: float = 5.0 / 3,
    mu_0: float = 1.0, inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Temperature ``T = p / rho``.

  Args:
    data: MHD conserved variables; must be NumPy-backed.
    gas_gamma: Adiabatic index.
    mu_0: Vacuum permeability used to compute the pressure.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the temperature.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "temp", _REASON)
  grid, values = _get_mhd_temp(data.grid, data.values, gas_gamma=gas_gamma,
      mu_0=mu_0)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def sound(data: "GDataState", *, gas_gamma: float = 5.0 / 3,
    mu_0: float = 1.0, inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Sound speed ``c_s = sqrt(gas_gamma * p / rho)``.

  Args:
    data: MHD conserved variables; must be NumPy-backed.
    gas_gamma: Adiabatic index.
    mu_0: Vacuum permeability used to compute the pressure.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the sound speed.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "sound", _REASON)
  grid, values = _get_mhd_sound(data.grid, data.values, gas_gamma=gas_gamma,
      mu_0=mu_0)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def mach(data: "GDataState", *, gas_gamma: float = 5.0 / 3,
    mu_0: float = 1.0, inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Sonic Mach number ``M = |v| / c_s``.

  Args:
    data: MHD conserved variables; must be NumPy-backed.
    gas_gamma: Adiabatic index.
    mu_0: Vacuum permeability used to compute the pressure.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the Mach number.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "mach", _REASON)
  grid, values = _get_mhd_mach(data.grid, data.values, gas_gamma=gas_gamma,
      mu_0=mu_0)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


VARIABLES = {
    "density": density, "xvel": xvel, "yvel": yvel, "zvel": zvel, "vel": vel,
    "Bx": bx, "By": by, "Bz": bz, "Bi": bi,
    "magpressure": mag_pressure,
    "pressure": pressure, "temp": temp, "sound": sound, "mach": mach,
}
