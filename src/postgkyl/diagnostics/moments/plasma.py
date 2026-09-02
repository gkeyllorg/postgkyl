"""Plasma parameters: field magnitude, thermal/Alfven velocity, cyclotron and
plasma frequency, inertial length, Debye length, gyroradius, plasma beta.

These never had a verb layer of their own (only the array math lived in the
old ``models`` package) -- every public function here is a fresh GData-facing
wrapper (species/field datasets in, ``GDataState`` out) over that moved
array math.

The old ``postgkeyll.tools.params`` functions read ``mass``/``charge``/
``mu_0``/``epsilon_0`` from a ``GData.ctx`` dict, falling back to a keyword
argument when the context held nothing. These are pure keyword-only
arguments instead -- no ctx, no fallback chain. A consequence of dropping the
GData/ctx duality is that a few old parameters were never anything but ctx
lookups (unused otherwise) and are dropped here because keeping them would
misstate what the function actually needs (Doctrine IV): ``omegaC`` does not
take ``species`` (only ``field`` values were ever used), ``omegaP``/``d``/
``lambdaD`` do not take ``field`` (only ``species`` values were ever used),
and ``rho`` drops the never-referenced ``epsilon_0`` parameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ... import numerics
from ...gdatastate.guards import require_field_domain as _require_field_domain
from .five_moment import _get_density, _get_temp
from .mhd import _get_mhd_temp

if TYPE_CHECKING:
  from ...gdatastate.gdatastate import GDataState
# end

_REASON = "computing plasma parameters from raw DG coefficients would mix basis functions"


# --------------------------------------------------------- array-level math
def _get_magB(field_grid: list[np.ndarray],
    field_values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the magnitude of the magnetic field ``|B|``.

  Args:
    field_grid: EM field grid.
    field_values: EM field array laid out ``[Ex, Ey, Ez, Bx, By, Bz, ...]``;
      components 3:6 are used.

  Returns:
    ``(grid, values)`` holding ``|B| = sqrt(Bx**2 + By**2 + Bz**2)``.
  """
  b_values = field_values[..., 3:6]
  _, mag_B_sq = numerics.mag_sq(field_grid, b_values)
  return list(field_grid), np.sqrt(mag_B_sq)
# end


def _get_vt(species_grid: list[np.ndarray], species_values: np.ndarray, *,
    gas_gamma: float = 5.0 / 3.0, num_moms: int | None = None,
    mass: float = 1.0, mu_0: float = 1.0, sqrt2: bool = True,
    mhd: bool = False) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the thermal velocity ``v_th = sqrt(2 T/m)`` (or ``sqrt(T/m)``
  when ``sqrt2`` is ``False``) of a species.

  Args:
    species_grid: Species moment grid.
    species_values: Species moment array (5- or 10-moment, or MHD when
      ``mhd=True``).
    gas_gamma: Adiabatic index used when computing the temperature/pressure.
    num_moms: Number of moments (5 or 10); inferred when ``None``.
    mass: Particle mass.
    mu_0: Vacuum permeability, forwarded to the MHD temperature when
      ``mhd=True``.
    sqrt2: If ``True`` (default), scale the result by ``sqrt(2)``.
    mhd: If ``True``, compute the temperature from MHD moments; otherwise
      use the fluid moments.

  Returns:
    ``(grid, values)`` holding the thermal velocity field.
  """
  if mhd:
    out_grid, temp = _get_mhd_temp(species_grid, species_values,
        gas_gamma=gas_gamma, mu_0=mu_0)
  # end
  else:
    out_grid, temp = _get_temp(species_grid, species_values,
        gas_gamma=gas_gamma, num_moms=num_moms)
  # end

  out_values = np.sqrt(temp / mass)
  if sqrt2:
    out_values = out_values * np.sqrt(2.0)
  # end

  return out_grid, out_values
# end


def _get_vA(species_grid: list[np.ndarray], species_values: np.ndarray,
    field_grid: list[np.ndarray], field_values: np.ndarray, *,
    mu_0: float = 1.0) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the Alfven velocity ``v_A = |B| / sqrt(mu_0 * rho)``.

  Fluid moment data already includes the mass factor in the density.
  """
  _, magB = _get_magB(field_grid, field_values)
  out_grid, rho = _get_density(species_grid, species_values)
  return out_grid, magB / np.sqrt(mu_0 * rho)
# end


def _get_omegaC(field_grid: list[np.ndarray], field_values: np.ndarray, *,
    mass: float = 1.0, charge: float = 1.0,
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the cyclotron (gyro) frequency ``omega_c = |q| * |B| / m``."""
  out_grid, magB = _get_magB(field_grid, field_values)
  return out_grid, abs(charge) * magB / mass
# end


def _get_omegaP(species_grid: list[np.ndarray], species_values: np.ndarray, *,
    mass: float = 1.0, charge: float = 1.0, epsilon_0: float = 1.0,
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the plasma frequency
  ``omega_p = sqrt(q**2 * n / (m**2 * epsilon_0))``.

  Fluid moment data already includes the mass factor in the density.
  """
  out_grid, rho = _get_density(species_grid, species_values)
  qbym2 = charge**2 / mass**2
  return out_grid, np.sqrt(qbym2 * rho / epsilon_0)
# end


def _get_d(species_grid: list[np.ndarray], species_values: np.ndarray, *,
    mass: float = 1.0, charge: float = 1.0, epsilon_0: float = 1.0,
    mu_0: float = 1.0) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the inertial (skin-depth) length ``d = c / omega_p``, with
  ``c = 1 / sqrt(epsilon_0 * mu_0)``."""
  out_grid, omegaP = _get_omegaP(species_grid, species_values, mass=mass,
      charge=charge, epsilon_0=epsilon_0)
  light_speed = 1.0 / np.sqrt(epsilon_0 * mu_0)
  return out_grid, light_speed / omegaP
# end


def _get_lambdaD(species_grid: list[np.ndarray], species_values: np.ndarray, *,
    gas_gamma: float = 5.0 / 3.0, num_moms: int | None = None,
    mass: float = 1.0, charge: float = 1.0, epsilon_0: float = 1.0,
    mu_0: float = 1.0, sqrt2: bool = True,
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the Debye length ``lambda_D = v_th / omega_p``.

  When ``sqrt2`` is ``True`` the extra ``sqrt(2)`` factor carried by
  ``v_th`` is divided back out, so the conventional Debye length is
  returned.
  """
  _, omegaP = _get_omegaP(species_grid, species_values, mass=mass,
      charge=charge, epsilon_0=epsilon_0)
  out_grid, vt = _get_vt(species_grid, species_values, gas_gamma=gas_gamma,
      num_moms=num_moms, mass=mass, mu_0=mu_0, sqrt2=sqrt2)
  out_values = vt / omegaP
  if sqrt2:
    out_values = out_values / np.sqrt(2.0)
  # end

  return out_grid, out_values
# end


def _get_rho(species_grid: list[np.ndarray], species_values: np.ndarray,
    field_grid: list[np.ndarray], field_values: np.ndarray, *,
    gas_gamma: float = 5.0 / 3.0, num_moms: int | None = None,
    mass: float = 1.0, charge: float = 1.0, mu_0: float = 1.0,
    sqrt2: bool = True) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the gyroradius (Larmor radius) ``rho = v_th / omega_c``.

  When ``sqrt2`` is ``False`` the result is multiplied by ``sqrt(2)`` so the
  gyroradius stays consistent with a ``sqrt(2)``-scaled thermal velocity.
  """
  _, omegaC = _get_omegaC(field_grid, field_values, mass=mass, charge=charge)
  out_grid, vt = _get_vt(species_grid, species_values, gas_gamma=gas_gamma,
      num_moms=num_moms, mass=mass, mu_0=mu_0, sqrt2=sqrt2)

  out_values = vt / omegaC
  if not sqrt2:
    out_values = out_values * np.sqrt(2.0)
  # end

  return out_grid, out_values
# end


def _get_beta(species_grid: list[np.ndarray], species_values: np.ndarray,
    field_grid: list[np.ndarray], field_values: np.ndarray, *,
    gas_gamma: float = 5.0 / 3.0, num_moms: int | None = None,
    mass: float = 1.0, mu_0: float = 1.0, sqrt2: bool = True,
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the plasma beta ``v_th**2 / v_A**2``.

  When ``sqrt2`` is ``False`` the result is multiplied by ``2`` to account
  for the missing ``sqrt(2)`` factor in the thermal velocity.
  """
  _, v_A = _get_vA(species_grid, species_values, field_grid, field_values,
      mu_0=mu_0)
  out_grid, vt = _get_vt(species_grid, species_values, gas_gamma=gas_gamma,
      num_moms=num_moms, mass=mass, mu_0=mu_0, sqrt2=sqrt2)
  out_values = vt**2 / v_A**2
  if not sqrt2:
    out_values = out_values * 2.0
  # end

  return out_grid, out_values
# end


# ---------------------------------------------------------------- GData verbs
def magB(field: "GDataState", *, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """Magnitude of the magnetic field ``|B|``.

  Args:
    field: EM field data (components 3:6 are ``Bx, By, Bz``); must be
      NumPy-backed.
    inplace: mutate and return ``field`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A single-component dataset of ``|B|``.

  Raises:
    ValueError: if ``field`` is native modal (gkyl-backed).
  """
  _require_field_domain(field, "magB", _REASON)
  grid, values = _get_magB(field.grid, field.values)
  return field._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def vt(species: "GDataState", *, gas_gamma: float = 5.0 / 3.0,
    num_moms: int | None = None, mass: float = 1.0, mu_0: float = 1.0,
    sqrt2: bool = True, mhd: bool = False, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """Thermal velocity ``v_th = sqrt(2 T/m)`` of a species.

  Args:
    species: Species moment data (5- or 10-moment, or MHD when ``mhd=True``);
      must be NumPy-backed.
    gas_gamma: Adiabatic index used when computing the temperature/pressure.
    num_moms: Number of moments (5 or 10); inferred when ``None``.
    mass: Particle mass.
    mu_0: Vacuum permeability, forwarded to the MHD temperature when
      ``mhd=True``.
    sqrt2: If ``True`` (default), scale the result by ``sqrt(2)``.
    mhd: If ``True``, compute the temperature from MHD moments; otherwise
      use the fluid moments.
    inplace: mutate and return ``species`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A single-component dataset of the thermal velocity.

  Raises:
    ValueError: if ``species`` is native modal (gkyl-backed).
  """
  _require_field_domain(species, "vt", _REASON)
  grid, values = _get_vt(species.grid, species.values, gas_gamma=gas_gamma,
      num_moms=num_moms, mass=mass, mu_0=mu_0, sqrt2=sqrt2, mhd=mhd)
  return species._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def vA(species: "GDataState", field: "GDataState", *, mu_0: float = 1.0,
    inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Alfven velocity ``v_A = |B| / sqrt(mu_0 * rho)``.

  Args:
    species: Species moment data providing the density; must be
      NumPy-backed.
    field: EM field data providing ``|B|``; must be NumPy-backed.
    mu_0: Vacuum permeability.
    inplace: mutate and return ``species`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A single-component dataset of the Alfven velocity.

  Raises:
    ValueError: if either input is native modal (gkyl-backed).
  """
  _require_field_domain(species, "vA", _REASON)
  _require_field_domain(field, "vA", _REASON)
  grid, values = _get_vA(species.grid, species.values, field.grid,
      field.values, mu_0=mu_0)
  return species._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def omegaC(field: "GDataState", *, mass: float = 1.0, charge: float = 1.0,
    inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Cyclotron (gyro) frequency ``omega_c = |q| * |B| / m``.

  Args:
    field: EM field data providing ``Bx, By, Bz``; must be NumPy-backed.
    mass: Particle mass.
    charge: Particle charge; only its magnitude affects the result.
    inplace: Mutate and return ``field`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the cyclotron frequency.

  Raises:
    ValueError: if ``field`` is native modal (gkyl-backed).
  """
  _require_field_domain(field, "omegaC", _REASON)
  grid, values = _get_omegaC(field.grid, field.values, mass=mass, charge=charge)
  return field._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def omegaP(species: "GDataState", *, mass: float = 1.0, charge: float = 1.0,
    epsilon_0: float = 1.0, inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Plasma frequency ``omega_p = sqrt(q**2 * n / (m**2 * epsilon_0))``.

  Args:
    species: Fluid moment data providing mass density; must be NumPy-backed.
    mass: Particle mass.
    charge: Particle charge.
    epsilon_0: Vacuum permittivity.
    inplace: Mutate and return ``species`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the plasma frequency.

  Raises:
    ValueError: if ``species`` is native modal (gkyl-backed).
  """
  _require_field_domain(species, "omegaP", _REASON)
  grid, values = _get_omegaP(species.grid, species.values, mass=mass,
      charge=charge, epsilon_0=epsilon_0)
  return species._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def d(species: "GDataState", *, mass: float = 1.0, charge: float = 1.0,
    epsilon_0: float = 1.0, mu_0: float = 1.0, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """Inertial (skin-depth) length ``d = c / omega_p``.

  Args:
    species: Fluid moment data providing mass density; must be NumPy-backed.
    mass: Particle mass.
    charge: Particle charge.
    epsilon_0: Vacuum permittivity.
    mu_0: Vacuum permeability used in ``c = 1/sqrt(epsilon_0*mu_0)``.
    inplace: Mutate and return ``species`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the inertial length.

  Raises:
    ValueError: if ``species`` is native modal (gkyl-backed).
  """
  _require_field_domain(species, "d", _REASON)
  grid, values = _get_d(species.grid, species.values, mass=mass,
      charge=charge, epsilon_0=epsilon_0, mu_0=mu_0)
  return species._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def lambdaD(species: "GDataState", *, gas_gamma: float = 5.0 / 3.0,
    num_moms: int | None = None, mass: float = 1.0, charge: float = 1.0,
    epsilon_0: float = 1.0, mu_0: float = 1.0, sqrt2: bool = True,
    inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Debye length ``lambda_D = v_th / omega_p``.

  Args:
    species: Fluid moment data; must be NumPy-backed.
    gas_gamma: Adiabatic index used to compute thermal velocity.
    num_moms: Number of fluid moments (5 or 10); inferred when ``None``.
    mass: Particle mass.
    charge: Particle charge.
    epsilon_0: Vacuum permittivity used in the plasma frequency.
    mu_0: Vacuum permeability forwarded to thermal-velocity calculation.
    sqrt2: Use the ``sqrt(2*T/m)`` thermal-velocity convention; the extra
      factor is removed from the returned conventional Debye length.
    inplace: Mutate and return ``species`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the Debye length.

  Raises:
    ValueError: if ``species`` is native modal (gkyl-backed).
  """
  _require_field_domain(species, "lambdaD", _REASON)
  grid, values = _get_lambdaD(species.grid, species.values,
      gas_gamma=gas_gamma, num_moms=num_moms, mass=mass, charge=charge,
      epsilon_0=epsilon_0, mu_0=mu_0, sqrt2=sqrt2)
  return species._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def rho(species: "GDataState", field: "GDataState", *,
    gas_gamma: float = 5.0 / 3.0, num_moms: int | None = None,
    mass: float = 1.0, charge: float = 1.0, mu_0: float = 1.0,
    sqrt2: bool = True, inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Gyroradius (Larmor radius) ``rho = v_th / omega_c``.

  Args:
    species: Fluid moment data used for thermal velocity; must be
      NumPy-backed.
    field: EM field data used for cyclotron frequency; must be NumPy-backed.
    gas_gamma: Adiabatic index used to compute thermal velocity.
    num_moms: Number of fluid moments (5 or 10); inferred when ``None``.
    mass: Particle mass.
    charge: Particle charge.
    mu_0: Vacuum permeability forwarded to thermal-velocity calculation.
    sqrt2: Select the thermal-velocity convention; the result is normalized
      to the ``sqrt(2*T/m)`` convention either way.
    inplace: Mutate and return ``species`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the gyroradius.

  Raises:
    ValueError: if either input is native modal (gkyl-backed).
  """
  _require_field_domain(species, "rho", _REASON)
  _require_field_domain(field, "rho", _REASON)
  grid, values = _get_rho(species.grid, species.values, field.grid,
      field.values, gas_gamma=gas_gamma, num_moms=num_moms, mass=mass,
      charge=charge, mu_0=mu_0, sqrt2=sqrt2)
  return species._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def beta(species: "GDataState", field: "GDataState", *,
    gas_gamma: float = 5.0 / 3.0, num_moms: int | None = None,
    mass: float = 1.0, mu_0: float = 1.0, sqrt2: bool = True,
    inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Plasma beta ``v_th**2 / v_A**2``.

  Args:
    species: Fluid moment data used for thermal velocity and density; must
      be NumPy-backed.
    field: EM field data used for Alfven velocity; must be NumPy-backed.
    gas_gamma: Adiabatic index used to compute thermal velocity.
    num_moms: Number of fluid moments (5 or 10); inferred when ``None``.
    mass: Particle mass.
    mu_0: Vacuum permeability.
    sqrt2: Select the thermal-velocity convention; the result is normalized
      to the ``sqrt(2*T/m)`` convention either way.
    inplace: Mutate and return ``species`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of plasma beta.

  Raises:
    ValueError: if either input is native modal (gkyl-backed).
  """
  _require_field_domain(species, "beta", _REASON)
  _require_field_domain(field, "beta", _REASON)
  grid, values = _get_beta(species.grid, species.values, field.grid,
      field.values, gas_gamma=gas_gamma, num_moms=num_moms, mass=mass,
      mu_0=mu_0, sqrt2=sqrt2)
  return species._result(grid, values, inplace=inplace, tag=tag, label=label)
# end
