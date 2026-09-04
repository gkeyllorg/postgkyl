"""Multi-species diagnostics: energy-balance decomposition and current
accumulation.

``energetics`` separates a two-species (electron + ion) fluid/field system
into its constituent energy components; ``accumulate_current`` scales a
single species' moment data by its charge (or charge-to-mass ratio) so that
several species can be summed into a total current.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ... import numerics
from ...gdatastate.guards import require_field_domain as _require_field_domain
from .five_moment import _get_ke, _get_p

if TYPE_CHECKING:
  from ...gdatastate.gdatastate import GDataState

_REASON = "decomposing energy from raw DG coefficients would mix basis functions"


# --------------------------------------------------------- array-level math
def _energetics(
    elc_grid: list[np.ndarray],
    elc_values: np.ndarray,
    ion_grid: list[np.ndarray],
    ion_values: np.ndarray,
    field_grid: list[np.ndarray],
    field_values: np.ndarray,
    *,
    gas_gamma: float = 5.0 / 3,
    num_moms: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
  """Separate a two-species plasma's energy into its constituent parts.

  Args:
    elc_grid: Electron moment grid.
    elc_values: Electron fluid moment array.
    ion_grid: Ion moment grid.
    ion_values: Ion fluid moment array.
    field_grid: EM field grid.
    field_values: EM field array laid out ``[Ex, Ey, Ez, Bx, By, Bz]``.
    gas_gamma: Adiabatic index, forwarded to the pressure/kinetic-energy
      calculation for both species.
    num_moms: Number of moments (5 or 10) for both species; inferred from
      the component count when ``None``.

  Returns:
    ``(grid, values)`` with a 7-component field:
    ``(electron thermal, electron kinetic, ion thermal, ion kinetic,
    electric, magnetic, total)``.
  """
  out = np.zeros(field_values.shape[:-1] + (7, ))

  _, pre = _get_p(elc_grid, elc_values, gas_gamma=gas_gamma, num_moms=num_moms)
  _, kee = _get_ke(elc_grid, elc_values, gas_gamma=gas_gamma, num_moms=num_moms)
  _, pri = _get_p(ion_grid, ion_values, gas_gamma=gas_gamma, num_moms=num_moms)
  _, kei = _get_ke(ion_grid, ion_values, gas_gamma=gas_gamma, num_moms=num_moms)
  _, esq = numerics.mag_sq(field_grid, field_values, coords="0:3")
  _, bsq = numerics.mag_sq(field_grid, field_values, coords="3:6")

  out[..., 0] = np.squeeze(pre)
  out[..., 1] = np.squeeze(kee)
  out[..., 2] = np.squeeze(pri)
  out[..., 3] = np.squeeze(kei)
  out[..., 4] = np.squeeze(esq / 2.0)
  out[..., 5] = np.squeeze(bsq / 2.0)
  out[..., 6] = np.squeeze(pre + kee + pri + kei + esq / 2.0 + bsq / 2.0)

  return list(field_grid), out


def _accumulate_current(
    grid: list[np.ndarray],
    values: np.ndarray,
    *,
    qbym: bool = False,
    charge: float | None = None,
    mass: float | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
  """Scale a species' moment data into its contribution to the current.

  Args:
    grid: Species moment grid.
    values: Species moment array.
    qbym: If ``True``, scale by the charge-to-mass ratio ``charge / mass``
      (appropriate for fluid moment data, which already carries a mass
      factor in the density); otherwise scale by ``-1.0``.
    charge: Particle charge, required when ``qbym`` is ``True``.
    mass: Particle mass, required (and must be nonzero) when ``qbym`` is
      ``True``.

  Returns:
    ``(grid, values)`` holding the current contribution.
  """
  if qbym and mass and charge is not None:
    factor = charge / mass
  else:
    factor = -1.0

  return list(grid), factor * values


# ---------------------------------------------------------------- GData verbs
def energetics(elc: "GDataState",
               ion: "GDataState",
               field: "GDataState",
               *,
               gas_gamma: float = 5.0 / 3,
               num_moms: int | None = None,
               inplace: bool = False,
               tag: str | None = None,
               label: str | None = None) -> "GDataState":
  """Decompose energy (kinetic, thermal, EM) for a two-species plasma.

  Splits the plasma energy into its constituent parts for a two-species
  (electron/ion) plasma plus an EM field. The result carries the EM
  field's grid and metadata and has seven components, in order:

  0. electron thermal energy
  1. electron kinetic energy
  2. ion thermal energy
  3. ion kinetic energy
  4. electric field energy (|E|^2 / 2)
  5. magnetic field energy (|B|^2 / 2)
  6. total energy (sum of the above)

  Args:
    elc: Electron fluid moments (used to compute thermal pressure and
      kinetic energy); must be NumPy-backed.
    ion: Ion fluid moments (used to compute thermal pressure and kinetic
      energy); must be NumPy-backed.
    field: EM field whose components 0:3 are the electric field and 3:6
      are the magnetic field; its grid/metadata are carried to the output.
      Must be NumPy-backed.
    gas_gamma: Adiabatic index, forwarded to the pressure/kinetic-energy
      calculation for both species.
    num_moms: Number of moments (5 or 10) for both species; inferred from
      the component count when ``None``.
    inplace: mutate and return ``field`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A seven-component dataset of the energy decomposition.

  Raises:
    ValueError: if any input is native modal (gkyl-backed).
  """
  _require_field_domain(elc, "energetics", _REASON)
  _require_field_domain(ion, "energetics", _REASON)
  _require_field_domain(field, "energetics", _REASON)
  grid, values = _energetics(elc.grid,
                             elc.values,
                             ion.grid,
                             ion.values,
                             field.grid,
                             field.values,
                             gas_gamma=gas_gamma,
                             num_moms=num_moms)
  return field._result(grid, values, inplace=inplace, tag=tag, label=label)


def accumulate_current(data: "GDataState",
                       *,
                       qbym: bool = False,
                       charge: float | None = None,
                       mass: float | None = None,
                       inplace: bool = False,
                       tag: str | None = None,
                       label: str | None = None) -> "GDataState":
  """Accumulate current from species moments.

  Scales the species' momentum/flow moments by a per-species factor to
  form its contribution to the current. By default the factor is ``-1.0``;
  with ``qbym=True`` (and ``charge``/``mass`` given) the charge/mass ratio
  is used instead. Should be used with ``qbym=True`` for fluid data.

  Args:
    data: A species dataset carrying the flow/momentum moments to scale;
      must be NumPy-backed.
    qbym: When True, scale by the charge-to-mass ratio (q/m); otherwise
      scale by ``-1.0``. Set True for fluid data.
    charge: Particle charge, required when ``qbym`` is True.
    mass: Particle mass, required (and must be nonzero) when ``qbym`` is
      True.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A dataset of the scaled current contribution.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed); if ``qbym`` is
      True and ``charge``/``mass`` are not both given (a nonzero ``mass``).
  """
  if data.backend == "gkyl":
    raise ValueError(
        "accumulate_current operates on interpolated (NumPy) values; call "
        ".interpolate() first -- scaling raw DG coefficients by a per-species "
        "factor is still valid numerically, but this verb is field-domain "
        "only.")
  if qbym and (charge is None or not mass):
    raise ValueError(
        "accumulate_current: qbym=True requires both 'charge' and a "
        f"nonzero 'mass' -- got charge={charge!r}, mass={mass!r}.")
  grid, values = _accumulate_current(data.grid,
                                     data.values,
                                     qbym=qbym,
                                     charge=charge,
                                     mass=mass)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
