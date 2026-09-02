"""Five-moment (Euler) diagnostics -- density, velocity, pressure, temperature,
sound speed, Mach number.

Fluid moment data is laid out ``[rho, rho*vx, rho*vy, rho*vz, E, ...]``: the
first four components are shared with 10-moment/MHD data, and ``pressure``/
``ke``/``temp``/``sound``/``mach`` additionally accept 10-moment data
(``num_moms=10``), inferring which layout applies from the number of
components when ``num_moms`` is not given.

Each public function takes a ``GDataState`` and returns one (funneling
through ``_result``); the array-level math is kept in module-private
``_get_*`` helpers, copied verbatim from the pre-restructure ``models`` /
``operations`` layers (06/08) so ``ten_moment``/``mhd``/``plasma``/``multispecies``
can compose the same formulas without re-deriving them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..gdatastate.guards import require_field_domain as _require_field_domain

if TYPE_CHECKING:
  from ..gdatastate.gdatastate import GDataState
# end

_REASON = ("extracting primitive variables from raw DG coefficients would "
    "mix basis functions")


# --------------------------------------------------------- array-level math
def _get_density(grid: list[np.ndarray],
    values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Extract the (mass) density from fluid moment data.

  The density is component 0 of the moment array.

  Args:
    grid: Nodal coordinate arrays, one per spatial dimension.
    values: Moment array whose last axis holds the conserved variables.

  Returns:
    ``(grid, values)`` with the density as a single trailing component.
  """
  return list(grid), values[..., 0, np.newaxis]
# end


def _get_vx(grid: list[np.ndarray],
    values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Extract the x velocity: x momentum (component 1) over density."""
  _, rho = _get_density(grid, values)
  return list(grid), values[..., 1, np.newaxis] / rho
# end


def _get_vy(grid: list[np.ndarray],
    values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Extract the y velocity: y momentum (component 2) over density."""
  _, rho = _get_density(grid, values)
  return list(grid), values[..., 2, np.newaxis] / rho
# end


def _get_vz(grid: list[np.ndarray],
    values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Extract the z velocity: z momentum (component 3) over density."""
  _, rho = _get_density(grid, values)
  return list(grid), values[..., 3, np.newaxis] / rho
# end


def _get_vi(grid: list[np.ndarray],
    values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Extract the velocity vector ``(vx, vy, vz)``: momentum (1:4) over density."""
  _, rho = _get_density(grid, values)
  return list(grid), values[..., 1:4] / rho
# end


def _infer_num_moms(values: np.ndarray, num_moms: int | None) -> int:
  """Resolve the moment count, inferring it from the component count."""
  if num_moms is not None:
    return num_moms
  # end
  num_comps = values.shape[-1]
  if num_comps == 5:
    return 5
  # end
  if num_comps == 10:
    return 10
  # end
  raise ValueError(
      f"Number of components appears to be {num_comps:d}; it needs to be "
      "specified using 'num_moms' (5 or 10)")
# end


def _get_p(grid: list[np.ndarray], values: np.ndarray, *,
    gas_gamma: float = 5.0 / 3, num_moms: int | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the scalar pressure from fluid moment data.

  For 5-moment data the pressure is the total energy minus the bulk kinetic
  energy, scaled by ``gas_gamma - 1``. For 10-moment data it is the trace of
  the pressure tensor over three: ``(P_xx + P_yy + P_zz) / 3``.

  Args:
    grid: Nodal coordinate arrays, one per spatial dimension.
    values: Moment array (5- or 10-moment).
    gas_gamma: Adiabatic index, used only for 5-moment data.
    num_moms: Number of moments (5 or 10); inferred from the component count
      when ``None``.

  Returns:
    ``(grid, values)`` holding the scalar pressure field.

  Raises:
    ValueError: If ``num_moms`` is ``None`` and cannot be inferred.
  """
  num_moms = _infer_num_moms(values, num_moms)

  if num_moms == 5:
    _, rho = _get_density(grid, values)
    _, vx = _get_vx(grid, values)
    _, vy = _get_vy(grid, values)
    _, vz = _get_vz(grid, values)
    out_values = (gas_gamma - 1) * (
        values[..., 4, np.newaxis] - 0.5 * rho * (vx**2 + vy**2 + vz**2))
  # end
  else:  # num_moms == 10
    # Trace of the pressure tensor, computed inline (rather than calling
    # ten_moment._get_pxx/_get_pyy/_get_pzz) to keep five_moment ->
    # ten_moment a one-way edge; ten_moment._get_pxx/pyy/pzz apply this same
    # M_ii - rho*v_i*v_i formula component-wise.
    _, rho = _get_density(grid, values)
    _, vx = _get_vx(grid, values)
    _, vy = _get_vy(grid, values)
    _, vz = _get_vz(grid, values)
    pxx = values[..., 4, np.newaxis] - rho * vx * vx
    pyy = values[..., 7, np.newaxis] - rho * vy * vy
    pzz = values[..., 9, np.newaxis] - rho * vz * vz
    out_values = (pxx + pyy + pzz) / 3.0
  # end

  return list(grid), out_values
# end


def _get_ke(grid: list[np.ndarray], values: np.ndarray, *,
    gas_gamma: float = 5.0 / 3, num_moms: int | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the kinetic (bulk-flow) energy density from fluid moment data.

  For 5-moment data it is the total energy minus the thermal energy
  ``p / (gas_gamma - 1)``. For 10-moment data it is
  ``0.5 * rho * (vx**2 + vy**2 + vz**2)`` directly.

  Args:
    grid: Nodal coordinate arrays, one per spatial dimension.
    values: Moment array (5- or 10-moment).
    gas_gamma: Adiabatic index, used only for 5-moment data.
    num_moms: Number of moments (5 or 10); inferred from the component count
      when ``None``.

  Returns:
    ``(grid, values)`` holding the kinetic energy density field.
  """
  num_moms = _infer_num_moms(values, num_moms)

  if num_moms == 5:
    _, pr = _get_p(grid, values, gas_gamma=gas_gamma, num_moms=num_moms)
    out_values = values[..., 4, np.newaxis] - pr / (gas_gamma - 1)
  # end
  else:  # num_moms == 10
    _, rho = _get_density(grid, values)
    _, vx = _get_vx(grid, values)
    _, vy = _get_vy(grid, values)
    _, vz = _get_vz(grid, values)
    out_values = 0.5 * rho * (vx**2 + vy**2 + vz**2)
  # end

  return list(grid), out_values
# end


def _get_temp(grid: list[np.ndarray], values: np.ndarray, *,
    gas_gamma: float = 5.0 / 3, num_moms: int | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the temperature ``T = p / rho`` from fluid moment data."""
  _, rho = _get_density(grid, values)
  _, pr = _get_p(grid, values, gas_gamma=gas_gamma, num_moms=num_moms)
  return list(grid), pr / rho
# end


def _get_sound(grid: list[np.ndarray], values: np.ndarray, *,
    gas_gamma: float = 5.0 / 3, num_moms: int | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the sound speed ``c_s = sqrt(gas_gamma * p / rho)``."""
  _, rho = _get_density(grid, values)
  _, pr = _get_p(grid, values, gas_gamma=gas_gamma, num_moms=num_moms)
  return list(grid), np.sqrt(gas_gamma * pr / rho)
# end


def _get_mach(grid: list[np.ndarray], values: np.ndarray, *,
    gas_gamma: float = 5.0 / 3, num_moms: int | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the sonic Mach number ``M = |v| / c_s``."""
  _, vx = _get_vx(grid, values)
  _, vy = _get_vy(grid, values)
  _, vz = _get_vz(grid, values)
  _, cs = _get_sound(grid, values, gas_gamma=gas_gamma, num_moms=num_moms)
  return list(grid), np.sqrt(vx**2 + vy**2 + vz**2) / cs
# end


# ---------------------------------------------------------------- GData verbs
def density(data: "GDataState", *, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """Mass density (component 0 of fluid moment data).

  Args:
    data: Fluid moment data; must be NumPy-backed.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A single-component dataset of the density.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "density", _REASON)
  grid, values = _get_density(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def xvel(data: "GDataState", *, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """x velocity: x momentum (component 1) over density.

  Args:
    data: Fluid moment data; must be NumPy-backed.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A single-component dataset of the x velocity.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "xvel", _REASON)
  grid, values = _get_vx(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def yvel(data: "GDataState", *, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """y velocity: y momentum (component 2) over density.

  Args:
    data: Fluid moment data; must be NumPy-backed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the y velocity.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "yvel", _REASON)
  grid, values = _get_vy(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def zvel(data: "GDataState", *, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """z velocity: z momentum (component 3) over density.

  Args:
    data: Fluid moment data; must be NumPy-backed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the z velocity.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "zvel", _REASON)
  grid, values = _get_vz(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def vel(data: "GDataState", *, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """Velocity vector ``(vx, vy, vz)``: momentum (1:4) over density.

  Args:
    data: Fluid moment data; must be NumPy-backed.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A three-component dataset of the fluid velocity.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed).
  """
  _require_field_domain(data, "vel", _REASON)
  grid, values = _get_vi(data.grid, data.values)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def pressure(data: "GDataState", *, gas_gamma: float = 5.0 / 3,
    num_moms: int | None = None, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """Scalar pressure from fluid moment data (5- or 10-moment).

  Args:
    data: Fluid moment data (5- or 10-moment); must be NumPy-backed.
    gas_gamma: Adiabatic index, used only for 5-moment data.
    num_moms: Number of moments (5 or 10); inferred from the component count
      when ``None``.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A single-component dataset of the scalar pressure.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed), or ``num_moms`` is
      ``None`` and cannot be inferred.
  """
  _require_field_domain(data, "pressure", _REASON)
  grid, values = _get_p(data.grid, data.values, gas_gamma=gas_gamma,
      num_moms=num_moms)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def ke(data: "GDataState", *, gas_gamma: float = 5.0 / 3,
    num_moms: int | None = None, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """Kinetic (bulk-flow) energy density from fluid moment data.

  Args:
    data: Fluid moment data; must be NumPy-backed.
    gas_gamma: Adiabatic index, used only for 5-moment data.
    num_moms: Number of moments (5 or 10); inferred from component count
      when ``None``.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the bulk-flow energy density.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed), or ``num_moms`` is
      ``None`` and cannot be inferred.
  """
  _require_field_domain(data, "ke", _REASON)
  grid, values = _get_ke(data.grid, data.values, gas_gamma=gas_gamma,
      num_moms=num_moms)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def temp(data: "GDataState", *, gas_gamma: float = 5.0 / 3,
    num_moms: int | None = None, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """Temperature ``T = p / rho`` from fluid moment data.

  Args:
    data: Fluid moment data; must be NumPy-backed.
    gas_gamma: Adiabatic index, used only for 5-moment data.
    num_moms: Number of moments (5 or 10); inferred from component count
      when ``None``.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the temperature.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed), or ``num_moms`` is
      ``None`` and cannot be inferred.
  """
  _require_field_domain(data, "temp", _REASON)
  grid, values = _get_temp(data.grid, data.values, gas_gamma=gas_gamma,
      num_moms=num_moms)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def sound(data: "GDataState", *, gas_gamma: float = 5.0 / 3,
    num_moms: int | None = None, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """Sound speed ``c_s = sqrt(gas_gamma * p / rho)``.

  Args:
    data: Fluid moment data; must be NumPy-backed.
    gas_gamma: Adiabatic index.
    num_moms: Number of moments (5 or 10); inferred from component count
      when ``None``.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the sound speed.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed), or ``num_moms`` is
      ``None`` and cannot be inferred.
  """
  _require_field_domain(data, "sound", _REASON)
  grid, values = _get_sound(data.grid, data.values, gas_gamma=gas_gamma,
      num_moms=num_moms)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def mach(data: "GDataState", *, gas_gamma: float = 5.0 / 3,
    num_moms: int | None = None, inplace: bool = False,
    tag: str | None = None, label: str | None = None) -> "GDataState":
  """Sonic Mach number ``M = |v| / c_s``.

  Args:
    data: Fluid moment data; must be NumPy-backed.
    gas_gamma: Adiabatic index used to compute the sound speed.
    num_moms: Number of moments (5 or 10); inferred from component count
      when ``None``.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the Mach number.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed), or ``num_moms`` is
      ``None`` and cannot be inferred.
  """
  _require_field_domain(data, "mach", _REASON)
  grid, values = _get_mach(data.grid, data.values, gas_gamma=gas_gamma,
      num_moms=num_moms)
  return data._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def velocity(density: "GDataState", momentum: "GDataState", *,
    inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Velocity from separate density and momentum moments.

  Computes the flow velocity by dividing the ``momentum`` moments by the
  ``density`` moment, component-wise. The two inputs are assumed to share
  the same grid; the result carries the ``density`` dataset's grid.

  Args:
    density: Number/mass density moment (single component); the divisor.
      Must be NumPy-backed.
    momentum: Momentum moment(s) to divide by the density. Must be
      NumPy-backed.
    inplace: mutate and return ``density`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A dataset of the velocity.

  Raises:
    ValueError: if either input is native modal (gkyl-backed).
  """
  _require_field_domain(density, "velocity", _REASON)
  _require_field_domain(momentum, "velocity", _REASON)
  values = momentum.values / density.values
  return density._result(density.grid, values, inplace=inplace, tag=tag,
      label=label)
# end


VARIABLES = {
    "density": density, "xvel": xvel, "yvel": yvel, "zvel": zvel, "vel": vel,
    "pressure": pressure, "ke": ke, "temp": temp, "sound": sound,
    "mach": mach,
}
