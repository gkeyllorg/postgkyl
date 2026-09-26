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

import operator
from typing import TYPE_CHECKING

import numpy as np

from ...gdatastate import materialize_point_values
from ...operations.arithmetic import binary
from ...operations.select import select
from ...gdatastate.guards import require_field_domain as _require_field_domain

if TYPE_CHECKING:
  from ...gdatastate.gdatastate import GDataState

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


def _get_vx(grid: list[np.ndarray],
            values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Extract the x velocity: x momentum (component 1) over density."""
  _, rho = _get_density(grid, values)
  return list(grid), values[..., 1, np.newaxis] / rho


def _get_vy(grid: list[np.ndarray],
            values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Extract the y velocity: y momentum (component 2) over density."""
  _, rho = _get_density(grid, values)
  return list(grid), values[..., 2, np.newaxis] / rho


def _get_vz(grid: list[np.ndarray],
            values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Extract the z velocity: z momentum (component 3) over density."""
  _, rho = _get_density(grid, values)
  return list(grid), values[..., 3, np.newaxis] / rho


def _get_vi(grid: list[np.ndarray],
            values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  """Extract the velocity vector ``(vx, vy, vz)``: momentum (1:4) over density."""
  _, rho = _get_density(grid, values)
  return list(grid), values[..., 1:4] / rho


def _infer_num_moms(values: np.ndarray, num_moms: int | None) -> int:
  """Resolve the moment count, inferring it from the component count."""
  if num_moms is not None:
    return num_moms
  num_comps = values.shape[-1]
  if num_comps == 5:
    return 5
  if num_comps == 10:
    return 10
  raise ValueError(
      f"Number of components appears to be {num_comps:d}; it needs to be "
      "specified using 'num_moms' (5 or 10)")


def _get_p(
    grid: list[np.ndarray],
    values: np.ndarray,
    *,
    gas_gamma: float = 5.0 / 3,
    num_moms: int | None = None,
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
    out_values = (gas_gamma - 1) * (values[..., 4, np.newaxis] - 0.5 * rho *
                                    (vx**2 + vy**2 + vz**2))
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

  return list(grid), out_values


def _get_ke(
    grid: list[np.ndarray],
    values: np.ndarray,
    *,
    gas_gamma: float = 5.0 / 3,
    num_moms: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the kinetic (bulk-flow) energy density from fluid moment data.

  For either moment layout this is ``0.5 * |momentum|**2 / rho``.
  Computing it directly avoids cancellation against the internal energy.

  Args:
    grid: Nodal coordinate arrays, one per spatial dimension.
    values: Moment array (5- or 10-moment).
    gas_gamma: Adiabatic index, used only for 5-moment data.
    num_moms: Number of moments (5 or 10); inferred from the component count
      when ``None``.

  Returns:
    ``(grid, values)`` holding the kinetic energy density field.
  """
  _infer_num_moms(values, num_moms)
  if values.shape[-1] < 4:
    raise ValueError("ke requires density and all three momentum components")
  _, rho = _get_density(grid, values)
  out_values = 0.5 * np.sum(values[..., 1:4]**2, axis=-1, keepdims=True) / rho

  return list(grid), out_values


def _get_temp(
    grid: list[np.ndarray],
    values: np.ndarray,
    *,
    gas_gamma: float = 5.0 / 3,
    num_moms: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the temperature ``T = p / rho`` from fluid moment data."""
  _, rho = _get_density(grid, values)
  _, pr = _get_p(grid, values, gas_gamma=gas_gamma, num_moms=num_moms)
  return list(grid), pr / rho


def _get_sound(
    grid: list[np.ndarray],
    values: np.ndarray,
    *,
    gas_gamma: float = 5.0 / 3,
    num_moms: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the sound speed ``c_s = sqrt(gas_gamma * p / rho)``."""
  _, rho = _get_density(grid, values)
  _, pr = _get_p(grid, values, gas_gamma=gas_gamma, num_moms=num_moms)
  return list(grid), np.sqrt(gas_gamma * pr / rho)


def _get_mach(
    grid: list[np.ndarray],
    values: np.ndarray,
    *,
    gas_gamma: float = 5.0 / 3,
    num_moms: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
  """Compute the sonic Mach number ``M = |v| / c_s``."""
  _, vx = _get_vx(grid, values)
  _, vy = _get_vy(grid, values)
  _, vz = _get_vz(grid, values)
  _, cs = _get_sound(grid, values, gas_gamma=gas_gamma, num_moms=num_moms)
  return list(grid), np.sqrt(vx**2 + vy**2 + vz**2) / cs


# ---------------------------------------------------------------- GData verbs
def density(data: "GDataState",
            *,
            inplace: bool = False,
            tag: str | None = None,
            label: str | None = None) -> "GDataState":
  """Mass density (component 0 of fluid moment data).

  Args:
    data: Fluid moment data at physical points (interpolated, nodal, or quadrature).
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A single-component dataset of the density.

  Raises:
    ValueError: if ``data`` contains unevaluated modal coefficients.
  """
  _require_field_domain(data, "density", _REASON)
  point_data = materialize_point_values(data)
  grid, values = _get_density(point_data.grid, point_data.values)
  return data._result(grid,
                      values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True,
                      value_form=None)


def xvel(data: "GDataState",
         *,
         inplace: bool = False,
         tag: str | None = None,
         label: str | None = None) -> "GDataState":
  """x velocity: x momentum (component 1) over density.

  Args:
    data: Fluid moment data at physical points (interpolated, nodal, or quadrature).
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A single-component dataset of the x velocity.

  Raises:
    ValueError: if ``data`` contains unevaluated modal coefficients.
  """
  _require_field_domain(data, "xvel", _REASON)
  point_data = materialize_point_values(data)
  grid, values = _get_vx(point_data.grid, point_data.values)
  return data._result(grid,
                      values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True,
                      value_form=None)


def yvel(data: "GDataState",
         *,
         inplace: bool = False,
         tag: str | None = None,
         label: str | None = None) -> "GDataState":
  """y velocity: y momentum (component 2) over density.

  Args:
    data: Fluid moment data at physical points (interpolated, nodal, or quadrature).
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the y velocity.

  Raises:
    ValueError: if ``data`` contains unevaluated modal coefficients.
  """
  _require_field_domain(data, "yvel", _REASON)
  point_data = materialize_point_values(data)
  grid, values = _get_vy(point_data.grid, point_data.values)
  return data._result(grid,
                      values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True,
                      value_form=None)


def zvel(data: "GDataState",
         *,
         inplace: bool = False,
         tag: str | None = None,
         label: str | None = None) -> "GDataState":
  """z velocity: z momentum (component 3) over density.

  Args:
    data: Fluid moment data at physical points (interpolated, nodal, or quadrature).
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the z velocity.

  Raises:
    ValueError: if ``data`` contains unevaluated modal coefficients.
  """
  _require_field_domain(data, "zvel", _REASON)
  point_data = materialize_point_values(data)
  grid, values = _get_vz(point_data.grid, point_data.values)
  return data._result(grid,
                      values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True,
                      value_form=None)


def vel(data: "GDataState",
        *,
        inplace: bool = False,
        tag: str | None = None,
        label: str | None = None) -> "GDataState":
  """Velocity vector ``(vx, vy, vz)``: momentum (1:4) over density.

  Args:
    data: Fluid moment data at physical points (interpolated, nodal, or quadrature).
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A three-component dataset of the fluid velocity.

  Raises:
    ValueError: if ``data`` contains unevaluated modal coefficients.
  """
  _require_field_domain(data, "vel", _REASON)
  point_data = materialize_point_values(data)
  grid, values = _get_vi(point_data.grid, point_data.values)
  return data._result(grid,
                      values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True,
                      value_form=None)


def pressure(data: "GDataState",
             *,
             gas_gamma: float = 5.0 / 3,
             num_moms: int | None = None,
             inplace: bool = False,
             tag: str | None = None,
             label: str | None = None) -> "GDataState":
  """Scalar pressure from fluid moment data (5- or 10-moment).

  Args:
    data: Fluid moment data (5- or 10-moment) at physical points.
    gas_gamma: Adiabatic index, used only for 5-moment data.
    num_moms: Number of moments (5 or 10); inferred from the component count
      when ``None``.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A single-component dataset of the scalar pressure.

  Raises:
    ValueError: if ``data`` contains unevaluated modal coefficients, or ``num_moms`` is
      ``None`` and cannot be inferred.
  """
  _require_field_domain(data, "pressure", _REASON)
  point_data = materialize_point_values(data)
  grid, values = _get_p(point_data.grid,
                        point_data.values,
                        gas_gamma=gas_gamma,
                        num_moms=num_moms)
  return data._result(grid,
                      values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True,
                      value_form=None)


def ke(data: "GDataState",
       *,
       gas_gamma: float = 5.0 / 3,
       num_moms: int | None = None,
       inplace: bool = False,
       tag: str | None = None,
       label: str | None = None) -> "GDataState":
  """Kinetic (bulk-flow) energy density from fluid moment data.

  Native data retains its representation and uses semantic component
  selection and arithmetic. On modal inputs, this computes the weak
  products of each momentum component, then weakly divides their sum by
  density. It never indexes modal coefficients as physical components.

  Args:
    data: Native DG or physical point-value fluid moments.
    gas_gamma: Retained for API compatibility; bulk energy is independent of it.
    num_moms: Number of moments (5 or 10); inferred from component count
      when ``None``.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the bulk-flow energy density.

  Raises:
    ValueError: if the moment layout cannot be inferred, or modal data
      lacks native arithmetic support.
  """
  if data.backend == "gkyl":
    rho = select(data, comp=0)
    moments = data.num_comps // rho.num_comps
    if (num_moms is None and moments not in (5, 10)) or moments < 4:
      raise ValueError("ke requires five- or ten-moment fluid data")
    momentum = [select(data, comp=i) for i in (1, 2, 3)]
    squares = [binary(operator.mul, p, p) for p in momentum]
    squared = binary(operator.add, binary(operator.add, squares[0], squares[1]),
                     squares[2])
    energy = binary(operator.mul, 0.5, binary(operator.truediv, squared, rho))
    return data._result(data.grid,
                        energy.native,
                        inplace=inplace,
                        tag=tag,
                        label=label)
  _require_field_domain(data, "ke", _REASON)
  point_data = materialize_point_values(data)
  grid, values = _get_ke(point_data.grid,
                         point_data.values,
                         gas_gamma=gas_gamma,
                         num_moms=num_moms)
  return data._result(grid,
                      values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True,
                      value_form=None)


def temp(data: "GDataState",
         *,
         gas_gamma: float = 5.0 / 3,
         num_moms: int | None = None,
         inplace: bool = False,
         tag: str | None = None,
         label: str | None = None) -> "GDataState":
  """Temperature ``T = p / rho`` from fluid moment data.

  Args:
    data: Fluid moment data at physical points (interpolated, nodal, or quadrature).
    gas_gamma: Adiabatic index, used only for 5-moment data.
    num_moms: Number of moments (5 or 10); inferred from component count
      when ``None``.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the temperature.

  Raises:
    ValueError: if ``data`` contains unevaluated modal coefficients, or ``num_moms`` is
      ``None`` and cannot be inferred.
  """
  _require_field_domain(data, "temp", _REASON)
  point_data = materialize_point_values(data)
  grid, values = _get_temp(point_data.grid,
                           point_data.values,
                           gas_gamma=gas_gamma,
                           num_moms=num_moms)
  return data._result(grid,
                      values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True,
                      value_form=None)


def sound(data: "GDataState",
          *,
          gas_gamma: float = 5.0 / 3,
          num_moms: int | None = None,
          inplace: bool = False,
          tag: str | None = None,
          label: str | None = None) -> "GDataState":
  """Sound speed ``c_s = sqrt(gas_gamma * p / rho)``.

  Args:
    data: Fluid moment data at physical points (interpolated, nodal, or quadrature).
    gas_gamma: Adiabatic index.
    num_moms: Number of moments (5 or 10); inferred from component count
      when ``None``.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the sound speed.

  Raises:
    ValueError: if ``data`` contains unevaluated modal coefficients, or ``num_moms`` is
      ``None`` and cannot be inferred.
  """
  _require_field_domain(data, "sound", _REASON)
  point_data = materialize_point_values(data)
  grid, values = _get_sound(point_data.grid,
                            point_data.values,
                            gas_gamma=gas_gamma,
                            num_moms=num_moms)
  return data._result(grid,
                      values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True,
                      value_form=None)


def mach(data: "GDataState",
         *,
         gas_gamma: float = 5.0 / 3,
         num_moms: int | None = None,
         inplace: bool = False,
         tag: str | None = None,
         label: str | None = None) -> "GDataState":
  """Sonic Mach number ``M = |v| / c_s``.

  Args:
    data: Fluid moment data at physical points (interpolated, nodal, or quadrature).
    gas_gamma: Adiabatic index used to compute the sound speed.
    num_moms: Number of moments (5 or 10); inferred from component count
      when ``None``.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.

  Returns:
    A single-component dataset of the Mach number.

  Raises:
    ValueError: if ``data`` contains unevaluated modal coefficients, or ``num_moms`` is
      ``None`` and cannot be inferred.
  """
  _require_field_domain(data, "mach", _REASON)
  point_data = materialize_point_values(data)
  grid, values = _get_mach(point_data.grid,
                           point_data.values,
                           gas_gamma=gas_gamma,
                           num_moms=num_moms)
  return data._result(grid,
                      values,
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True,
                      value_form=None)


def velocity(density: "GDataState",
             momentum: "GDataState",
             *,
             inplace: bool = False,
             tag: str | None = None,
             label: str | None = None) -> "GDataState":
  """Velocity from separate density and momentum moments.

  Computes the flow velocity by dividing the ``momentum`` moments by the
  ``density`` moment, component-wise. The two inputs are assumed to share
  the same grid; the result carries the ``density`` dataset's grid.

  Args:
    density: Number/mass density moment (single component); the divisor.
      Must contain physical point values.
    momentum: Momentum moment(s) to divide by the density. Must be
      NumPy-backed.
    inplace: mutate and return ``density`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A dataset of the velocity.

  Raises:
    ValueError: if either input contains unevaluated modal coefficients.
  """
  _require_field_domain(density, "velocity", _REASON)
  _require_field_domain(momentum, "velocity", _REASON)
  point_density = materialize_point_values(density)
  point_momentum = materialize_point_values(momentum)
  values = point_momentum.values / point_density.values
  return density._result(point_density.grid,
                         values,
                         inplace=inplace,
                         tag=tag,
                         label=label,
                         interpolated=True,
                         value_form=None)


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
}
