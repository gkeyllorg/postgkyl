"""Vector rotation parallel/perpendicular to a reference (e.g. the magnetic
field).

For a field ``u`` and a rotator ``v`` (assumed three-component, last axis),
``parrotate`` computes the projection of ``u`` onto ``v``'s direction,
``(u . v_hat) v_hat``; ``perprotate`` is the remainder, ``u - (u . v_hat)
v_hat``.

Note: :mod:`postgkyl.numerics.rotation_matrix` builds a matrix whose first
row is the *elementwise sign* of its input, not a true unit vector (see its
own tests) -- using it here would change the projection's numerical result,
so this module keeps the original dot-product formula instead (Doctrine:
copy numerics verbatim).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...gdatastate.guards import require_field_domain as _require_field_domain

if TYPE_CHECKING:
  from ...gdatastate.gdatastate import GDataState
# end

_REASON = "rotating raw DG coefficients would mix basis functions"


# --------------------------------------------------------- array-level math
def _parrotate(grid: list[np.ndarray], values: np.ndarray,
    rotator_values: np.ndarray, *, rotate_coords: str = "0:3",
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Rotate a three-component field into the direction of a rotator field.

  Args:
    grid: Nodal coordinate arrays, one per spatial dimension.
    values: Three-component field to rotate (last axis is components).
    rotator_values: Field providing the rotation direction, on the same
      grid as ``values``.
    rotate_coords: ``"start:end"`` slice of ``rotator_values``'s component
      axis to use as the rotation direction (e.g. ``"3:6"`` to rotate into
      a magnetic field stored after three electric-field components).

  Returns:
    ``(grid, values)`` holding the parallel component
    ``(u . v_hat) v_hat``.

  Raises:
    ValueError: If ``values`` or the sliced ``rotator_values`` do not have
      exactly three components.
  """
  lo, hi = rotate_coords.split(":")
  valuesrot = rotator_values[..., slice(int(lo), int(hi))]

  if values.shape[-1] != 3 or valuesrot.shape[-1] != 3:
    raise ValueError(
        "parrotate requires three-component vector fields; data has "
        f"{values.shape[-1]:d} components, rotator (after 'rotate_coords' "
        f"slicing) has {valuesrot.shape[-1]:d}")
  # end

  scale = np.sum(values * valuesrot, axis=-1) / np.sum(
      valuesrot * valuesrot, axis=-1)
  outrot = scale[..., np.newaxis] * valuesrot

  return list(grid), outrot
# end


def _perprotate(grid: list[np.ndarray], values: np.ndarray,
    rotator_values: np.ndarray, *, rotate_coords: str = "0:3",
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Rotate a three-component field perpendicular to a rotator field.

  Computed as the remainder after :func:`_parrotate`:
  ``u - (u . v_hat) v_hat``.
  """
  grid, par = _parrotate(grid, values, rotator_values,
      rotate_coords=rotate_coords)
  return grid, values - par
# end


# ---------------------------------------------------------------- GData verbs
def parrotate(array: "GDataState", rotator: "GDataState", *,
    coords: str = "0:3", inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Component of ``array`` parallel to ``rotator``: ``(u . v_hat) v_hat``.

  Projects the three-component vector field ``array`` (u) onto the unit
  vector of the ``rotator`` field (v), returning the parallel vector
  ``(u . v_hat) v_hat`` with its x, y, z components. Both fields are
  assumed to be three-component with components on the last axis.

  Args:
    array: The three-component vector field to be rotated/projected; must
      be NumPy-backed.
    rotator: The field defining the rotation direction; must be
      NumPy-backed.
    coords: Half-open 'lo:hi' slice string selecting which ``rotator``
      components form the direction vector. Defaults to '0:3'; use '3:6'
      to rotate along the magnetic field of a six-component EM field.
    inplace: mutate and return ``array`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A three-component dataset of the parallel projection.

  Raises:
    ValueError: if either input is native modal (gkyl-backed), or the
      component counts do not match a three-component field.
  """
  _require_field_domain(array, "parrotate", _REASON)
  _require_field_domain(rotator, "parrotate", _REASON)
  grid, values = _parrotate(array.grid, array.values, rotator.values,
      rotate_coords=coords)
  return array._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def perprotate(array: "GDataState", rotator: "GDataState", *,
    coords: str = "0:3", inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Component of ``array`` perpendicular to ``rotator``:
  ``u - (u . v_hat) v_hat``.

  Both fields are assumed to be three-component with components on the
  last axis.

  Args:
    array: The three-component vector field to be rotated/projected; must
      be NumPy-backed.
    rotator: The field defining the rotation direction; must be
      NumPy-backed.
    coords: Half-open 'lo:hi' slice string selecting which ``rotator``
      components form the direction vector. Defaults to '0:3'; use '3:6'
      to rotate along the magnetic field of a six-component EM field.
    inplace: mutate and return ``array`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A three-component dataset of the perpendicular component.

  Raises:
    ValueError: if either input is native modal (gkyl-backed), or the
      component counts do not match a three-component field.
  """
  _require_field_domain(array, "perprotate", _REASON)
  _require_field_domain(rotator, "perprotate", _REASON)
  grid, values = _perprotate(array.grid, array.values, rotator.values,
      rotate_coords=coords)
  return array._result(grid, values, inplace=inplace, tag=tag, label=label)
# end


def bparrotate(array: "GDataState", field: "GDataState", *,
    inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Project an array parallel to the magnetic field.

  Args:
    array: Vector or tensor dataset to project.
    field: Electromagnetic field whose components 3 through 5 are magnetic.
    inplace: Mutate and return ``array`` instead of creating a dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.
  """
  return parrotate(array, field, coords="3:6", inplace=inplace, tag=tag,
      label=label)
# end


def bperprotate(array: "GDataState", field: "GDataState", *,
    inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Project an array perpendicular to the magnetic field.

  Args:
    array: Vector or tensor dataset to project.
    field: Electromagnetic field whose components 3 through 5 are magnetic.
    inplace: Mutate and return ``array`` instead of creating a dataset.
    tag: Optional tag for the returned dataset.
    label: Optional label for the returned dataset.
  """
  return perprotate(array, field, coords="3:6", inplace=inplace, tag=tag,
      label=label)
# end


__all__ = ["parrotate", "perprotate", "bparrotate", "bperprotate"]
