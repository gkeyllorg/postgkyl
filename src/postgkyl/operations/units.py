"""Convert a dataset's values between supported physical units."""

from __future__ import annotations

import operator

from scipy.constants import elementary_charge

from postgkyl.gdatastate.gdatastate import GDataState
from . import arithmetic

# Unit -> (dimension, size of one unit in SI units).
_UNITS = {
    "m^-3": ("number density", 1.0),
    "cm^-3": ("number density", 1e6),
    "J": ("energy", 1.0),
    "eV": ("energy", elementary_charge),
    "keV": ("energy", 1e3 * elementary_charge),
    "m/s": ("speed", 1.0),
    "km/s": ("speed", 1e3),
    "V": ("potential", 1.0),
    "kV": ("potential", 1e3),
    "V/m": ("electric field", 1.0),
    "kV/m": ("electric field", 1e3),
    "T": ("magnetic field", 1.0),
    "G": ("magnetic field", 1e-4),
}


def available_units() -> list[tuple[str, str]]:
  """List compatible value-unit families for CLI discovery."""
  families: dict[str, list[str]] = {}
  for unit, (dimension, _) in _UNITS.items():
    families.setdefault(dimension, []).append(unit)
  return [(" / ".join(members), dimension)
          for dimension, members in families.items()]


def _converted_label(current: str, source: str, dest: str) -> str | None:
  """Change only a complete, recognizable unit suffix in a display label."""
  if source == dest:
    return None
  if source == "m^-3" and "(m$^{-3}$)" in current:
    return current.replace("(m$^{-3}$)", "(cm$^{-3}$)")
  if source == "cm^-3" and "(cm$^{-3}$)" in current:
    return current.replace("(cm$^{-3}$)", "(m$^{-3}$)")
  old = f"({source})"
  return current.replace(old, f"({dest})") if old in current else None


def units(data: GDataState,
          *,
          source: str,
          dest: str,
          inplace: bool = False,
          tag: str | None = None,
          label: str | None = None) -> GDataState:
  """Scale dataset values from ``source`` units to ``dest`` units.

  Source and destination must describe the same dimension. Use ``units
  --avail`` to list supported translations. The grid is not converted. Native
  modal coefficients are scaled without interpolation.

  Args:
    data: Dataset to convert.
    source: Unit currently used by the values.
    dest: Unit desired for the values.
    inplace: Mutate and return ``data`` instead of a new dataset.
    tag: Optional tag for the result.
    label: Optional display label, overriding an automatic unit replacement.
  """
  if source not in _UNITS:
    raise ValueError(f"unsupported source unit {source!r}")
  if dest not in _UNITS:
    raise ValueError(f"unsupported destination unit {dest!r}")
  source_dimension, source_scale = _UNITS[source]
  dest_dimension, dest_scale = _UNITS[dest]
  if source_dimension != dest_dimension:
    raise ValueError(f"incompatible units {source!r} and {dest!r}")

  factor = source_scale / dest_scale
  scaled = arithmetic.binary(operator.mul, data, factor)
  result_label = label if label is not None else _converted_label(
      data.get_label(), source, dest)
  values = scaled.native if scaled.backend == "gkyl" else scaled.values
  return data._result(data.grid,
                      values,
                      inplace=inplace,
                      tag=tag,
                      label=result_label,
                      value_units=dest)
