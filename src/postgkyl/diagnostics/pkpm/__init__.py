"""PKPM diagnostics -- distribution-function reconstruction from Laguerre
moments.

Composes the full distribution function ``f(x, v_par, v_perp)`` out of the
Laguerre expansion coefficients ``F0(x, v_par)``, ``F1(x, v_par)`` (hardcoded
for ``l=0``, ``n=0,1``) and the PKPM ``T/m`` moment. See Jimmy Juno's slides:
https://drive.google.com/file/d/1548tLF9o7vyW3bkrsq6FvAMV-8XJvKtY/view

``load_pkpm`` is the equation-internal loader for PKPM output files (ported
from ``src_bak/postgkyl/loaders/pkpm.py``): it loads the distribution and its
companion ``pkpm_vars`` file, interpolates them, and applies
``laguerre_compose`` + ``kinetic.transform_frame``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...gdata.gdata import GData
from ...gdatastate.guards import require_field_domain as _require_field_domain
from ..vm.kinetic import transform_frame

if TYPE_CHECKING:
  from ...gdatastate.gdatastate import GDataState
# end

_REASON = "composing raw DG coefficients would mix basis functions"


# --------------------------------------------------------- array-level math
def _laguerre_compose(f_grid: list[np.ndarray], f_values: np.ndarray,
    t_over_m_values: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
  """Compose PKPM expansion coefficients into a single distribution function.

  Args:
    f_grid: ``[x, v_par]`` nodal coordinate arrays.
    f_values: 2-component Laguerre expansion coefficients ``(F0, G)``.
    t_over_m_values: PKPM ``T / m`` moment, single component.

  Returns:
    ``([x, v_par, v_perp], values)``: the extended grid (``v_perp`` a copy
    of the ``v_par`` axis) and the composed distribution function, with a
    trailing singleton component axis.
  """
  x, vpar = f_grid[0], f_grid[1]
  vperp = np.copy(vpar)

  x_cc = (x[:-1] + x[1:]) / 2
  vpar_cc = (vpar[:-1] + vpar[1:]) / 2
  vperp_cc = (vpar[:-1] + vpar[1:]) / 2

  _, _, vperp_3D = np.meshgrid(x_cc, vpar_cc, vperp_cc, indexing="ij")

  F0 = f_values[..., 0]
  G = f_values[..., 1]
  T_m = t_over_m_values[..., 0]

  F1 = F0 - (G.transpose() / T_m).transpose()

  # Adding the np.newaxis allows the subsequent np.multiply (called when
  # doing * on numpy arrays) to work. The arrays need to have the same
  # number of axes, e.g. one cannot multiply (3, 3) and (3,) arrays but can
  # multiply (3, 3) with (3, 1) or (1, 3).
  F0, F1 = F0[..., np.newaxis], F1[..., np.newaxis]
  # T_m gains two new axes here (F0/F1 gain only one above), one deeper than
  # needed to broadcast against vperp_3D -- an extra, constant-along-itself
  # trailing axis leaks into the returned array's shape. Preserved verbatim
  # from src_bak/postgkyl/tools/laguerre_compose.py; pinned by
  # tests/test_diagnostics_pkpm.py.
  T_m = T_m[..., np.newaxis, np.newaxis]

  # Hardcoded for l=0, n=0,1 in
  # https://drive.google.com/file/d/1548tLF9o7vyW3bkrsq6FvAMV-8XJvKtY/view
  f = (F0 + F1 * (1 - vperp_3D**2 / 2 / T_m)) / (2 * np.pi * T_m) * np.exp(
      -(vperp_3D**2) / 2 / T_m)

  f = f[..., np.newaxis]  # Adding the component index

  return [x, vpar, vperp], f
# end


# ---------------------------------------------------------------- GData verb
def laguerre_compose(distribution: "GDataState", variables: "GDataState", *,
    inplace: bool = False, tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Compose PKPM Laguerre coefficients into a full distribution function.

  Reconstructs the full distribution function ``f(x, v_par, v_perp)`` from
  the PKPM Laguerre expansion coefficients ``F0`` and ``G`` (stored as the
  two components of ``distribution``) together with the PKPM
  temperature-over-mass field carried in ``variables``.

  Args:
    distribution: The two-component PKPM Laguerre expansion coefficients
      ``F0(x, v_par)`` and ``G(x, v_par)``; must be NumPy-backed.
    variables: The PKPM variables dataset providing T/m(x) (used as the
      first component); must be NumPy-backed.
    inplace: mutate and return ``distribution`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A dataset holding the composed ``f(x, v_par, v_perp)``.

  Raises:
    ValueError: if either input is native modal (gkyl-backed).
  """
  _require_field_domain(distribution, "laguerre_compose", _REASON)
  _require_field_domain(variables, "laguerre_compose", _REASON)
  grid, values = _laguerre_compose(distribution.grid, distribution.values,
      variables.values)
  return distribution._result(grid, values, inplace=inplace, tag=tag,
      label=label)
# end


# ------------------------------------------------------------------- loader
def load_pkpm(name: str, species: str, idx: "str | int", poly_order: int, *,
    tag: str | None = None, label: str | None = None) -> "GData":
  """Load, interpolate, and frame-transform Gkeyll PKPM data.

  Loads the PKPM distribution (its two Laguerre coefficients ``F0``/``G``)
  and its companion ``pkpm_vars`` file (whose component 3 is ``T/m`` and
  components 0:3 are the bulk velocity ``(ux, uy, uz)``), interpolates both,
  composes the full distribution function (:func:`laguerre_compose`), and
    shifts it into the bulk-flow frame (:func:`~postgkyl.diagnostics.vm.kinetic.
  transform_frame`).

  Args:
    name: Root name (file prefix) of the simulation.
    species: Species name.
    idx: Frame/file number.
    poly_order: Polynomial order of the DG representation.
    tag: Optional tag for the resulting dataset.
    label: Optional label for the resulting dataset.

  Returns:
    A populated, interpolated, frame-transformed
    :class:`~postgkyl.gdata.gdata.GData`.
  """
  gf = GData(f"{name!s}-{species!s}_{idx!s}.gkyl",
      basis_type="hybrid", poly_order=poly_order)
  gvars = GData(f"{name!s}-{species!s}_pkpm_vars_{idx!s}.gkyl",
      basis_type="serendipity", poly_order=poly_order)

  c_dim = gf.num_dims - 1

  gf_interpolated = gf.interpolate()
  gvars_interpolated = gvars.interpolate()

  t_over_m = gvars_interpolated.select(comp=3)
  bulk_u = gvars_interpolated.select(comp="0:3")

  composed = laguerre_compose(gf_interpolated, t_over_m)
  out = transform_frame(composed, bulk_u, cdim=c_dim)

  if tag is not None:
    out.set_tag(tag)
  # end
  if label is not None:
    out.set_label(label)
  # end
  return out
# end


__all__ = ["laguerre_compose", "load_pkpm"]
