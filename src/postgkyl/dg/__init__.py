"""Discontinuous-Galerkin layer -- orchestrates Gkeyll's compiled DG engine.

Four modules, one per domain boundary:

- :mod:`.interpolate` -- the one-way modal -> NumPy bridge (matrix from
  Gkeyll's basis functions, applied with NumPy); also ``local_poly``, the
  same bridge evaluated at whole-cell points with NaN-separated interfaces
  (the discontinuity-preserving plotting mesh).
- :mod:`.modal` -- operations that stay in the modal domain (weak algebra,
  coefficient linear combinations, integration), all executed by Gkeyll
  kernels on native arrays.
- :mod:`.rep` -- explicit value_form changes (modal · nodal · quad) and
  pointwise functions via quadrature; the field never leaves the native domain.
- :mod:`.map` -- grid mapping: evaluate a coordinate-map field's coefficients
  at a target's own grid points (see ``MAPPING.md``).
"""

from .interpolate import interpolate, local_poly, num_basis
from .map import eval_at_points, map_grid, map_grid_separable
from . import modal, rep

__all__ = [
    "interpolate", "local_poly", "num_basis", "modal", "rep", "eval_at_points",
    "map_grid", "map_grid_separable"
]
