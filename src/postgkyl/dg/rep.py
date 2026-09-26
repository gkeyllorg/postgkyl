"""Representation changes within the native domain -- modal · nodal · quad.

One DG field, three per-cell representations (REFACTOR_GKEYLL_FFI.md §3b):
modal coefficients, values at the basis nodes, values at Gauss–Legendre
quadrature points. Conversions are per-cell matrix applications built from
Gkeyll's basis function pointers (:mod:`postgkyl.gpython.basis`); data enters and
leaves as a native :class:`~postgkyl.gpython.array.GkylArray`, so the field never
leaves the native domain. **Nothing here converts implicitly** -- these are the
backends of the explicit ``.represent()`` and ``.apply()`` verbs.

Exactness: nodal↔modal and modal→basis quadrature→modal preserve the expansion.
An explicit Gauss rule must resolve the projection integrand in each direction.

Note: this is the *cell-local* nodal value_form (N unshared values per
cell). Grid-level shared-node nodal fields (``gkyl_nodal_ops``, used by the
geometry/mapped-grid workflow) are phase C.
"""

from __future__ import annotations

import numpy as np

from postgkyl.gpython import basis as gpython_basis
from postgkyl.gpython.array import GkylArray


def _apply_per_field(arr: GkylArray, comps_in: int,
                     mat: np.ndarray) -> GkylArray:
  """Apply ``mat`` (comps_out × comps_in) to every field of every cell."""
  if arr.ncomp % comps_in:
    raise ValueError(f"ncomp {arr.ncomp} is not a multiple of {comps_in}")
  nfields = arr.ncomp // comps_in
  v = arr.view().reshape(arr.size, nfields, comps_in)
  out = np.einsum("pk,cfk->cfp", mat, v).reshape(arr.size,
                                                 nfields * mat.shape[0])
  return GkylArray.from_numpy(out)


def modal_to_nodal(basis_type: str,
                   ndim: int,
                   poly_order: int,
                   arr: GkylArray,
                   *,
                   cdim: int | None = None,
                   vdim: int | None = None) -> GkylArray:
  """Coefficients -> values at the basis ``node_list`` points (exact)."""
  nb = gpython_basis.num_basis(basis_type,
                               ndim,
                               poly_order,
                               cdim=cdim,
                               vdim=vdim)
  return _apply_per_field(
      arr, nb,
      gpython_basis.modal_to_nodal_matrix(basis_type,
                                          ndim,
                                          poly_order,
                                          cdim=cdim,
                                          vdim=vdim))


def nodal_to_modal(basis_type: str,
                   ndim: int,
                   poly_order: int,
                   arr: GkylArray,
                   *,
                   cdim: int | None = None,
                   vdim: int | None = None) -> GkylArray:
  """Values at the basis nodes -> coefficients (exact inverse)."""
  nb = gpython_basis.num_basis(basis_type,
                               ndim,
                               poly_order,
                               cdim=cdim,
                               vdim=vdim)
  return _apply_per_field(
      arr, nb,
      gpython_basis.nodal_to_modal_matrix(basis_type,
                                          ndim,
                                          poly_order,
                                          cdim=cdim,
                                          vdim=vdim))


def modal_to_quad(basis_type: str,
                  ndim: int,
                  poly_order: int,
                  arr: GkylArray,
                  num_quad: int | None = None,
                  *,
                  cdim: int | None = None,
                  vdim: int | None = None) -> GkylArray:
  """Coefficients -> Gkeyll quadrature values (or an explicit Gauss rule)."""
  nb = gpython_basis.num_basis(basis_type,
                               ndim,
                               poly_order,
                               cdim=cdim,
                               vdim=vdim)
  return _apply_per_field(
      arr, nb,
      gpython_basis.modal_to_quad_matrix(basis_type,
                                         ndim,
                                         poly_order,
                                         num_quad,
                                         cdim=cdim,
                                         vdim=vdim))


def quad_to_modal(basis_type: str,
                  ndim: int,
                  poly_order: int,
                  arr: GkylArray,
                  num_quad: int | None = None,
                  *,
                  cdim: int | None = None,
                  vdim: int | None = None) -> GkylArray:
  """Quadrature values -> coefficients using the matching transform."""
  mat = gpython_basis.quad_to_modal_matrix(basis_type,
                                           ndim,
                                           poly_order,
                                           num_quad,
                                           cdim=cdim,
                                           vdim=vdim)
  return _apply_per_field(arr, mat.shape[1], mat)


def wrap(values: np.ndarray) -> GkylArray:
  """Wrap ``(cells..., ncomp)`` NumPy values back into a native array.

  The doorway for pointwise NumPy results on nodal/quad data: computed on the
  view, wrapped back, so the dataset stays gkyl-native and in-value_form.
  """
  return GkylArray.from_numpy(values)


def _tensor_point_layout(basis_type: str,
                         ndim: int,
                         poly_order: int,
                         rep: str,
                         num_quad: int | None,
                         *,
                         cdim: int | None = None,
                         vdim: int | None = None):
  """Per-dimension reference points + permutation into Fortran tensor order.

  Returns ``(pts_1d_per_dim, perm)`` where ``values[..., perm]`` reorders a
  cell's point values into F-order tensor indexing (dimension 0 fastest).
  Quadrature points are a tensor product by construction; nodal sets are
  checked -- non-tensor node sets (e.g. serendipity p2 in 2-D+) raise.
  """
  if rep == "quad" and num_quad is not None:
    pts_1d, _ = np.polynomial.legendre.leggauss(num_quad)
    return [pts_1d] * ndim, None
  coords = (gpython_basis.quad_node_coords(
      basis_type, ndim, poly_order, cdim=cdim, vdim=vdim)
            if rep == "quad" else gpython_basis.node_coords(
                basis_type, ndim, poly_order, cdim=cdim, vdim=vdim))
  nb = coords.shape[0]
  # Native coordinate transforms introduce roundoff. Group coordinates by
  # distance, since decimal rounding can split a node across a bin boundary.
  uniq = []
  for d in range(ndim):
    axis = np.sort(coords[:, d])
    uniq.append(axis[np.r_[True, np.diff(axis) > 1e-12]])
  counts = [len(u) for u in uniq]
  if int(np.prod(counts)) != nb:
    raise ValueError(
        f"the {basis_type} p{poly_order} {ndim}D node set is not a tensor "
        "product; use .represent(to='quad') for point-value work in this basis."
    )
  lin = np.zeros(nb, dtype=np.int64)
  stride = 1
  for d in range(ndim):
    k = np.argmin(np.abs(coords[:, d, None] - uniq[d]), axis=1)
    if not np.allclose(uniq[d][k], coords[:, d]):
      raise ValueError("node coordinates do not align on a tensor grid")
    lin += k * stride
    stride *= counts[d]
  if len(np.unique(lin)) != nb:
    raise ValueError(
        f"the {basis_type} p{poly_order} {ndim}D node set is not a tensor "
        "product; use .represent(to='quad') for point-value work in this basis."
    )
  return [uniq[d] for d in range(ndim)], np.argsort(lin)


def materialize(basis_type: str,
                ndim: int,
                poly_order: int,
                arr: GkylArray,
                grid: list,
                rep: str,
                num_quad: int | None = None,
                *,
                cdim: int | None = None,
                vdim: int | None = None):
  """Point-value data -> ``(point coordinate grid, ndarray)`` at the TRUE
  physical point locations -- the render path for nodal/quad datasets.

  Unlike ``interpolate`` (which evaluates modal data on an equispaced mesh),
  this performs no basis math at all: the values *are* the field at their
  points; only coordinates and ordering are computed.
  """
  pts_1d, perm = _tensor_point_layout(basis_type,
                                      ndim,
                                      poly_order,
                                      rep,
                                      num_quad,
                                      cdim=cdim,
                                      vdim=vdim)
  counts = [len(p) for p in pts_1d]
  npc = int(np.prod(counts))
  raw = arr.view() if isinstance(arr, GkylArray) else np.asarray(arr)
  ncomp = raw.shape[-1]
  if ncomp % npc:
    raise ValueError(f"ncomp {ncomp} is not a multiple of {npc} points/cell")
  nfields = ncomp // npc
  cells = [len(g) - 1 for g in grid]
  v = raw.reshape(*cells, nfields, npc)
  if perm is not None:
    v = v[..., perm]

  out = np.zeros([cells[d] * counts[d] for d in range(ndim)] + [nfields])
  for n in range(npc):
    off = np.unravel_index(n, counts, order="F")
    idxs = tuple(
        slice(int(off[d]), cells[d] * counts[d], counts[d])
        for d in range(ndim))
    out[idxs] = v[..., n]

  coordinates = []
  for d in range(ndim):
    g = np.asarray(grid[d], dtype=np.float64)
    centers, dxs = 0.5 * (g[:-1] + g[1:]), np.diff(g)
    pts = (centers[:, None] + 0.5 * dxs[:, None] * pts_1d[d][None, :]).ravel()
    coordinates.append(pts)
  return coordinates, out


def apply_pointwise(basis_type: str,
                    ndim: int,
                    poly_order: int,
                    arr: GkylArray,
                    fn,
                    num_quad: int | None = None,
                    *,
                    cdim: int | None = None,
                    vdim: int | None = None) -> GkylArray:
  """``fn`` applied pointwise via quadrature: modal → quad → fn → modal.

  The standard DG treatment of nonlinear operations. ``fn`` receives the
  ``(cells, nfields*nq)`` array of quadrature values and must return the same
  shape (any NumPy ufunc qualifies). The result is modal again.
  """
  quad = modal_to_quad(basis_type,
                       ndim,
                       poly_order,
                       arr,
                       num_quad,
                       cdim=cdim,
                       vdim=vdim)
  vals = fn(quad.view())
  vals = np.asarray(vals, dtype=np.float64)
  if vals.shape != (quad.size, quad.ncomp):
    raise ValueError(
        f"apply(fn): fn changed the shape {(quad.size, quad.ncomp)} -> "
        f"{vals.shape}; it must act pointwise.")
  return quad_to_modal(basis_type,
                       ndim,
                       poly_order,
                       GkylArray.from_numpy(vals),
                       num_quad,
                       cdim=cdim,
                       vdim=vdim)
