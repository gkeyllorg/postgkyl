"""
Python bindings for Gkeyll DG binary operations via ctypes.

Usage:
  ops = GkeyllDGops("/path/to/gkylsoft/gkeyll")
  ops.invert(0, out_gdata, 0, inp_gdata)
  ops.multiply(0, out_gdata, 0, lop_gdata, 0, rop_gdata)
"""

import ctypes
import os

import numpy as np

from postgkyl._gkylsoft_path import resolve_gkylsoft_path
from postgkyl.data import GData
import postgkyl.utils.gkeyll_enums as gke
from postgkyl.data.dg import _getnum_nodes
from postgkyl.modalDG.kernels import expand_1d

# gkyl_elem_type enum ordinal for double (INT=0, FLOAT=1, DOUBLE=2)
_GKYL_DOUBLE = ctypes.c_int(2)

class GkeyllDGops:
  """
  Operations on DG data, returning DG data.
  Some of these are implemented in Gkeyll, and we get them from libg0core.so.

  Inputs:
    gkylsoft_path: Path to the gkylsoft directory (the one containing gkeyll/lib/libg0core.so).
                   If None, falls back to the GKYLSOFT env var, ~/.postgkyl/gkylsoft_path,
                   and the build-time default in postgkyl._gkylsoft_path.
  """

  def __init__(self, gkylsoft_path: str | None = None):
    path = resolve_gkylsoft_path(gkylsoft_path)
    if path is None:
      raise RuntimeError("gkylsoft path not configured. Set the GKYLSOFT environment variable, "
                         "write the path to ~/.postgkyl/gkylsoft_path, or pass gkylsoft_path= "
                         "to GkeyllDGops().")
    # end
    lib_file = os.path.join(path, "gkeyll/lib", "libg0core.so")
    if not os.path.isfile(lib_file):
      raise FileNotFoundError(f"libg0core.so not found at {lib_file}. "
                               "Check that the gkylsoft path is correct.")
    # end
    self._lib = ctypes.CDLL(lib_file)
    self._setup_signatures()

  def _setup_signatures(self) -> None:
    lib = self._lib
    c_vp = ctypes.c_void_p
    c_i  = ctypes.c_int
    c_sz = ctypes.c_size_t
    c_d  = ctypes.c_double

    # gkyl_array_new_from_buff(type, ncomp, size, buff) -> gkyl_array*
    lib.gkyl_array_new_from_buff.argtypes = [c_i, c_sz, c_sz, c_vp]
    lib.gkyl_array_new_from_buff.restype  = c_vp

    # gkyl_array_release(arr)
    lib.gkyl_array_release.argtypes = [c_vp]
    lib.gkyl_array_release.restype  = None

    # gkyl_cart_modal_serendip_new(ndim, poly_order) -> gkyl_basis*
    lib.gkyl_cart_modal_serendip_new.argtypes = [c_i, c_i]
    lib.gkyl_cart_modal_serendip_new.restype  = c_vp

    # gkyl_cart_modal_gkhybrid_new(cdim, vdim) -> gkyl_basis*
    lib.gkyl_cart_modal_gkhybrid_new.argtypes = [c_i, c_i]
    lib.gkyl_cart_modal_gkhybrid_new.restype  = c_vp

    # gkyl_cart_modal_basis_get_num_basis(*basis) -> int
    lib.gkyl_cart_modal_basis_get_num_basis.argtypes = [c_vp]
    lib.gkyl_cart_modal_basis_get_num_basis.restype  = c_i

    # gkyl_cart_modal_basis_release(basis)
    lib.gkyl_cart_modal_basis_release.argtypes = [c_vp]
    lib.gkyl_cart_modal_basis_release.restype  = None

    # gkyl_rect_grid_new(ndim, *lower, *upper, *cells) -> gkyl_rect_grid*
    lib.gkyl_rect_grid_new.argtypes = [c_i, ctypes.POINTER(c_d),
                                       ctypes.POINTER(c_d), ctypes.POINTER(c_i)]
    lib.gkyl_rect_grid_new.restype  = c_vp

    # gkyl_rect_grid_release(grid)
    lib.gkyl_rect_grid_release.argtypes = [c_vp]
    lib.gkyl_rect_grid_release.restype  = None

    # gkyl_range_new(ndim, *lower, *upper) -> gkyl_range*
    lib.gkyl_range_new.argtypes = [c_i, ctypes.POINTER(c_i), ctypes.POINTER(c_i)]
    lib.gkyl_range_new.restype  = c_vp

    # gkyl_range_release(rng)
    lib.gkyl_range_release.argtypes = [c_vp]
    lib.gkyl_range_release.restype  = None

    # gkyl_dg_mul_op(*basis, c_oop, *out, c_lop, *lop, c_rop, rop*)
    lib.gkyl_dg_mul_op.argtypes = [c_vp, c_i, c_vp, c_i, c_vp, c_i, c_vp]
    lib.gkyl_dg_mul_op.restype  = None

    # gkyl_dg_mul_conf_phase_op_range(*cbasis, *pbasis, *pout, *cop, *pop, *crange, *prange)
    lib.gkyl_dg_mul_conf_phase_op_range.argtypes = [c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp]
    lib.gkyl_dg_mul_conf_phase_op_range.restype  = None

    # gkyl_dg_inv_op(*basis, c_oop, *out, c_iop, *iop)
    lib.gkyl_dg_inv_op.argtypes = [c_vp, c_i, c_vp, c_i, c_vp]
    lib.gkyl_dg_inv_op.restype  = None

    # gkyl_dg_differentiate_op_local(*basis, dir, diff_order, dx, c_oop, *out, c_iop, inp*)
    lib.gkyl_dg_differentiate_op_local.argtypes = [c_vp, c_i, c_i, c_d, c_i, c_vp, c_i, c_vp]
    lib.gkyl_dg_differentiate_op_local.restype  = None

    # gkyl_dg_eval_at_coord_proj_new(cdim_do, *basis_do, num_eval_dirs, *eval_dirs, use_gpu)
    lib.gkyl_dg_eval_at_coord_proj_new.argtypes = [c_i, c_vp, c_i, ctypes.POINTER(c_i), ctypes.c_bool]
    lib.gkyl_dg_eval_at_coord_proj_new.restype  = c_vp

    # gkyl_dg_eval_at_coord_proj_target_basis(up*, cdim*, ndim*, btype*, poly_order*, num_basis*)
    lib.gkyl_dg_eval_at_coord_proj_target_basis.argtypes = [
      c_vp, ctypes.POINTER(c_i), ctypes.POINTER(c_i), ctypes.POINTER(c_i),
      ctypes.POINTER(c_i), ctypes.POINTER(c_i),
    ]
    lib.gkyl_dg_eval_at_coord_proj_target_basis.restype  = None

    # gkyl_dg_eval_at_coord_proj_advance(up*, eval_coords*, grid*, pick_lower*,
    #                                     known_index*, rng_do*, rng_tar*, fdo*, ftar*)
    lib.gkyl_dg_eval_at_coord_proj_advance.argtypes = [
      c_vp, ctypes.POINTER(c_d), c_vp, ctypes.POINTER(ctypes.c_bool),
      ctypes.POINTER(c_i), c_vp, c_vp, c_vp, c_vp,
    ]
    lib.gkyl_dg_eval_at_coord_proj_advance.restype  = None

    # gkyl_dg_eval_at_coord_proj_release(up*)
    lib.gkyl_dg_eval_at_coord_proj_release.argtypes = [c_vp]
    lib.gkyl_dg_eval_at_coord_proj_release.restype  = None

    # gkyl_proj_powsqrt_on_basis_new(*basis, num_quad, use_gpu) -> gkyl_proj_powsqrt_on_basis*
    lib.gkyl_proj_powsqrt_on_basis_new.argtypes = [c_vp, c_i, ctypes.c_bool]
    lib.gkyl_proj_powsqrt_on_basis_new.restype  = c_vp

    # gkyl_proj_powsqrt_on_basis_advance(up*, *range, expIn, *fIn, *fOut)
    lib.gkyl_proj_powsqrt_on_basis_advance.argtypes = [c_vp, c_vp, c_d, c_vp, c_vp]
    lib.gkyl_proj_powsqrt_on_basis_advance.restype  = None

    # gkyl_proj_powsqrt_on_basis_release(up*)
    lib.gkyl_proj_powsqrt_on_basis_release.argtypes = [c_vp]
    lib.gkyl_proj_powsqrt_on_basis_release.restype  = None

    # gkyl_array_average_new(*grid, *basis, *basis_avg, *local, *local_avg,
    #                         *local_avg_ext, *weight, *avg_dim, use_gpu) -> gkyl_array_average*
    lib.gkyl_array_average_new.argtypes = [c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                                           ctypes.POINTER(c_i), ctypes.c_bool]
    lib.gkyl_array_average_new.restype  = c_vp

    # gkyl_array_average_advance(up*, fin*, avgout*)
    lib.gkyl_array_average_advance.argtypes = [c_vp, c_vp, c_vp]
    lib.gkyl_array_average_advance.restype  = None

    # gkyl_array_average_release(up*)
    lib.gkyl_array_average_release.argtypes = [c_vp]
    lib.gkyl_array_average_release.restype  = None

  def _gkyl_array_new_from_gdata(self, gdata):
    """
    Wrap a GData's value buffer in a gkyl_array without copying.

    Returns (arr_ptr, values) where values is the numpy array kept alive
    to prevent GC while arr_ptr is in use.
    """
    values = np.squeeze(gdata.get_values())
    # Ensure C-contiguous float64 layout expected by gkyl kernels
    values = np.ascontiguousarray(values, dtype=np.float64)
    size  = ctypes.c_size_t(int(np.prod(values.shape[:-1])))
    ncomp = ctypes.c_size_t(int(values.shape[-1]))
    data_ptr = values.ctypes.data_as(ctypes.c_void_p)
    arr_ptr = self._lib.gkyl_array_new_from_buff(_GKYL_DOUBLE, ncomp, size, data_ptr)
    return arr_ptr, values

  def _gkyl_basis_new_from_gdata(self, gdata):
    """Create a basis from a GData's metadata. Caller must release."""
    ndim       = gdata.get_num_dims()
    poly_order = int(gdata.ctx["poly_order"])
    basis_type = gdata.ctx["basis_type"]
    if basis_type == "gkhybrid":
      vdim = 1 if ndim == 2 else 2
      cdim = ndim - vdim
      return self._lib.gkyl_cart_modal_gkhybrid_new(ctypes.c_int(cdim), ctypes.c_int(vdim))
    else:
      return self._lib.gkyl_cart_modal_serendip_new(ctypes.c_int(ndim), ctypes.c_int(poly_order))

  def _gkyl_range_new_from_gdata(self, gdata):
    """Create a 1-indexed gkyl_range covering all cells of gdata. Caller must release."""
    values = gdata.get_values()
    cells  = list(values.shape[:-1])
    ndim   = len(cells)
    c_lo   = (ctypes.c_int * ndim)(*([1] * ndim))
    c_up   = (ctypes.c_int * ndim)(*cells)
    return self._lib.gkyl_range_new(ctypes.c_int(ndim), c_lo, c_up)

  def multiply(self, c_oop: int, oop, c_lop: int, lop, c_rop: int, rop) -> None:
    """
    Weak DG multiply: oop[c_oop] = lop[c_lop] * rop[c_rop].

    Inputs:
      c_oop, c_lop, c_rop: Physical component indices (0-based) within each multi-component field.
                           Use 0 for single-component (scalar) fields.
      oop, lop, rop: Output and input operand datasets. Must be pre-allocated.
    """
    basis = self._gkyl_basis_new_from_gdata(lop)
    arr_oop, _ = self._gkyl_array_new_from_gdata(oop)
    arr_lop, _ = self._gkyl_array_new_from_gdata(lop)
    arr_rop, _ = self._gkyl_array_new_from_gdata(rop)
    try:
      self._lib.gkyl_dg_mul_op(basis,
        ctypes.c_int(c_oop), arr_oop,
        ctypes.c_int(c_lop), arr_lop,
        ctypes.c_int(c_rop), arr_rop,)
    finally:
      self._lib.gkyl_cart_modal_basis_release(basis)
      self._lib.gkyl_array_release(arr_oop)
      self._lib.gkyl_array_release(arr_lop)
      self._lib.gkyl_array_release(arr_rop)

  def multiply_conf_phase(self, pout, cop, pop) -> None:
    """
    Weak DG conf-phase multiply: pout = cop * pop on all cells.

    cop is a conf-space field and pop/pout are phase-space fields.
    Ranges are constructed automatically from the shape of each dataset.

    Inputs:
      pout: Output phase-space dataset. Must be pre-allocated.
      cop:  Conf-space operand dataset.
      pop:  Phase-space operand dataset.
    """
    cbasis = self._gkyl_basis_new_from_gdata(cop)
    pbasis = self._gkyl_basis_new_from_gdata(pop)
    arr_pout, _ = self._gkyl_array_new_from_gdata(pout)
    arr_cop,  _ = self._gkyl_array_new_from_gdata(cop)
    arr_pop,  _ = self._gkyl_array_new_from_gdata(pop)
    crange = self._gkyl_range_new_from_gdata(cop)
    prange = self._gkyl_range_new_from_gdata(pop)
    try:
      self._lib.gkyl_dg_mul_conf_phase_op_range(
        cbasis, pbasis, arr_pout, arr_cop, arr_pop, crange, prange)
    finally:
      self._lib.gkyl_cart_modal_basis_release(cbasis)
      self._lib.gkyl_cart_modal_basis_release(pbasis)
      self._lib.gkyl_array_release(arr_pout)
      self._lib.gkyl_array_release(arr_cop)
      self._lib.gkyl_array_release(arr_pop)
      self._lib.gkyl_range_release(crange)
      self._lib.gkyl_range_release(prange)

  def differentiate(self, dir: int, diff_order: int, dx: float, c_oop: int, oop, c_iop: int, iop) -> None:
    """
    Local DG differentiation: oop[c_oop] = d^diff_order/dx_dir^diff_order iop[c_iop].

    Differentiates the DG expansion in each cell independently (no inter-cell stencil).

    Inputs:
      dir:        Direction of differentiation (0-based).
      diff_order: Order of the derivative (1 or 2).
      dx:         Cell length in the direction of differentiation.
      c_oop, c_iop: Physical component indices (0-based).
      oop, iop:   Output and input datasets. oop must be allocated.
    """
    basis = self._gkyl_basis_new_from_gdata(iop)
    arr_oop, _ = self._gkyl_array_new_from_gdata(oop)
    arr_iop, _ = self._gkyl_array_new_from_gdata(iop)
    try:
      self._lib.gkyl_dg_differentiate_op_local(basis,
        ctypes.c_int(dir), ctypes.c_int(diff_order), ctypes.c_double(dx),
        ctypes.c_int(c_oop), arr_oop,
        ctypes.c_int(c_iop), arr_iop,)
    finally:
      self._lib.gkyl_cart_modal_basis_release(basis)
      self._lib.gkyl_array_release(arr_oop)
      self._lib.gkyl_array_release(arr_iop)

  def eval_at_coord_proj(self, eval_dirs: list, eval_coords: list, gdata,
                         comp_grid: bool = False) -> GData:
    """
    Evaluate a DG field at physical coordinates in eval_dirs and project onto
    the lower-dimensional target basis.

    Inputs:
      eval_dirs:   Sorted list of 0-based direction indices to eliminate.
      eval_coords: Physical coordinates, one per entry in eval_dirs.
      gdata:       Donor DG dataset (must have poly_order in ctx).
      comp_grid:   Passed to the output GData constructor.

    Returns:
      GData with the projected field. The surviving grid dimensions, cells,
      lower, upper, and num_comps in ctx are set correctly for the target.
    """
    ndim       = gdata.get_num_dims()
    vals       = gdata.get_values()
    poly_order = int(gdata.ctx["poly_order"])

    basis_type = gdata.ctx["basis_type"]
    grid_type = gdata.ctx["grid_type"]

    ggrid = gdata.get_grid()
    grid_edges = [np.copy(ggrid[d]) for d in range(ndim)]
    if basis_type == "gkhybrid" and grid_type == "c2p_vel":
      # Grid has DG coefficients of v-space mapping along v-dims. Evaluate at cell boundaries.
      # MF 2026/06/28: I think this should happen outside of this function,
      # but we do it here for now to avoid modifying other code.
      poly_order_vmap = 1
      num_cdim = gdata.ctx["num_cdim"]
      num_vdim = gdata.ctx["num_vdim"]
      num_basis_1v = int(_getnum_nodes(1, 1, "serendipity")) # 1D p1 basis for single v dimension.
      nodes = [-1.0, 1.0]
      for d in range(num_vdim):
        q = grid_edges[num_cdim+d]
        grid_edges_1v = np.zeros(np.size(q,0)+1)
        for i, vmap_c in enumerate(q):
          grid_edges_1v[i] = expand_1d[int(poly_order_vmap - 1)](vmap_c, nodes[0])
        # end
        # Append upper boundary surface.
        grid_edges_1v[-1] = expand_1d[int(poly_order_vmap - 1)](q[-1], nodes[1])

        grid_edges[num_cdim+d] = grid_edges_1v
      # end
    # end

    cells = [len(grid_edges[d]) - 1 for d in range(ndim)]
    lower = [float(grid_edges[d][0])   for d in range(ndim)]
    upper = [float(grid_edges[d][-1])  for d in range(ndim)]

    num_eval  = len(eval_dirs)
    keep_dirs = [d for d in range(ndim) if d not in eval_dirs]
    ndim_tar  = len(keep_dirs)
    cells_tar = [cells[d] for d in keep_dirs] if num_eval<ndim else [1 for d in range(ndim)]

    # Donor grid.
    c_lower  = (ctypes.c_double * ndim)(*lower)
    c_upper  = (ctypes.c_double * ndim)(*upper)
    c_cells  = (ctypes.c_int   * ndim)(*cells)
    grid_ptr = self._lib.gkyl_rect_grid_new(ctypes.c_int(ndim), c_lower, c_upper, c_cells)

    # Donor range (1-indexed).
    c_rng_lo_do = (ctypes.c_int * ndim)(*([1] * ndim))
    c_rng_up_do = (ctypes.c_int * ndim)(*cells)
    rng_do_ptr  = self._lib.gkyl_range_new(ctypes.c_int(ndim), c_rng_lo_do, c_rng_up_do)

    # Donor basis.
    basis_do_ptr = self._gkyl_basis_new_from_gdata(gdata)
    num_basis_do = int(self._lib.gkyl_cart_modal_basis_get_num_basis(basis_do_ptr))

    # Updater.
    c_eval_dirs = (ctypes.c_int * num_eval)(*eval_dirs)
    updater     = self._lib.gkyl_dg_eval_at_coord_proj_new(ctypes.c_int(ndim), basis_do_ptr, 
      ctypes.c_int(num_eval), c_eval_dirs, ctypes.c_bool(False), )

    # Get number of basis in target field.
    _cdim_tar = ctypes.c_int(); _ndim_tar = ctypes.c_int(); _btype_tar = ctypes.c_int()
    _poly_order_tar = ctypes.c_int(); _num_basis_tar = ctypes.c_int()
    self._lib.gkyl_dg_eval_at_coord_proj_target_basis(updater, ctypes.byref(_cdim_tar), ctypes.byref(_ndim_tar),
      ctypes.byref(_btype_tar), ctypes.byref(_poly_order_tar), ctypes.byref(_num_basis_tar), )
    num_basis_tar = int(_num_basis_tar.value)

    # Target range and grid.
    if ndim_tar > 0:
      c_rng_lo_tar = (ctypes.c_int * ndim_tar)(*([1] * ndim_tar))
      c_rng_up_tar = (ctypes.c_int * ndim_tar)(*cells_tar)
      rng_tar_ptr  = self._lib.gkyl_range_new(ctypes.c_int(ndim_tar), c_rng_lo_tar, c_rng_up_tar)
      tar_grid     = [ggrid[d] for d in keep_dirs] # Use original grid to keep mapping if c2p_vel.
    else:
      c_one       = (ctypes.c_int * 1)(1)
      rng_tar_ptr = self._lib.gkyl_range_new(ctypes.c_int(1), c_one, c_one)
      tar_grid    = [np.array([eval_coords[d]]) for d in range(num_eval)]

    # Donor array.
    arr_do, values = self._gkyl_array_new_from_gdata(gdata)
    ncomp_raw = int(values.shape[-1])

    # Target buffer.
    num_phys_comps = ncomp_raw // num_basis_do
    ncomp_tar      = num_phys_comps * num_basis_tar
    size_tar       = int(np.prod(cells_tar))
    tar_shape      = (*cells_tar, ncomp_tar)
    tar_buf        = np.zeros(tar_shape, dtype=np.float64)
    arr_tar        = self._lib.gkyl_array_new_from_buff(_GKYL_DOUBLE, ctypes.c_size_t(ncomp_tar),
      ctypes.c_size_t(size_tar), tar_buf.ctypes.data_as(ctypes.c_void_p), )

    c_eval_coords = (ctypes.c_double * num_eval)(*eval_coords)
    c_pick_lower  = (ctypes.c_bool   * num_eval)(*([False] * num_eval))
    c_known_idx   = (ctypes.c_int    * ndim)(*([-1] * ndim))
    try:
      self._lib.gkyl_dg_eval_at_coord_proj_advance(updater, c_eval_coords, grid_ptr,
        c_pick_lower, c_known_idx, rng_do_ptr, rng_tar_ptr, arr_do, arr_tar,)
    finally:
      self._lib.gkyl_dg_eval_at_coord_proj_release(updater)
      self._lib.gkyl_cart_modal_basis_release(basis_do_ptr)
      self._lib.gkyl_array_release(arr_do)
      self._lib.gkyl_array_release(arr_tar)
      self._lib.gkyl_rect_grid_release(grid_ptr)
      self._lib.gkyl_range_release(rng_do_ptr)
      self._lib.gkyl_range_release(rng_tar_ptr)

    out = GData(ctx=gdata.ctx, comp_grid=comp_grid)
    out.push(tar_grid, tar_buf)

    # Re-set the basis in the context in case it changed.
    out.ctx["basis_type"] = gke.basis_type_gkyl_to_pgkyl(int(_btype_tar.value))
    out.ctx["poly_order"] = int(_poly_order_tar.value)
    out.ctx["num_cdim"] = int(_cdim_tar.value)
    out.ctx["num_vdim"] = int(_ndim_tar.value - _cdim_tar.value)

    return out

  def average(self, avg_dirs: list, gdata, weight=None, comp_grid: bool = False) -> GData:
    """
    Average a DG field over the directions in avg_dirs (gkyl_array_average).

    Returns a GData over the surviving dimensions. With a weight GData (same
    dims/basis as gdata) the weighted average is computed instead. Serendipity
    basis, poly_order <= 2 only.
    """
    basis_type = gdata.ctx["basis_type"]
    if basis_type.lower() != "serendipity":
      raise ValueError(f"average only supports the serendipity basis, got '{basis_type}'. "
                       "gkyl_array_average provides serendipity kernels only.")

    ndim       = gdata.get_num_dims()
    poly_order = int(gdata.ctx["poly_order"])
    if poly_order > 2:
      raise ValueError(f"average only supports poly_order <= 2, got {poly_order}.")

    if weight is not None:
      w_basis_type = weight.ctx["basis_type"]
      if w_basis_type.lower() != "serendipity":
        raise ValueError(f"weight must use the serendipity basis, got '{w_basis_type}'.")
      if weight.get_num_dims() != ndim:
        raise ValueError(f"weight has {weight.get_num_dims()} dims but the field has {ndim}; "
                         "they must match.")
      if int(weight.ctx["poly_order"]) != poly_order:
        raise ValueError(f"weight poly_order {int(weight.ctx['poly_order'])} != field "
                         f"poly_order {poly_order}.")

    avg_dirs  = sorted(set(avg_dirs))
    if not avg_dirs or avg_dirs[0] < 0 or avg_dirs[-1] >= ndim:
      raise ValueError(f"average dirs {avg_dirs} out of range for a {ndim}D field.")
    keep_dirs = [d for d in range(ndim) if d not in avg_dirs]
    ndim_tar  = len(keep_dirs)

    ggrid      = gdata.get_grid()
    grid_edges = [np.copy(ggrid[d]) for d in range(ndim)]
    cells = [len(grid_edges[d]) - 1 for d in range(ndim)]
    lower = [float(grid_edges[d][0])  for d in range(ndim)]
    upper = [float(grid_edges[d][-1]) for d in range(ndim)]

    # For a full average (no surviving dims), Gkeyll keeps a 1D, single-cell
    # target following the same convention as eval_at_coord_proj.
    ndim_red  = ndim_tar if ndim_tar > 0 else 1
    cells_tar = [cells[d] for d in keep_dirs] if ndim_tar > 0 else [1]

    # Donor grid.
    c_lower  = (ctypes.c_double * ndim)(*lower)
    c_upper  = (ctypes.c_double * ndim)(*upper)
    c_cells  = (ctypes.c_int    * ndim)(*cells)
    grid_ptr = self._lib.gkyl_rect_grid_new(ctypes.c_int(ndim), c_lower, c_upper, c_cells)

    # Donor (full) range, 1-indexed.
    c_rng_lo = (ctypes.c_int * ndim)(*([1] * ndim))
    c_rng_up = (ctypes.c_int * ndim)(*cells)
    rng_ptr  = self._lib.gkyl_range_new(ctypes.c_int(ndim), c_rng_lo, c_rng_up)

    # Target (reduced) range, 1-indexed.
    c_rng_lo_tar = (ctypes.c_int * ndim_red)(*([1] * ndim_red))
    c_rng_up_tar = (ctypes.c_int * ndim_red)(*cells_tar)
    rng_tar_ptr  = self._lib.gkyl_range_new(ctypes.c_int(ndim_red), c_rng_lo_tar, c_rng_up_tar)

    # Full (donor) and reduced (target) serendipity bases.
    basis_do  = self._gkyl_basis_new_from_gdata(gdata)
    basis_avg = self._lib.gkyl_cart_modal_serendip_new(ctypes.c_int(ndim_red),
      ctypes.c_int(poly_order))
    num_basis_do  = int(self._lib.gkyl_cart_modal_basis_get_num_basis(basis_do))
    num_basis_tar = int(self._lib.gkyl_cart_modal_basis_get_num_basis(basis_avg))

    # Donor array.
    arr_do, values = self._gkyl_array_new_from_gdata(gdata)
    ncomp_raw = int(values.shape[-1])

    # Optional weight array (spans the full donor range/basis). Keep _w_values
    # alive so its numpy buffer is not collected while the kernel runs.
    arr_w, _w_values = (None, None)
    if weight is not None:
      arr_w, _w_values = self._gkyl_array_new_from_gdata(weight)

    # Target buffer.
    num_phys_comps = ncomp_raw // num_basis_do
    ncomp_tar      = num_phys_comps * num_basis_tar
    size_tar       = int(np.prod(cells_tar))
    tar_shape      = (*cells_tar, ncomp_tar)
    tar_buf        = np.zeros(tar_shape, dtype=np.float64)
    arr_tar        = self._lib.gkyl_array_new_from_buff(_GKYL_DOUBLE, ctypes.c_size_t(ncomp_tar),
      ctypes.c_size_t(size_tar), tar_buf.ctypes.data_as(ctypes.c_void_p), )

    # avg_dim flags (1 = averaged) over the full dimensionality.
    avg_flags = [1 if d in avg_dirs else 0 for d in range(ndim)]
    c_avg_dim = (ctypes.c_int * ndim)(*avg_flags)

    # rng_tar_ptr doubles as local_avg_ext (only read to size the integrated weight).
    updater = self._lib.gkyl_array_average_new(grid_ptr, basis_do, basis_avg,
      rng_ptr, rng_tar_ptr, rng_tar_ptr, arr_w, c_avg_dim, ctypes.c_bool(False))
    try:
      self._lib.gkyl_array_average_advance(updater, arr_do, arr_tar)
    finally:
      self._lib.gkyl_array_average_release(updater)
      self._lib.gkyl_cart_modal_basis_release(basis_do)
      self._lib.gkyl_cart_modal_basis_release(basis_avg)
      self._lib.gkyl_array_release(arr_do)
      self._lib.gkyl_array_release(arr_tar)
      if arr_w is not None:
        self._lib.gkyl_array_release(arr_w)
      self._lib.gkyl_rect_grid_release(grid_ptr)
      self._lib.gkyl_range_release(rng_ptr)
      self._lib.gkyl_range_release(rng_tar_ptr)

    if ndim_tar == 0 and weight is None:
      tar_buf *= np.sqrt(2.0)

    tar_grid = [ggrid[d] for d in keep_dirs] if ndim_tar > 0 else [np.array([0.0, 1.0])]

    out = GData(ctx=gdata.ctx, comp_grid=comp_grid)
    out.push(tar_grid, tar_buf)

    out.ctx["basis_type"] = "serendipity"
    out.ctx["poly_order"] = poly_order
    out.ctx["num_cdim"]   = ndim_tar
    out.ctx["num_vdim"]   = 0

    return out

  @staticmethod
  def _lift_matrix(ndim: int, avg_dirs: list, poly_order: int, basis_type: str) -> np.ndarray:
    """
    Matrix T mapping the modal coefficients of a field over the kept
    directions (ndim - len(avg_dirs) dims) onto the modal coefficients of the
    same field in the full ndim basis, constant along avg_dirs.
    """
    from postgkyl.data.computeInterpolationMatrices import createInterpMatrix

    keep_dirs = [d for d in range(ndim) if d not in avg_dirs]
    nnodes = poly_order + 1
    mat_full = createInterpMatrix(ndim, poly_order, basis_type, nnodes, True)
    if keep_dirs:
      mat_red = createInterpMatrix(len(keep_dirs), poly_order, basis_type, nnodes, True)
    else:
      mat_red = np.ones((1, 1))  # A full average is a single constant.

    # Node n of the full grid (dim 0 fastest) sits on reduced node lift_idx[n].
    lift_idx = np.zeros(nnodes**ndim, dtype=int)
    for n in range(nnodes**ndim):
      multi = np.unravel_index(n, [nnodes]*ndim, order="F")
      if keep_dirs:
        lift_idx[n] = np.ravel_multi_index([multi[d] for d in keep_dirs],
                                           [nnodes]*len(keep_dirs), order="F")
    # end

    return np.linalg.pinv(mat_full) @ mat_red[lift_idx, :]

  def expand(self, reduced, like, avg_dirs: list) -> GData:
    """
    Lift a reduced DG field (e.g. the output of average(avg_dirs, like)) back
    onto the grid and basis of 'like', constant along the directions avg_dirs.
    """
    ndim       = like.get_num_dims()
    poly_order = int(like.ctx["poly_order"])
    basis_type = like.ctx["basis_type"]
    avg_dirs   = sorted(set(avg_dirs))
    keep_dirs  = [d for d in range(ndim) if d not in avg_dirs]

    lift = self._lift_matrix(ndim, avg_dirs, poly_order, basis_type)
    nb_full, nb_red = lift.shape

    full_vals = like.get_values()
    cells = full_vals.shape[:-1]
    num_comps = full_vals.shape[-1] // nb_full

    red_vals = reduced.get_values()
    if not keep_dirs:
      # A full average is stored in a single 1D cell: keep the value of the
      # constant mode (its basis function is 1/sqrt(2)) of each component.
      nb_1d = red_vals.shape[-1] // num_comps
      red_vals = red_vals.reshape(num_comps, nb_1d)[:, 0] / np.sqrt(2.0)

    # Lift every component's coefficients, then broadcast along avg_dirs.
    red_vals = red_vals.reshape(*red_vals.shape[:-1], num_comps, nb_red)
    lifted = np.einsum("fr,...cr->...cf", lift, red_vals)
    lifted = lifted.reshape(*lifted.shape[:-2], num_comps*nb_full)
    for d in avg_dirs:
      lifted = np.expand_dims(lifted, axis=d)
    lifted = np.broadcast_to(lifted, full_vals.shape).copy()

    out = GData(ctx=like.ctx)
    out.push(like.get_grid(), lifted)
    return out

  def fluctuation(self, avg_dirs: list, gdata, weight=None) -> GData:
    """
    Fluctuation of a DG field about its (optionally weighted) average over
    avg_dirs: gdata - <gdata>_{avg_dirs}. The output keeps the full grid.
    """
    mean = self.average(avg_dirs, gdata, weight=weight)
    mean_full = self.expand(mean, gdata, avg_dirs)
    out = GData(ctx=gdata.ctx)
    out.push(gdata.get_grid(), gdata.get_values() - mean_full.get_values())
    return out

  def invert(self, c_oop: int, oop, c_iop: int, iop) -> None:
    """
    Weak DG invert: oop[c_oop] = 1 / iop[c_iop].

    Only supported for serendipity basis at p=1 (gkeyll limitation).

    Inputs:
      c_oop, c_iop: Physical component indices (0-based).
      oop, iop: Output and input datasets. oop be allocated.
    """
    basis  = self._gkyl_basis_new_from_gdata(iop)
    arr_oop, _ = self._gkyl_array_new_from_gdata(oop)
    arr_iop, _ = self._gkyl_array_new_from_gdata(iop)
    try:
      self._lib.gkyl_dg_inv_op(basis,
        ctypes.c_int(c_oop), arr_oop,
        ctypes.c_int(c_iop), arr_iop,)
    finally:
      self._lib.gkyl_cart_modal_basis_release(basis)
      self._lib.gkyl_array_release(arr_oop)
      self._lib.gkyl_array_release(arr_iop)

  def powsqrt(self, oop, iop, exponent: float, num_quad: int | None = None) -> None:
    """
    Weak DG power of a square root: oop = pow(sqrt(iop), exponent), projected
    onto the basis by Gauss-Legendre quadrature (gkyl_proj_powsqrt_on_basis).

    Inputs:
      oop, iop:  Output and input datasets. oop must be pre-allocated.
      exponent:  Exponent applied to sqrt(iop).
      num_quad:  Quadrature nodes per direction. Defaults to poly_order+1,
                 matching the gyrokinetic app's own use of this updater.
    """
    basis = self._gkyl_basis_new_from_gdata(iop)
    try:
      num_basis = int(self._lib.gkyl_cart_modal_basis_get_num_basis(basis))
      for name, gdata in (("iop", iop), ("oop", oop)):
        num_comps = int(np.squeeze(gdata.get_values()).shape[-1])
        if num_comps != num_basis:
          raise ValueError(
            f"powsqrt: '{name}' has {num_comps} coefficients per cell but the basis has "
            f"{num_basis}; this operation only takes single-component (scalar) fields.")
        # end
      # end

      if num_quad is None:
        num_quad = int(iop.ctx["poly_order"]) + 1
      # end

      up = self._lib.gkyl_proj_powsqrt_on_basis_new(
        basis, ctypes.c_int(num_quad), ctypes.c_bool(False))
      arr_oop, _ = self._gkyl_array_new_from_gdata(oop)
      arr_iop, _ = self._gkyl_array_new_from_gdata(iop)
      rng = self._gkyl_range_new_from_gdata(iop)
      try:
        self._lib.gkyl_proj_powsqrt_on_basis_advance(
          up, rng, ctypes.c_double(exponent), arr_iop, arr_oop)
      finally:
        self._lib.gkyl_proj_powsqrt_on_basis_release(up)
        self._lib.gkyl_array_release(arr_oop)
        self._lib.gkyl_array_release(arr_iop)
        self._lib.gkyl_range_release(rng)
    finally:
      self._lib.gkyl_cart_modal_basis_release(basis)

