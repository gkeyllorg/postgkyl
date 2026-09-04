"""Module including Gkeyll binary reader class."""

from collections.abc import Iterable
from typing import Tuple
import msgpack as mp
import numpy as np
import os.path

from . import mapping

# Format description for raw Gkeyll output file from
# gkyl_array_rio_format_desc.h

# The format of the gkyl binary output is as follows.

# ----------------------------------------------------------------------
# ## Version 0: Jan 2021. Created by A.H.
#    Note Version 0 has no header information

# Data      Type and meaning
# --------------------------
# ndim      uint64_t Dimension of field
# cells     uint64_t[ndim] number of cells in each direction
# lower     float64[ndim] Lower bounds of grid
# upper     float64[ndim] Upper bounds of grid
# esznc     uint64_t Element-size * number of components in field
# size      uint64_t Total number of cells in field
# DATA      size*esznc bytes of data

# ----------------------------------------------------------------------
# ## Version 1: May 9th 2022. Created by A.H

# Data      Type and meaning
# --------------------------
# gkyl0     5 bytes
# version   uint64_t
# file_type uint64_t (See header gkyl_elem_type.h for file types)
# meta_size uint64_t Number of bytes of meta-data
# DATA      meta_size bytes of data. This is in msgpack format

# * For file_type = 1 (field) the above header is followed by

# real_type uint64_t. Indicates real type of data
# ndim      uint64_t Dimension of field
# cells     uint64_t[ndim] number of cells in each direction
# lower     float64[ndim] Lower bounds of grid
# upper     float64[ndim] Upper bounds of grid
# esznc     uint64_t Element-size * number of components in field
# size      uint64_t Total number of cells in field
# DATA      size*esznc bytes of data

# * For file_type = 2 (dynvec) the above header is followed by

# real_type uint64_t. Indicates real type of data
# esznc     uint64_t Element-size * number of components in field
# size      uint64_t Total number of cells in field
# TIME_DATA float64[size] bytes of data
# DATA      size*esznc bytes of data

# * For file_type = 3 (multi-range field) the above header is followed by

# real_type uint64_t. Indicates real type of data
# ndim      uint64_t Dimension of field
# cells     uint64_t[ndim] number of cells in each direction
# lower     float64[ndim] Lower bounds of grid
# upper     float64[ndim] Upper bounds of grid
# esznc     uint64_t Element-size * number of components in field
# size      uint64_t Total number of cells in field
# nrange    uint64_t Number of ranges stored in this file

# For each of the nrange ranges in the field the following data is
# present

# loidx     uint64_t[ndim] Index of lower-left corner of the range
# upidx     uint64_t[ndim] Index of upper-right corner of the range
# size      uint64_t Total number of cells in range
# DATA      size*esznc bytes of data

# Note: the global range in Gkeyll, of which each range is a part,
# is 1-indexed.


class GkylReader(object):
  """Provides a framework to read Gkeyll binary output."""

  def __init__(self,
               file_name: str,
               ctx: dict | None = None,
               axes: tuple | None = (None, None, None, None, None, None),
               comp: str | int | None = None,
               value_form: str | None = None,
               basis_type: str | None = None,
               poly_order: int | None = None,
               **kwargs):
    """Initialize the instance of Gkeyll reader.

    Args:
      file_name: str
      ctx: dict
        Passes context variable with metadata.
      var_name: str = "CartGridField"
      axes: tuple
        Allows to specify the axes to be loaded.
      comp: int or slice
        Allows to specify the components to be loaded.
      value_form: overrides the ``ctx["value_form"]`` the header
        metadata would otherwise imply (e.g. a file tagged with basis
        metadata whose stored values are already point values, not modal
        coefficients).
      basis_type: overrides the ``ctx["basis_type"]`` the header metadata
        would otherwise imply -- for files with no basis metadata at all,
        or metadata that mislabels the basis actually used.
      poly_order: overrides the ``ctx["poly_order"]`` the header metadata
        would otherwise imply. Independent of ``basis_type``/``value_form`` --
        it corrects only the polynomial order, asserting nothing about
        modality.
      **kwargs
        This is not directly used but allowes for unified interface to all the readers
        we use.
    """
    self.file_name = file_name
    self._value_form_override = value_form
    self._basis_type_override = basis_type
    self._poly_order_override = poly_order

    self.dtf = np.dtype("f8")
    self.dti = np.dtype("i8")

    self.offset = 0
    self.doffset = 8

    self.file_type = 1
    self.version = 0

    self.lower: np.ndarray
    self.upper: np.ndarray
    self.num_comps: int
    self.cells: np.ndarray

    if ctx is not None:
      self.ctx = ctx
    else:
      self.ctx = {}

    if not ("grid_type" in self.ctx.keys()):
      self.ctx["grid_type"] = "uniform"

    # Prepare for partial load
    self.partial_load = False
    self.partial_idxs = [""] * 7
    if axes is not None:
      for i, ax in enumerate(axes):
        if ax is not None:
          self.partial_load = True
          self.partial_idxs[i] = str(ax)
    if comp is not None:
      self.partial_load = True
      self.partial_idxs[6] = str(comp)

  def is_compatible(self) -> bool:
    """Checks if file can be read with Gkeyll reader."""
    try:
      magic = np.fromfile(self.file_name,
                          dtype=np.dtype("b"),
                          count=5,
                          offset=0)
      if np.array_equal(magic, [103, 107, 121, 108, 48]):
        self.version = np.fromfile(self.file_name,
                                   dtype=self.dti,
                                   count=1,
                                   offset=5)[0]
        return True
      else:
        return False
    except:
      return False

  # Starting with version 1, .gkyl files contain a header;
  # Version 0 files only include the real-type info
  def _read_header(self) -> None:
    """Reads header information for version 1 files and above."""
    if self.is_compatible():
      self.offset += 5  # Header contatins the gkyl magic sequence

      self.version = np.fromfile(self.file_name,
                                 dtype=self.dti,
                                 count=1,
                                 offset=self.offset)[0]
      self.offset += 8

      self.file_type = np.fromfile(self.file_name,
                                   dtype=self.dti,
                                   count=1,
                                   offset=self.offset)[0]
      self.offset += 8

      meta_size = np.fromfile(self.file_name,
                              dtype=self.dti,
                              count=1,
                              offset=self.offset)[0]
      self.offset += 8

      # read meta
      has_basis = False
      if meta_size > 0:
        fh = open(self.file_name, "rb")
        fh.seek(self.offset)
        unp = mp.unpackb(fh.read(meta_size))
        if isinstance(unp, dict) and self.ctx is not None:
          for key in unp:
            if key == "polyOrder" or key == "poly_order":
              self.ctx["poly_order"] = unp[key]
            elif key == "basisType" or key == "basis_type":
              self.ctx["basis_type"] = unp[key]
              has_basis = True
            else:
              # Covers "value_form" too, if the writer stamped one
              # directly: a file's own metadata is the next-best source of
              # truth once no explicit override was given (see below).
              self.ctx[key] = unp[key]
        self.offset += meta_size
        fh.close()
      if self._basis_type_override is not None:
        # Wins over the file's own metadata (or its absence): lets a caller
        # correct a missing/mislabeled basis_type so downstream verbs
        # (interpolate, average, integrate, ...) resolve the right basis.
        self.ctx["basis_type"] = self._basis_type_override
        has_basis = True
      if self._poly_order_override is not None:
        # Independent of basis_type/value_form: corrects only the polynomial
        # order, asserting nothing about modality.
        self.ctx["poly_order"] = self._poly_order_override
      if has_basis and "value_form" not in self.ctx:
        self.ctx["value_form"] = "modal"
    if self._value_form_override is not None:
      # Wins over both the file's own metadata and the "modal" default.
      self.ctx["value_form"] = self._value_form_override

    # read real-type
    real_type = np.fromfile(self.file_name,
                            dtype=self.dti,
                            count=1,
                            offset=self.offset)[0]
    if real_type == 1:
      self.dtf = np.dtype("f4")
      self.doffset = 4
    self.offset += 8

  def _read_t1t3_v1_domain(self) -> None:
    """Read domain information for file type 1 and 3."""
    # read grid dimensions
    self.num_dims = np.fromfile(self.file_name,
                                dtype=self.dti,
                                count=1,
                                offset=self.offset)[0]
    self.offset += 8

    # read grid shape
    self.cells = np.fromfile(self.file_name,
                             dtype=self.dti,
                             count=self.num_dims,
                             offset=self.offset)
    self.offset += self.num_dims * 8

    # read lower/upper
    self.lower = np.fromfile(self.file_name,
                             dtype=self.dtf,
                             count=self.num_dims,
                             offset=self.offset)
    self.offset += self.num_dims * self.doffset
    self.upper = np.fromfile(self.file_name,
                             dtype=self.dtf,
                             count=self.num_dims,
                             offset=self.offset)
    self.offset += self.num_dims * self.doffset

    # read array elem_ez (the div by doffset is as elem_sz includes
    # sizeof(real_type) = doffset)
    elem_sz_raw = int(
        np.fromfile(self.file_name, dtype=self.dti, count=1,
                    offset=self.offset)[0])
    elem_sz = elem_sz_raw / self.doffset
    self.num_comps = int(elem_sz)
    self.offset += 8

    # read array size
    self.asize = np.fromfile(self.file_name,
                             dtype=self.dti,
                             count=1,
                             offset=self.offset)[0]
    self.offset += 8

    # prep for partial loading
    self.orig_size_array = np.zeros(self.num_dims + 1, dtype=self.dti)
    self.orig_size_array[:-1] = self.cells.copy()
    self.orig_size_array[-1] = self.num_comps
    if self.partial_load:
      # The offsets are set to zero by default
      self.global_offsets = np.zeros((self.num_dims + 1, 2), dtype=self.dti)

      # The offsets need to be parsed; note that for ":", the Python syntax is used,
      # i.e., the first index is included, the second is excluded. Negative indices are
      # also allowed, e.g., ":-1".
      for i in range(self.num_dims):
        sl = self.partial_idxs[i]
        if sl.isdigit():
          self.global_offsets[i, 0] = int(sl)
          self.global_offsets[i, 1] = self.cells[i] - int(sl) - 1
        elif ":" in sl:
          start, stop = sl.split(":")
          if start:
            self.global_offsets[i, 0] = int(start)
          if stop and int(stop) > 0:
            self.global_offsets[i, 1] = self.cells[i] - int(stop)
          elif stop:
            self.global_offsets[i, 1] = -int(stop)

      sl = self.partial_idxs[6]
      if sl.isdigit():
        self.global_offsets[-1, 0] = int(sl)
        self.global_offsets[-1, 1] = self.num_comps - int(sl) - 1
      elif ":" in sl:
        start, stop = sl.split(":")
        if start:
          self.global_offsets[-1, 0] = int(start)
        if stop and int(stop) > 0:
          self.global_offsets[-1, 1] = self.num_comps - int(stop)
        elif stop:
          self.global_offsets[-1, 1] = -int(stop)

      self.cells -= (self.global_offsets[:-1, 1] + self.global_offsets[:-1, 0])
      cell_size = (self.upper - self.lower) / self.orig_size_array[:-1]
      self.lower += self.global_offsets[:-1, 0] * cell_size
      self.upper -= self.global_offsets[:-1, 1] * cell_size
      self.num_comps -= (self.global_offsets[-1, 1] +
                         self.global_offsets[-1, 0])

  def _get_block(self, dim: int, out: np.ndarray, idx: int,
                 dim_offsets: np.ndarray, num_elems: np.ndarray,
                 cells: np.ndarray) -> int:
    """Reads a block of data.

    A recursion is used to read the data from the fastest going index (the last one;
    i.e., the field components) to the slowest.
    """
    if dim == self.num_dims:
      self.offset += dim_offsets[-1, 0] * self.doffset
      out[idx:idx + self.num_comps] = np.fromfile(file=self.file_name,
                                                  dtype=self.dtf,
                                                  count=self.num_comps,
                                                  offset=self.offset)
      self.offset += (self.num_comps + dim_offsets[-1, 1]) * self.doffset
      idx += self.num_comps
    else:
      self.offset += dim_offsets[dim, 0] * np.prod(
          num_elems[dim + 1:]) * self.doffset
      for _ in range(cells[dim]):
        idx = self._get_block(dim=dim + 1,
                              out=out,
                              idx=idx,
                              dim_offsets=dim_offsets,
                              num_elems=num_elems,
                              cells=cells)
      self.offset += dim_offsets[dim, 1] * np.prod(
          num_elems[dim + 1:]) * self.doffset
    return idx

  def _get_data(self,
                count: int,
                lo_idx: np.ndarray | None = None,
                up_idx: np.ndarray | None = None) -> Tuple[np.ndarray, Tuple]:
    """Read raw data and account for partial load."""
    slices = []
    gshape = np.ones(self.num_dims + 1, dtype=self.dti)
    gshape[-1] = self.num_comps

    if not self.partial_load:
      out = np.fromfile(self.file_name,
                        dtype=self.dtf,
                        count=count,
                        offset=self.offset)
      self.offset += count * self.doffset

      if lo_idx is not None:
        for d in range(self.num_dims):
          gshape[d] = up_idx[d] - lo_idx[d] + 1
        slices = [
            slice(lo_idx[d] - 1, up_idx[d]) for d in range(self.num_dims)
        ]  # Gkeyll is 1-indexed
      else:
        for d in range(self.num_dims):
          gshape[d] = self.cells[d]

    else:
      if lo_idx is None:
        lo_idx = np.ones(self.num_dims,
                         dtype=self.dti)  # Gkeyll index is 1-indexed
      if up_idx is None:
        up_idx = self.orig_size_array[:-1]
      num_elems = self.orig_size_array.copy()

      # Adjust the offsets for the partial load for distributed memory data
      dim_offsets = np.zeros_like(self.global_offsets, dtype=self.dti)
      dim_offsets[:-1, 0] = self.global_offsets[:-1, 0] - (lo_idx - 1)
      dim_offsets[:-1,
                  1] = self.global_offsets[:-1, 1] - (num_elems[:-1] - up_idx)
      dim_offsets[-1, :] = self.global_offsets[-1, :]
      dim_offsets = dim_offsets.clip(min=0)

      # Calculate the size to allocate the memory
      num_elems[:-1] = up_idx - lo_idx + 1  # Gkeyll index is 1-indexed
      cells = num_elems[:-1] - dim_offsets[:-1, 1] - dim_offsets[:-1, 0]
      if np.any(cells < 1):
        self.offset += count * self.doffset
        return np.array([]), tuple(slices)
      size = np.prod(cells) * self.num_comps
      out = np.zeros(size, dtype=self.dtf)  # Allocate space for the data
      self._get_block(dim=0,
                      out=out,
                      idx=0,
                      dim_offsets=dim_offsets,
                      num_elems=num_elems,
                      cells=cells)

      lo_idx = (lo_idx - self.global_offsets[:-1, 0]).clip(min=1)
      up_idx = (up_idx - self.global_offsets[:-1, 0] -
                dim_offsets[:-1, 1]).clip(min=1)

      for d in range(self.num_dims):
        gshape[d] = up_idx[d] - lo_idx[d] + 1

      slices = [slice(lo_idx[d] - 1, up_idx[d])
                for d in range(self.num_dims)]  # Gkeyll is 1-indexed
    return out.reshape(gshape, order="C"), tuple(slices)

  def _read_t1_v1_data(self) -> np.ndarray:
    """Read field data for file type 1."""
    data, _ = self._get_data(self.asize * self.num_comps)
    return data

  def _read_t3_v1_data(self) -> np.ndarray:
    """Read field data for file type 3."""
    # get the number of stored ranges
    num_range = np.fromfile(self.file_name,
                            dtype=self.dti,
                            count=1,
                            offset=self.offset)[0]
    self.offset += 8

    gshape = np.ones(self.num_dims + 1, dtype=self.dti)
    for d in range(self.num_dims):
      gshape[d] = self.cells[d]
    gshape[-1] = self.num_comps
    data = np.zeros(gshape, dtype=self.dtf)  # Allocate space for the data

    for _ in range(num_range):
      lo_idx = np.fromfile(self.file_name,
                           dtype=self.dti,
                           count=self.num_dims,
                           offset=self.offset)
      self.offset += self.num_dims * 8
      up_idx = np.fromfile(self.file_name,
                           dtype=self.dti,
                           count=self.num_dims,
                           offset=self.offset)
      self.offset += self.num_dims * 8

      asize = np.fromfile(self.file_name,
                          dtype=self.dti,
                          count=1,
                          offset=self.offset)[0]
      self.offset += 8
      #data_raw = np.fromfile(self.file_name, dtype=self.dtf, count=asize*self.num_comps,
      #    offset=self.offset)
      #self.offset += asize * self.num_comps * self.doffset
      data_block, slices = self._get_data(count=asize *
                                          self.orig_size_array[-1],
                                          lo_idx=lo_idx,
                                          up_idx=up_idx)

      if len(data_block) == 0:
        continue
      data[slices] = data_block
    return data

  def _read_t2_v1(self) -> Tuple[list, np.ndarray]:
    """Read dynvector data for file type 2."""
    cells = 0
    time = np.array([])
    data = np.array([[]])
    while True:  # Python does not have DO .. WHILE loop
      elem_sz_raw = int(
          np.fromfile(self.file_name,
                      dtype=self.dti,
                      count=1,
                      offset=self.offset)[0])
      num_comps = int(elem_sz_raw / self.doffset)
      self.offset += 8

      loop_cells = int(
          np.fromfile(self.file_name,
                      dtype=self.dti,
                      count=1,
                      offset=self.offset)[0])
      self.offset += 8

      loop_time = np.fromfile(self.file_name,
                              dtype=self.dtf,
                              count=loop_cells,
                              offset=self.offset)
      self.offset += loop_cells * 8

      data_raw = np.fromfile(self.file_name,
                             dtype=self.dtf,
                             count=num_comps * loop_cells,
                             offset=self.offset)
      self.offset += loop_cells * elem_sz_raw
      gshape = np.array((loop_cells, num_comps), dtype=self.dti)

      time = np.append(time, loop_time)
      if cells == 0:
        data = data_raw.reshape(gshape, order="C")
      else:
        data = np.append(data, data_raw.reshape(gshape, order="C"), axis=0)
      cells += loop_cells
      if self.offset >= os.path.getsize(self.file_name):
        break
      self._read_header()
      if self.file_type != 2:
        raise TypeError("Inconsitent data in g0 dynVector file.")
    self.cells = [cells]
    self.lower = np.atleast_1d(time.min())
    self.upper = np.atleast_1d(time.max())
    return time, data

  # ---- Exposed functions -----
  def preload(self) -> None:
    """Loads metadata."""
    self._read_header()
    if self.file_type == 1 or self.file_type == 3 or self.version == 0:
      self._read_t1t3_v1_domain()
      if self.ctx:
        self.ctx["cells"] = self.cells
        self.ctx["lower"] = self.lower
        self.ctx["upper"] = self.upper
        self.ctx["num_comps"] = self.num_comps

  def load(self) -> Tuple[list, np.ndarray]:
    """Loads data.

    Returns:
      A tuple including a grid list and a data NumPy array

    Notes:
      Needs to be called after the preload.
    """
    time = None
    if self.file_type == 1 or self.version == 0:
      data = self._read_t1_v1_data()
    elif self.file_type == 2:
      time, data = self._read_t2_v1()
    elif self.file_type == 3:
      data = self._read_t3_v1_data()
    else:
      raise TypeError("This g0 format is not presently supported")

    # Load or construct grid
    num_dims = len(self.cells)
    if time is not None:
      grid = [time]
      if self.ctx:
        self.ctx["grid_type"] = "nodal"
    else:  # Create sparse unifrom grid
      mapping.adjust_for_ghost_cells(self.lower, self.upper, self.cells,
                                     data.shape)
      grid = mapping.uniform_grid(self.lower, self.upper, self.cells)
      if self.ctx:
        self.ctx["grid_type"] = "uniform"

    return grid, data
