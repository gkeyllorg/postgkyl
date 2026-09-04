"""``GkylArray`` -- the Python owner of a native ``gkyl_array``.

The handle is a ``PyCapsule`` produced by the ``_gpython`` extension; its
destructor releases the C array, and zero-copy constructions pin the backing
NumPy buffer inside the capsule for the lifetime of the C view. Views of the
data take the capsule as their ndarray ``base``, so a view can never outlive
the native memory it aliases. No raw pointer ever reaches Python.
"""

from __future__ import annotations

import numpy as np

from . import _lib


class GkylArray:
  """Owns one native ``gkyl_array`` (double-precision) via its capsule."""

  def __init__(self, cap):
    self._cap = cap

  # ------------------------------------------------------------ constructors
  @classmethod
  def alloc(cls, ncomp: int, size: int) -> "GkylArray":
    """gkyl-owned zeroed array of ``size`` cells x ``ncomp`` doubles.

    Raises:
      ValueError: ``ncomp`` or ``size`` is not positive. Gkeyll's own
        allocator asserts on a zero-byte buffer at *release* time (an abort,
        not a Python exception) -- refusing here turns a process crash into a
        clean, early error.
    """
    if ncomp <= 0 or size <= 0:
      raise ValueError(f"GkylArray.alloc: ncomp={ncomp} and size={size} "
                       "must both be positive (Gkeyll cannot allocate a "
                       "zero-sized array)")
    return cls(_lib.require().array_new(ncomp, size))

  @classmethod
  def from_numpy(cls, values: np.ndarray) -> "GkylArray":
    """Zero-copy ``gkyl_array`` view of a ``(cells..., ncomp)`` NumPy array.

    The buffer is pinned inside the capsule for the C array's lifetime; data
    is made contiguous float64 first (copying only if needed).

    Raises:
      ValueError: ``values`` has fewer than 1 dimension, or is empty (see
        :meth:`alloc` -- an empty buffer crashes Gkeyll's allocator on
        release rather than raising).
    """
    buf = np.ascontiguousarray(values, dtype=np.float64)
    if buf.ndim < 1:
      raise ValueError("GkylArray.from_numpy: need at least a 1-D "
                       "(…, ncomp) array")
    if buf.size == 0:
      raise ValueError("GkylArray.from_numpy: array is empty (Gkeyll "
                       "cannot allocate a zero-sized array)")
    return cls(_lib.require().array_from_numpy(buf))

  def clone(self) -> "GkylArray":
    """Deep copy through ``gkyl_array_clone`` (gkyl-owned)."""
    return GkylArray(_lib.require().array_clone(self._cap))

  # ------------------------------------------------------------------ shape
  @property
  def ncomp(self) -> int:
    return int(_lib.require().array_ncomp(self._cap))

  @property
  def size(self) -> int:
    return int(_lib.require().array_size(self._cap))

  # ---------------------------------------------------------------- readout
  def view(self, cells=None) -> np.ndarray:
    """Read-only NumPy view of the C buffer, shaped ``(*cells, ncomp)``.

    The view's ``base`` chain holds the owning capsule, so
    ``dataset.values.copy()`` on a temporary dataset is safe -- the memory
    cannot be released while any view is reachable. Mutation must go through
    the kernels, never the view.
    """
    flat = _lib.require().array_view(self._cap)
    if cells is None:
      return flat
    return flat.reshape(tuple(int(c) for c in cells) + (flat.shape[-1], ))

  def to_numpy(self, cells=None) -> np.ndarray:
    """By-value copy out of the C buffer (what the ``interpolate`` bridge returns)."""
    return np.array(self.view(cells), copy=True)

  def __repr__(self) -> str:
    return f"<GkylArray {self.size} cells x {self.ncomp} comps (native)>"
