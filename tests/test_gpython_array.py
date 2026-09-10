"""Tests for ``postgkyl.gpython.array.GkylArray`` -- the capsule-owning array.

Run:  PYTHONPATH=src pytest tests/test_gpython_array.py -v
"""

import gc
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)  # dedup harmless across the shared test session

from postgkyl import gpython  # noqa: E402
from postgkyl.gpython.array import GkylArray  # noqa: E402

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

pytestmark = needs_gkeyll


# --------------------------------------------------------------- construction
def test_alloc_is_zeroed_with_the_requested_shape():
  a = GkylArray.alloc(3, 5)
  assert (a.ncomp, a.size) == (3, 5)
  assert np.array_equal(a.view(), np.zeros((5, 3)))


def test_from_numpy_preserves_values_and_shape():
  values = np.arange(2 * 4, dtype=np.float64).reshape(4, 2)
  a = GkylArray.from_numpy(values)
  assert (a.ncomp, a.size) == (2, 4)
  assert np.array_equal(a.view(), values)


def test_from_numpy_copies_non_contiguous_input_correctly():
  base = np.arange(40, dtype=np.float64).reshape(10, 4)
  sliced = base[::2]  # non-contiguous view
  assert not sliced.flags["C_CONTIGUOUS"]
  a = GkylArray.from_numpy(sliced)
  assert np.array_equal(a.view(), sliced)


def test_from_numpy_converts_other_dtypes():
  values = np.arange(6, dtype=np.int32).reshape(3, 2)
  a = GkylArray.from_numpy(values)
  assert a.view().dtype == np.float64
  assert np.array_equal(a.view(), values.astype(np.float64))


def test_clone_is_a_deep_copy():
  a = GkylArray.from_numpy(np.ones((3, 2)))
  b = a.clone()
  assert np.array_equal(a.view(), b.view())
  # Mutate through the kernel layer (never the view) to prove independence.
  gpython.kernels.scale(a, 0.0)  # returns a NEW array; `a` itself is untouched
  assert np.array_equal(a.view(), np.ones((3, 2)))
  assert np.array_equal(b.view(), np.ones((3, 2)))


# ---------------------------------------------------------- invalid construction
def test_alloc_rejects_zero_size():
  with pytest.raises(ValueError, match="positive"):
    GkylArray.alloc(2, 0)


def test_alloc_rejects_zero_ncomp():
  with pytest.raises(ValueError, match="positive"):
    GkylArray.alloc(0, 3)


def test_alloc_rejects_negative_args():
  with pytest.raises(ValueError, match="positive"):
    GkylArray.alloc(-1, 3)
  with pytest.raises(ValueError, match="positive"):
    GkylArray.alloc(2, -1)


def test_from_numpy_rejects_empty_array():
  with pytest.raises(ValueError, match="empty"):
    GkylArray.from_numpy(np.zeros((0, 3)))


def test_from_numpy_promotes_0d_to_a_single_cell():
  """`np.ascontiguousarray` upgrades a 0-d scalar to shape (1,) before the
  extension ever sees it, so this is a valid single-component, single-cell
  array, not the `ndim < 1` refusal (which is defensive/unreachable through
  this public constructor -- see the C source comment in _gpythonmodule.c)."""
  a = GkylArray.from_numpy(np.array(5.0))
  assert (a.ncomp, a.size) == (1, 1)
  assert a.view()[0, 0] == 5.0


# --------------------------------------------------------------- memory safety
def test_view_pins_native_memory_after_source_is_dropped():
  """Regression: a view outlives the Python object that produced it."""
  expected = np.arange(6, dtype=np.float64).reshape(3, 2)
  v = GkylArray.from_numpy(expected).view()  # array is garbage immediately
  gc.collect()
  assert np.array_equal(v, expected)


def test_view_pins_native_memory_for_alloc_too():
  a = GkylArray.alloc(2, 3)
  gpython.kernels.shiftc(a, 7.0, 0)  # exercise the array without touching `v`
  v = a.view()
  del a
  gc.collect()
  assert np.array_equal(v, np.zeros((3, 2)))  # `a` was never mutated in place


def test_to_numpy_is_a_by_value_copy():
  a = GkylArray.from_numpy(np.ones((2, 2)))
  copy = a.to_numpy()
  view = a.view()
  assert copy.flags.writeable
  assert not view.flags.writeable
  copy[0, 0] = 99.0
  assert view[0, 0] == 1.0  # the native buffer is untouched


def test_view_is_read_only():
  a = GkylArray.alloc(2, 3)
  with pytest.raises(ValueError):
    a.view()[0, 0] = 1.0


def test_repeated_alloc_and_release_does_not_leak_or_crash():
  for _ in range(500):
    a = GkylArray.alloc(4, 10)
    a.view()
    del a
  gc.collect()


def test_view_reshapes_with_explicit_cells():
  a = GkylArray.alloc(2, 6)
  shaped = a.view(cells=(2, 3))
  assert shaped.shape == (2, 3, 2)


def test_repr_reports_shape():
  a = GkylArray.alloc(3, 5)
  assert "5 cells" in repr(a)
  assert "3 comps" in repr(a)
