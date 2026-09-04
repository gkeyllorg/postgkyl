"""Tests for ``postgkyl.gpython.rio`` -- file I/O through Gkeyll's ``gkyl_array_rio``.

Run:  PYTHONPATH=src pytest tests/test_gpython_rio.py -v
"""

import glob
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)  # dedup harmless across the shared test session

from postgkyl import gpython  # noqa: E402
from postgkyl.gpython import rio  # noqa: E402
from postgkyl.gpython.array import GkylArray  # noqa: E402
from postgkyl.io.gkyl_reader import GkylReader  # noqa: E402

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

DATA = os.path.join(ROOT, "tests", "test_data")
FIELD_FILES = sorted(
    glob.glob(os.path.join(DATA, "rt_gk_tcv_iwl_1x2v_p1-*.gkyl")))
# Excludes the generated dynvector fixture (energy_dynvec.gkyl): this module's
# cross-checks are specifically about *field* files -- a dynvector already has
# its own coverage elsewhere in this module
# (test_file_type_of_a_dynvec_file_is_not_a_field_type, via a purpose-built
# temp fixture), so it never belongs in this parametrization.
GENERATED_FILES = sorted(
    f for f in glob.glob(os.path.join(DATA, "generated", "*.gkyl"))
    if not os.path.basename(f).endswith("_dynvec.gkyl"))

pytestmark = needs_gkeyll

# A non-field (dynvector) file, used below to check that `file_type` correctly
# excludes it from the field-file cross-check.
_NON_FIELD_FILE = None
if gpython.available():
  from postgkyl.gpython import rio as _rio
  _dynvec_dir = tempfile.mkdtemp()
  _NON_FIELD_FILE = os.path.join(_dynvec_dir, "not_a_field_dynvec.gkyl")
  _rio.write_dynvec(_NON_FIELD_FILE, np.array([0.0, 1.0]),
                    np.array([[1.0], [2.0]]))


# ------------------------------------------------------ cross-check vs GkylReader
def _read_with_pure_python(path):
  r = GkylReader(path, ctx={})
  r.preload()
  return r.load()


@pytest.mark.parametrize("path",
                         FIELD_FILES + GENERATED_FILES,
                         ids=os.path.basename)
def test_read_field_matches_the_pure_python_reader(path):
  """The strongest test in this layer: for every fixture the C reader
  accepts, its grid/cells/coefficients must agree exactly with the
  independent pure-Python implementation reading the same bytes."""
  py_grid, py_values = _read_with_pure_python(path)

  assert rio.file_type(path) in rio.FIELD_FILE_TYPES
  c_grid, c_arr = rio.read_field(path)
  c_values = c_arr.to_numpy(cells=c_grid["cells"])

  assert c_grid["ndim"] == len(py_grid)
  for d in range(c_grid["ndim"]):
    np.testing.assert_allclose(
        py_grid[d],
        np.asarray(
            np.linspace(c_grid["lower"][d], c_grid["upper"][d],
                        int(c_grid["cells"][d]) + 1)))
  np.testing.assert_allclose(c_values.squeeze(), np.squeeze(py_values))


def test_file_type_of_a_field_file():
  assert rio.file_type(FIELD_FILES[0]) in rio.FIELD_FILE_TYPES


def test_file_type_of_a_dynvec_file_is_not_a_field_type():
  assert rio.file_type(_NON_FIELD_FILE) not in rio.FIELD_FILE_TYPES


def test_file_type_nonexistent_path_returns_sentinel():
  """`file_type` is documented to return -1 for "not a gkyl file" rather
  than raise -- a nonexistent path is exactly that case."""
  assert rio.file_type("/no/such/file.gkyl") == -1


def test_file_type_non_gkyl_file_returns_sentinel(tmp_path):
  bogus = tmp_path / "not_a_gkyl_file.gkyl"
  bogus.write_bytes(b"definitely not a gkyl binary file")
  assert rio.file_type(str(bogus)) == -1


def test_read_header_nonexistent_path_raises():
  with pytest.raises(OSError):
    rio.read_header("/no/such/file.gkyl")


def test_read_field_nonexistent_path_raises():
  with pytest.raises(OSError):
    rio.read_field("/no/such/file.gkyl")


def test_read_field_non_gkyl_file_refuses_cleanly(tmp_path):
  bogus = tmp_path / "not_a_gkyl_file.gkyl"
  bogus.write_bytes(b"this is definitely not a gkyl binary file, at all!!")
  with pytest.raises(OSError):
    rio.read_field(str(bogus))


def test_read_header_reports_metadata_for_a_modal_file():
  grid, ftype, meta, esznc, tot_cells = rio.read_header(FIELD_FILES[0])
  assert ftype in rio.FIELD_FILE_TYPES
  assert esznc > 0
  assert tot_cells == int(np.prod(grid["cells"]))
  assert isinstance(meta,
                    bytes) and len(meta) > 0  # this fixture has msgpack meta


# -------------------------------------------------------------------- writing
def test_write_field_round_trips_bit_exactly(tmp_path):
  rng = np.random.default_rng(0)
  values = rng.normal(size=(4, 3, 2)).astype(np.float64)
  arr = GkylArray.from_numpy(values)
  grid = {
      "lower": np.array([0.0, -1.0]),
      "upper": np.array([2.0, 1.0]),
      "cells": np.array([4, 3])
  }
  path = str(tmp_path / "roundtrip.gkyl")
  rio.write_field(path, grid, arr)

  back_grid, back_arr = rio.read_field(path)
  np.testing.assert_array_equal(back_grid["lower"], grid["lower"])
  np.testing.assert_array_equal(back_grid["upper"], grid["upper"])
  np.testing.assert_array_equal(back_grid["cells"], grid["cells"])
  np.testing.assert_array_equal(back_arr.to_numpy(), arr.to_numpy())


def test_write_field_with_metadata_round_trips_the_bytes(tmp_path):
  import msgpack
  meta = msgpack.packb({"polyOrder": 1, "basisType": "serendipity"})
  arr = GkylArray.from_numpy(np.ones((3, 2)))
  grid = {
      "lower": np.array([0.0]),
      "upper": np.array([1.0]),
      "cells": np.array([3])
  }
  path = str(tmp_path / "with_meta.gkyl")
  rio.write_field(path, grid, arr, meta=meta)

  _, ftype, back_meta, _, _ = rio.read_header(path)
  assert msgpack.unpackb(back_meta) == {
      "polyOrder": 1,
      "basisType": "serendipity"
  }


def test_write_field_is_readable_by_the_pure_python_reader(tmp_path):
  """Interoperability: a file this floor writes must be a real, standard
  .gkyl file, not merely self-consistent with this floor's own reader."""
  arr = GkylArray.from_numpy(np.arange(10, dtype=np.float64).reshape(5, 2))
  grid = {
      "lower": np.array([0.0]),
      "upper": np.array([5.0]),
      "cells": np.array([5])
  }
  path = str(tmp_path / "interop.gkyl")
  rio.write_field(path, grid, arr)

  py_grid, py_values = _read_with_pure_python(path)
  np.testing.assert_allclose(py_grid[0], np.linspace(0.0, 5.0, 6))
  np.testing.assert_allclose(np.squeeze(py_values), arr.to_numpy())


def test_write_field_rejects_grid_array_mismatch(tmp_path):
  arr = GkylArray.alloc(2, 4)
  grid = {
      "lower": np.array([0.0]),
      "upper": np.array([1.0]),
      "cells": np.array([5])
  }
  with pytest.raises(ValueError, match="do not cover"):
    rio.write_field(str(tmp_path / "bad.gkyl"), grid, arr)


def test_write_field_bad_path_raises_oserror():
  arr = GkylArray.alloc(2, 3)
  grid = {
      "lower": np.array([0.0]),
      "upper": np.array([1.0]),
      "cells": np.array([3])
  }
  with pytest.raises(OSError):
    rio.write_field("/no/such/directory/out.gkyl", grid, arr)


# ------------------------------------------------------------------ dynvector
def test_dynvec_write_read_round_trip(tmp_path):
  time = np.array([0.0, 0.1, 0.25, 0.4])
  data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],
                   [10.0, 11.0, 12.0]])
  path = str(tmp_path / "series.gkyl")
  rio.write_dynvec(path, time, data)

  back_time, back_data = rio.read_dynvec(path)
  np.testing.assert_allclose(back_time, time)
  np.testing.assert_allclose(back_data, data)


def test_dynvec_write_read_round_trip_single_component(tmp_path):
  time = np.array([0.0, 1.0, 2.0])
  data = np.array([1.5, -2.5, 3.5])
  path = str(tmp_path / "series_1c.gkyl")
  rio.write_dynvec(path, time, data)

  back_time, back_data = rio.read_dynvec(path)
  np.testing.assert_allclose(back_time, time)
  np.testing.assert_allclose(back_data[:, 0], data)


def test_dynvec_write_rejects_length_mismatch(tmp_path):
  time = np.array([0.0, 1.0, 2.0])
  data = np.array([[1.0], [2.0]])  # only 2 rows
  with pytest.raises(ValueError, match="samples"):
    rio.write_dynvec(str(tmp_path / "bad.gkyl"), time, data)


def test_dynvec_read_nonexistent_file_raises():
  with pytest.raises(OSError):
    rio.read_dynvec("/no/such/dynvec.gkyl")


def test_dynvec_read_non_dynvec_file_raises(tmp_path):
  """A well-formed FIELD file is not a dynvector -- must refuse, not
  silently misinterpret the bytes."""
  arr = GkylArray.alloc(2, 3)
  grid = {
      "lower": np.array([0.0]),
      "upper": np.array([1.0]),
      "cells": np.array([3])
  }
  path = str(tmp_path / "field_not_dynvec.gkyl")
  rio.write_field(path, grid, arr)
  with pytest.raises(OSError):
    rio.read_dynvec(path)
