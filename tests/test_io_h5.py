"""Tests for ``postgkyl.io.gkyl_h5_reader`` and ``postgkyl.io.flash_h5_reader``.

The old test corpus has no ``.h5`` fixtures, so these build tiny files with
``tables`` directly matching each reader's expected on-disk layout (derived
from the reader source: a Gkeyll "frame" file needs a ``/StructGrid`` group
with ``vsLowerBounds``/``vsUpperBounds``/``vsNumCells`` attributes plus a
``/StructGridField`` array; a "diagnostic" file needs ``/DataStruct/timeMesh``
and ``/DataStruct/data``; a FLASH file needs ``coordinates``/``block
size``/``node type`` plus the named field array).

Run:  PYTHONPATH=src pytest tests/test_io_h5.py -v
"""

import os
import sys

import numpy as np
import pytest
import tables

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)  # dedup harmless across the shared test session

from postgkyl import io  # noqa: E402
from postgkyl.io.gkyl_h5_reader import GkylH5Reader  # noqa: E402
from postgkyl.io.flash_h5_reader import FlashH5Reader  # noqa: E402


# --------------------------------------------------------------------- gkyl h5
def _write_gkyl_h5_frame(path, lower, upper, cells, data, time=None):
  fh = tables.open_file(path, "w")
  grp = fh.create_group("/", "StructGrid", "grid")
  grp._v_attrs.vsLowerBounds = np.asarray(lower)
  grp._v_attrs.vsUpperBounds = np.asarray(upper)
  grp._v_attrs.vsNumCells = np.asarray(cells)
  fh.create_array("/", "StructGridField", data)
  if time is not None:
    tgrp = fh.create_group("/", "timeData", "time")
    tgrp._v_attrs.vsTime = time
  fh.close()


def _write_gkyl_h5_diagnostic(path, time_mesh, data):
  fh = tables.open_file(path, "w")
  grp = fh.create_group("/", "DataStruct", "diag")
  fh.create_array(grp, "timeMesh", time_mesh)
  fh.create_array(grp, "data", data)
  fh.close()


def test_gkyl_h5_frame_roundtrip(tmp_path):
  path = str(tmp_path / "frame.h5")
  data = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3)
  _write_gkyl_h5_frame(path, [0.0, -1.0], [2.0, 1.0], [4, 2], data, time=0.5)

  grid, out = io.read(path)
  np.testing.assert_allclose(out, data)
  assert grid[0].shape == (5, )
  assert grid[1].shape == (3, )
  np.testing.assert_allclose(grid[0], np.linspace(0.0, 2.0, 5))
  np.testing.assert_allclose(grid[1], np.linspace(-1.0, 1.0, 3))


def test_gkyl_h5_frame_ctx_and_time(tmp_path):
  path = str(tmp_path / "frame.h5")
  data = np.ones((4, 2, 1))
  _write_gkyl_h5_frame(path, [0.0, 0.0], [1.0, 1.0], [4, 2], data, time=1.25)

  r = GkylH5Reader(path, ctx={})
  assert r.is_compatible()
  r.preload()
  _, out = r.load()
  assert r.ctx["time"] == pytest.approx(1.25)
  np.testing.assert_array_equal(r.ctx["cells"], [4, 2])
  assert r.ctx["num_comps"] == 1
  assert r.ctx["grid_type"] == "uniform"
  assert out.shape == (4, 2, 1)


def test_gkyl_h5_diagnostic(tmp_path):
  path = str(tmp_path / "diag.h5")
  time_mesh = np.linspace(0.0, 1.0, 5)
  data = np.arange(5 * 3, dtype=np.float64).reshape(5, 3)
  _write_gkyl_h5_diagnostic(path, time_mesh, data)

  r = GkylH5Reader(path, ctx={})
  assert r.is_compatible()
  assert r.is_diagnostic and not r.is_frame
  r.preload()
  grid, out = r.load()
  np.testing.assert_allclose(out, data)
  assert r.ctx["num_comps"] == 3
  assert grid[0].shape == (6, )  # uniform pseudo-grid over [time[0], time[-1]]


def test_gkyl_h5_is_compatible_false_for_unrelated_file(tmp_path):
  path = str(tmp_path / "empty.h5")
  fh = tables.open_file(path, "w")
  fh.create_array("/", "SomethingElse", np.zeros(3))
  fh.close()
  assert GkylH5Reader(path, ctx={}).is_compatible() is False


def test_gkyl_h5_is_compatible_false_for_a_non_hdf5_file(tmp_path):
  path = tmp_path / "not_hdf5.h5"
  path.write_bytes(b"definitely not an hdf5 file")
  assert GkylH5Reader(str(path), ctx={}).is_compatible() is False
  assert GkylH5Reader("/no/such/file.h5", ctx={}).is_compatible() is False


# ------------------------------------------------------------------- flash h5
def _write_flash_h5(path,
                    *,
                    num_blocks=2,
                    nxb=4,
                    nyb=4,
                    var_name="dens",
                    seed=0):
  """FLASH stores blocks pre-transposed on disk relative to what the reader
  uses after its own ``.transpose()`` call -- see ``FlashH5Reader._read_frame``."""
  rng = np.random.default_rng(seed)
  coord = np.array([[0.25, 0.25], [0.75, 0.25]][:num_blocks])  # (N, 2)
  bsize = np.full((num_blocks, 2), 0.5)  # (N, 2)
  ntype = np.ones(num_blocks, dtype=np.int32)  # (N,) all leaf blocks
  bdata = rng.normal(size=(num_blocks, 1, nyb, nxb))  # -> (nxb,nyb,1,N)

  fh = tables.open_file(path, "w")
  fh.create_array("/", "coordinates", coord)
  fh.create_array("/", "block size", bsize)
  fh.create_array("/", "node type", ntype)
  fh.create_array("/", var_name, bdata)
  fh.close()


@pytest.mark.filterwarnings(
    "ignore:object name is not a valid Python identifier:tables.exceptions.NaturalNameWarning"
)
def test_flash_h5_frame_roundtrip(tmp_path):
  path = str(tmp_path / "flash.h5")
  _write_flash_h5(path, var_name="dens")

  r = FlashH5Reader(path, ctx={}, var_name="dens")
  assert r.is_compatible()
  r.preload()
  grid, out = r.load()
  assert out.ndim == 3  # (nx, ny, 1)
  assert out.shape[-1] == 1
  assert r.ctx["grid_type"] == "uniform"
  np.testing.assert_array_equal(r.ctx["cells"], out.shape[:-1])
  assert len(grid) == 2
  assert grid[0].shape == (out.shape[0] + 1, )
  assert grid[1].shape == (out.shape[1] + 1, )


@pytest.mark.filterwarnings(
    "ignore:object name is not a valid Python identifier:tables.exceptions.NaturalNameWarning"
)
def test_flash_h5_load_requires_var_name(tmp_path):
  path = str(tmp_path / "flash.h5")
  _write_flash_h5(path, var_name="dens")
  r = FlashH5Reader(path, ctx={})  # var_name defaults to None
  assert r.is_compatible()
  r.preload()
  with pytest.raises(ValueError, match="requires 'var_name'"):
    r.load()


def test_flash_h5_is_compatible_false_without_coordinates(tmp_path):
  path = str(tmp_path / "not_flash.h5")
  fh = tables.open_file(path, "w")
  fh.create_array("/", "SomethingElse", np.zeros(3))
  fh.close()
  assert FlashH5Reader(path, ctx={}).is_compatible() is False


def test_flash_h5_is_compatible_false_for_a_non_hdf5_file(tmp_path):
  path = tmp_path / "not_hdf5.h5"
  path.write_bytes(b"definitely not an hdf5 file")
  assert FlashH5Reader(str(path), ctx={}).is_compatible() is False
