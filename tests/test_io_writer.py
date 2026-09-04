"""Tests for ``postgkyl.io.writer`` -- the vtk format and series-file behavior.

npy/txt/gkyl round trips and error paths are covered in
``tests/test_coverage_io.py``; this file focuses on what layer 04 adds: the
``vtk`` extension and its ParaView ``.series`` sidecar, plus a byte-exact
gkyl round trip through ``io.read``.

Run:  PYTHONPATH=src pytest tests/test_io_writer.py -v
"""

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)  # dedup harmless across the shared test session

import matplotlib

matplotlib.use("Agg")

import postgkyl as pg  # noqa: E402
from postgkyl import io  # noqa: E402
from postgkyl.io import writer  # noqa: E402
from postgkyl.gdatastate.gdatastate import GDataState  # noqa: E402

DATA = os.path.join(ROOT, "tests", "test_data")
F1 = os.path.join(
    DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")
F2D = os.path.join(DATA, "generated", "2d_ms_p1.gkyl")


def _make_state(grid, values, *, time=None, frame=None):
  d = GDataState()
  d.push(grid, values)
  if time is not None:
    d.ctx["time"] = time
  if frame is not None:
    d.ctx["frame"] = frame
  return d


# --------------------------------------------------------------------- vtk
def test_vtk_writes_a_well_formed_legacy_header_1d(tmp_path):
  a = pg.load(F1).interpolate().select(comp=0)
  out = writer.save(a, out_name=str(tmp_path / "out1d.vtk"), extension="vtk")
  assert os.path.exists(out)
  with open(out, "rb") as fh:
    header = fh.read(96)
  assert header.startswith(b"# vtk DataFile Version")
  assert b"STRUCTURED_GRID" in header


def test_vtk_writes_a_well_formed_legacy_header_2d(tmp_path):
  b = pg.load(F2D).interpolate().select(comp=0)
  out = writer.save(b, out_name=str(tmp_path / "out2d.vtk"), extension="vtk")
  assert os.path.exists(out)
  with open(out, "rb") as fh:
    header = fh.read(96)
  assert header.startswith(b"# vtk DataFile Version")


def test_vtk_writes_a_3d_volume(tmp_path):
  grid = [
      np.linspace(0.0, 1.0, 3),
      np.linspace(0.0, 1.0, 4),
      np.linspace(0.0, 1.0, 5)
  ]
  values = np.arange(2 * 3 * 4 * 1, dtype=float).reshape(2, 3, 4, 1)
  d = _make_state(grid, values)
  out = writer.save(d, out_name=str(tmp_path / "out3d.vtk"), extension="vtk")
  assert os.path.exists(out)
  with open(out, "rb") as fh:
    header = fh.read(96)
  assert header.startswith(b"# vtk DataFile Version")


def test_vtk_rejects_unsupported_dimensionality(tmp_path):
  from postgkyl.io.writer import _write_vtk
  grid = [np.linspace(0, 1, 2)] * 4
  values = np.ones((1, 1, 1, 1, 1))
  d = _make_state(grid, values)
  with pytest.raises(ValueError, match="1-3 dimensions"):
    _write_vtk(str(tmp_path / "bad.vtk"), d, 4, d.num_cells, values)


def test_vtk_series_file_accumulates_entries_across_two_writes(tmp_path):
  grid = [np.linspace(0.0, 1.0, 4)]
  values = np.array([[1.0], [2.0], [3.0]])

  a = _make_state(grid, values, time=0.1)
  out1 = writer.save(a,
                     out_name=str(tmp_path / "solution_0001.vtk"),
                     extension="vtk")
  b = _make_state(grid, values, time=0.2)
  out2 = writer.save(b,
                     out_name=str(tmp_path / "solution_0002.vtk"),
                     extension="vtk")

  series_path = tmp_path / "solution.vtk.series"
  assert series_path.exists()
  with open(series_path) as fh:
    series = json.load(fh)
  assert series["file-series-version"] == "1.0"
  assert series["files"] == [
      {
          "name": os.path.basename(out1),
          "time": 0.1
      },
      {
          "name": os.path.basename(out2),
          "time": 0.2
      },
  ]


def test_vtk_series_file_updates_existing_entry_in_place(tmp_path):
  """Re-writing the same frame number refreshes its time instead of
  duplicating the entry."""
  grid = [np.linspace(0.0, 1.0, 4)]
  values = np.array([[1.0], [2.0], [3.0]])

  a = _make_state(grid, values, time=0.1)
  writer.save(a, out_name=str(tmp_path / "solution_0001.vtk"), extension="vtk")
  a2 = _make_state(grid, values, time=0.15)
  writer.save(a2, out_name=str(tmp_path / "solution_0001.vtk"), extension="vtk")

  with open(tmp_path / "solution.vtk.series") as fh:
    series = json.load(fh)
  assert len(series["files"]) == 1
  assert series["files"][0]["time"] == pytest.approx(0.15)


def test_vtk_series_uses_frame_when_time_is_absent(tmp_path):
  grid = [np.linspace(0.0, 1.0, 4)]
  values = np.array([[1.0], [2.0], [3.0]])
  a = _make_state(grid, values, frame=3)
  writer.save(a, out_name=str(tmp_path / "run_0003.vtk"), extension="vtk")
  with open(tmp_path / "run.vtk.series") as fh:
    series = json.load(fh)
  assert series["files"][0]["time"] == pytest.approx(3.0)


def test_vtk_series_recovers_from_a_corrupt_sidecar(tmp_path):
  grid = [np.linspace(0.0, 1.0, 4)]
  values = np.array([[1.0], [2.0], [3.0]])
  (tmp_path / "bad.vtk.series").write_text("not valid json{{{")
  a = _make_state(grid, values, time=0.5)
  writer.save(a, out_name=str(tmp_path / "bad_0001.vtk"), extension="vtk")
  with open(tmp_path / "bad.vtk.series") as fh:
    series = json.load(fh)
  assert len(series["files"]) == 1


# ------------------------------------------------------------------- gkyl rt
def test_gkyl_roundtrip_preserves_grid_and_values_exactly(tmp_path):
  """``io.read`` is exercised both directly (grid) and through ``pg.load``
  (values, via the ``.values`` property that abstracts the gkyl/numpy
  backend split -- see gdatastate/state.py) since a written already-interpolated
  field still carries file_type == 1 and so is picked up again by whichever
  reader is first compatible (GkylCReader when the FFI is available)."""
  a = pg.load(F1).interpolate().select(comp=0)
  out = writer.save(a, out_name=str(tmp_path / "rt.gkyl"), extension="gkyl")

  grid, _ = io.read(out)
  for g_out, g_in in zip(a.grid, grid):
    np.testing.assert_allclose(g_in, g_out)

  back = pg.load(out)
  np.testing.assert_allclose(np.asarray(back.values), np.asarray(a.values))
