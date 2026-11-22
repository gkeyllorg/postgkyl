"""Tests for ``postgkyl.diagnostics.gk.cell_values``.

On the synthetic three-block run of ``gk_tokamak_fixture`` (with its plate
files), whose fields are bilinear ``a + b x + c y + e x y`` in each block's
logical coordinates: a cell's average is the polynomial at the cell centre,
exactly. The density has no y term, the heat flux on the leg a y term whose
value differs between the leg's two theta cells.

Run: PYTHONPATH=src pytest tests/test_diagnostics_gk_cell_values.py -v
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

import gk_tokamak_fixture as fx

# The package exports the function under the module's name.
cv = importlib.import_module("postgkyl.diagnostics.gk.cell_values")


@pytest.fixture
def run(tmp_path):
  return str(fx.write_plate_files(fx.write_run(tmp_path / "run")))


def centres(block: int) -> tuple[np.ndarray, np.ndarray]:
  """The block's cell centres ``(nr, 1)`` in x and ``(1, nt)`` in y."""
  xe, ye = fx.cell_edges(block)
  return (0.5 * (xe[1:] + xe[:-1]))[:, None], (0.5 * (ye[1:] + ye[:-1]))[None]


def density_cells(block: int, frame: int) -> np.ndarray:
  """The density's cell averages ``(nr, nt)``: ``a + b x`` at the centres."""
  a, b, c, e = fx.density(block, frame)
  assert c == e == 0.0
  x, y = centres(block)
  return np.broadcast_to(a + b * x, (len(x), y.shape[1]))


class TestCellValues:

  def test_every_block_and_frame_summed_over_species(self, run):
    got = cv.cell_values(["M0"], fx.NAME, species=["elc", "ion"], path=run)
    assert (got.name, got.quantity, got.species) == (fx.NAME, ("M0", ), ("elc",
                                                                         "ion"))
    assert got.blocks == (0, 1, 2)
    np.testing.assert_array_equal(got.frames, [0, 1, 2])
    np.testing.assert_array_equal(got.time, fx.FRAME_TIMES)
    for b in got.blocks:
      want = np.stack([2 * density_cells(b, k) for k in range(3)])
      assert got.block(b).shape == want.shape
      np.testing.assert_allclose(got.block(b), want, rtol=1e-14, atol=0)
    # Block 2 at frame 2 holds a negative density: kept as it is.
    assert got.block(2)[2, 0, 0] < 0

  def test_quantities_and_species_sum_in_each_cell(self, run):
    got = cv.cell_values(["qpar", "qperp"],
                         fx.NAME,
                         "0,2",
                         species=["elc", "ion"],
                         path=run,
                         blocks=[fx.LEG])
    assert got.blocks == (fx.LEG, )
    np.testing.assert_array_equal(got.frames, [0, 2])
    x, y = centres(fx.LEG)
    for i, k in enumerate((0, 2)):
      h0, h1 = fx.HEAT_FLUX[k]
      want = h0 + h1 * x + fx.HEAT_FLUX_Y * (y - fx.Y_PLATE)
      np.testing.assert_allclose(got.block(fx.LEG)[i], want, rtol=1e-14, atol=0)

  def test_species_independent_quantity(self, run):
    got = cv.cell_values(["field"], fx.NAME, 1, path=run, blocks=[2, 0])
    assert got.blocks == (2, 0) and got.species is None
    np.testing.assert_array_equal(got.frames, [1])
    for v in got.values:
      np.testing.assert_array_equal(v, 0.0)

  def test_blocks_holding_different_frames_are_refused(self, run):
    for s in ("elc", "ion"):
      (Path(run) / f"{fx.NAME}_b1-{s}_M0_2.gkyl").unlink()
    with pytest.raises(ValueError, match="blocks 0 and 1 hold different"):
      cv.cell_values(["M0"], fx.NAME, species=["elc", "ion"], path=run)
    got = cv.cell_values(["M0"],
                         fx.NAME,
                         species=["elc", "ion"],
                         path=run,
                         blocks=[1])
    np.testing.assert_array_equal(got.frames, [0, 1])

  @pytest.mark.parametrize("blocks,match", [([3], "not \\[3\\]"),
                                            ([-1], "not \\[-1\\]"),
                                            ([], "at least one block")])
  def test_unknown_blocks_are_refused(self, run, blocks, match):
    with pytest.raises(ValueError, match=match):
      cv.cell_values(["field"], fx.NAME, path=run, blocks=blocks)

  def test_bare_strings_are_refused_before_reading(self):
    with pytest.raises(TypeError, match=r"\['M0'\]"):
      cv.cell_values("M0", "no-such-run")
    with pytest.raises(ValueError, match="at least one name"):
      cv.cell_values([], "no-such-run")


class TestRecord:

  @staticmethod
  def record(**kw):
    fields = dict(name="run",
                  path=".",
                  quantity=("M0", ),
                  species=("elc", ),
                  blocks=(0, 3),
                  frames=np.array([4, 6]),
                  time=np.array([1.0, 2.0]),
                  values=(np.zeros((2, 2, 3)), np.ones((2, 1, 2))))
    return cv.CellValues(**{**fields, **kw})

  def test_read_only_copies_and_block_lookup(self):
    values = (np.zeros((2, 2, 3)), np.ones((2, 1, 2)))
    r = self.record(values=values)
    values[0][0, 0, 0] = 5.0
    assert r.block(0)[0, 0, 0] == 0.0
    np.testing.assert_array_equal(r.block(3), 1.0)
    for a in (r.frames, r.time, r.block(0)):
      with pytest.raises(ValueError):
        a[0] = 0
    with pytest.raises(ValueError, match="not 1"):
      r.block(1)
    assert "2 frames (4 to 6)" in str(r) and "2x3" in str(r)

  @pytest.mark.parametrize("kw", [
      dict(values=(np.zeros((2, 2, 3)), )),
      dict(values=(np.zeros((2, 2, 3)), np.ones((3, 1, 2)))),
      dict(values=(np.zeros((2, 2)), np.ones((2, 1, 2)))),
  ])
  def test_values_must_be_one_frames_rows_cells_array_per_block(self, kw):
    with pytest.raises(ValueError, match="one \\(frames = 2"):
      self.record(**kw)

  @pytest.mark.parametrize("kw", [
      dict(time=np.array([1.0])),
      dict(frames=np.array([], int),
           time=np.array([]),
           values=(np.zeros((0, 2, 3)), np.ones((0, 1, 2)))),
  ])
  def test_frames_and_time_one_per_frame(self, kw):
    with pytest.raises(ValueError, match="one value per frame"):
      self.record(**kw)
