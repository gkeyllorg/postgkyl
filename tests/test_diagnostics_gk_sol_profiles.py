"""Tests for ``postgkyl.diagnostics.gk.sol_profiles``.

The geometric core is checked on synthetic blocks of corner nodes whose
answers are known in closed form:

- Node rows are straight in (R, Z) and R is linear in Z along each row, so a
  row's Z = 0 crossing, found by linear interpolation between its bracketing
  nodes, is exact: ``R = R0 + dr * i`` for node row ``i`` (inboard rows run
  to smaller R).
- Cell averages are linear in each cell centre's Z, ``A_i + B * Zc``, so the
  value interpolated linearly between the two centres bracketing Z = 0 is
  ``A_i`` exactly, on non-uniform theta spacing too.
- Heat-flux profiles are piecewise constant, so the distance holding a
  fraction of their integral follows by hand.
- Frames scale the cell values by their own factor (``SCALE``), so each
  frame's row of a record has its own closed-form answer.

The entry points that read a run are checked on the synthetic three-block run
of ``gk_tokamak_fixture`` (bilinear fields, three frames).

Run: PYTHONPATH=src pytest tests/test_diagnostics_gk_sol_profiles.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import gk_tokamak_fixture as fx
from postgkyl.diagnostics.gk import sol_profiles as sp
from postgkyl.diagnostics.gk.cell_values import CellValues
from postgkyl.diagnostics.gk.multib_layout import (MidplaneStack, MultibLayout,
                                                   Target)

# Theta node Z of row 0 (non-uniform; Z = 0 between nodes 2 and 3) and the
# shift of each further node row, so every row crosses at its own theta.
ZJ = np.array([-0.9, -0.5, -0.15, 0.25, 0.8])
DZ_ROW = 0.02
R0, DR, TILT = 1.3, 0.01, 0.4
A0, A1, B = 5.0, -0.7, 3.0  # cells: A0 + A1 * (global row) + B * Zc
# Three frames, their times, and the factor each scales the cells by.
FRAMES, TIMES = np.array([0, 4, 7]), np.array([0.0, 2.0e-6, 3.5e-6])
SCALE = np.array([1.0, -2.0, 0.5])


def block_nodes(nr: int, row0: int = 0, sign: float = 1.0) -> np.ndarray:
  """Corner nodes ``(nr + 1, nt + 1, 2)`` of global node rows ``row0`` to
  ``row0 + nr``: Z = ZJ + DZ_ROW * row, R = R0 + sign * DR * row + TILT * Z
  (``sign = -1``: rows run inboard, to smaller R)."""
  rows = np.arange(row0, row0 + nr + 1)[:, None]
  Z = ZJ[None, :] + DZ_ROW * rows
  R = R0 + sign * DR * rows + TILT * Z
  return np.stack([R, Z], axis=-1)


def linear_cells(nodes: np.ndarray, row0: int = 0) -> np.ndarray:
  """Cell values ``A0 + A1 * row + B * Zc``, Zc the mean of a cell's
  corners' Z."""
  Z = nodes[..., 1]
  zc = 0.25 * (Z[:-1, :-1] + Z[1:, :-1] + Z[:-1, 1:] + Z[1:, 1:])
  rows = np.arange(row0, row0 + zc.shape[0])[:, None]
  return A0 + A1 * rows + B * zc


def over_frames(cells: np.ndarray) -> np.ndarray:
  """``cells`` at each of FRAMES ``(3, ...)``: scaled by SCALE."""
  return SCALE.reshape((-1, ) + (1, ) * cells.ndim) * cells


def values(cells, frames=FRAMES, time=TIMES) -> CellValues:
  """``{block: (n_f, nr, nt)}`` at the frames, as cell values."""
  return CellValues("synthetic", ".", ("q", ), None, tuple(cells), frames, time,
                    tuple(cells.values()))


def make_layout(nodes, *, stacks=None, targets=(), reversed_=None):
  """A layout record holding just what the profiles read."""
  nb = len(nodes)
  return MultibLayout(name="synthetic",
                      path=".",
                      single_block=False,
                      nodes=tuple(nodes),
                      reversed=tuple(reversed_ or (False, ) * nb),
                      connections=np.zeros((nb, 2, 2, 3), dtype=int),
                      region=("sol", ) * nb,
                      under=(None, ) * nb,
                      lcfs_row=(None, ) * nb,
                      chains=(),
                      targets=tuple(targets),
                      stacks=dict(stacks or {}),
                      half_domain=False)


class TestMidplaneValues:

  def test_linear_in_z_field_is_exact_at_z0(self):
    nodes = block_nodes(3)
    got = sp.midplane_values(nodes, linear_cells(nodes))
    np.testing.assert_allclose(got, A0 + A1 * np.arange(3), rtol=0, atol=1e-13)

  def test_leading_frame_axis_is_kept(self):
    nodes = block_nodes(3)
    got = sp.midplane_values(nodes, over_frames(linear_cells(nodes)))
    np.testing.assert_allclose(got,
                               SCALE[:, None] * (A0 + A1 * np.arange(3)),
                               rtol=0,
                               atol=1e-13)

  def test_reflecting_midplane_takes_the_cell_next_to_it(self):
    # Z = 0 is the block's theta-lower edge: no cell centre lies below it.
    Zj = np.array([0.0, 0.3, 0.7, 1.2])
    R = R0 + DR * np.arange(4)[:, None] + 0.0 * Zj
    nodes = np.stack([R, np.broadcast_to(Zj, R.shape)], axis=-1)
    cells = np.arange(9.0).reshape(3, 3)**2
    np.testing.assert_array_equal(sp.midplane_values(nodes, cells), cells[:, 0])
    np.testing.assert_allclose(sp.midplane_cell_radii(nodes),
                               R0 + DR * (np.arange(3) + 0.5),
                               rtol=0,
                               atol=1e-15)


class TestMidplaneCellRadii:

  def test_cell_row_radius_is_mean_of_its_node_rows(self):
    got = sp.midplane_cell_radii(block_nodes(4))
    np.testing.assert_allclose(got,
                               R0 + DR * (np.arange(4) + 0.5),
                               rtol=0,
                               atol=1e-14)


class TestStackProfile:

  @pytest.mark.parametrize("side,sign", [("outboard", 1.0), ("inboard", -1.0)])
  def test_core_then_sol_inside_out(self, side, sign):
    core, sol = block_nodes(3, 0, sign), block_nodes(4, 3, sign)
    stack = MidplaneStack(side, core=0, sol=1, mirror=False)
    layout = make_layout([core, sol], stacks={side: stack})
    cells = {
        0: over_frames(linear_cells(core, 0)),
        1: over_frames(linear_cells(sol, 3))
    }
    p = sp.stack_profile(layout, side, values(cells))

    rows = np.arange(7)
    np.testing.assert_array_equal(p.block, [0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(p.row, [0, 1, 2, 0, 1, 2, 3])
    np.testing.assert_allclose(p.R,
                               R0 + sign * DR * (rows + 0.5),
                               rtol=0,
                               atol=1e-14)
    assert p.R_sep == pytest.approx(R0 + sign * DR * 3, abs=1e-14)
    # Positive into the SOL on either side: half a cell row past the
    # separatrix for the first SOL row.
    np.testing.assert_allclose(p.distance,
                               DR * (rows + 0.5 - 3),
                               rtol=0,
                               atol=1e-14)
    np.testing.assert_array_equal(p.frames, FRAMES)
    np.testing.assert_array_equal(p.time, TIMES)
    np.testing.assert_allclose(p.value,
                               SCALE[:, None] * (A0 + A1 * rows),
                               rtol=0,
                               atol=1e-13)

  def test_reversed_block_reports_file_rows(self):
    core, sol = block_nodes(2), block_nodes(3, 2)
    stack = MidplaneStack("outboard", core=0, sol=1, mirror=False)
    layout = make_layout([core, sol],
                         stacks={"outboard": stack},
                         reversed_=(False, True))
    cells = {0: linear_cells(core, 0)[None], 1: linear_cells(sol, 2)[None]}
    p = sp.stack_profile(layout, "outboard", values(cells, [3], [1e-6]))
    np.testing.assert_array_equal(p.row, [0, 1, 2, 1, 0])

  def test_sol_only_stack_and_its_separatrix_row(self):
    # A limited run's block: its LCFS (separatrix) at node row 2.
    sol = block_nodes(5)
    stack = MidplaneStack("outboard", core=None, sol=0, mirror=False, sep_row=2)
    p = sp.stack_profile(make_layout([sol], stacks={"outboard": stack}),
                         "outboard",
                         values({0: linear_cells(sol)[None]}, [0], [0.0]))
    assert p.R_sep == pytest.approx(R0 + 2 * DR, abs=1e-14)
    np.testing.assert_allclose(p.distance,
                               DR * (np.arange(5) + 0.5 - 2),
                               rtol=0,
                               atol=1e-14)

  def test_missing_side_is_refused(self):
    sol = block_nodes(2)
    layout = make_layout(
        [sol], stacks={"outboard": MidplaneStack("outboard", None, 0, False)})
    with pytest.raises(ValueError, match="no inboard midplane"):
      sp.stack_profile(layout, "inboard",
                       values({0: linear_cells(sol)[None]}, [0], [0.0]))


class TestHeatFluxWidth:

  def test_piecewise_constant_profile(self):
    # Integrals 3, 2, 0.5 (total 5.5); 0.63 * 5.5 = 3.465 = 3 + 1 * (x - 1).
    edges, q = np.array([0.0, 1.0, 3.0, 4.0]), np.array([3.0, 1.0, 0.5])
    assert sp.heat_flux_width(edges, q, 0.63) == pytest.approx(1.465, abs=1e-14)
    assert sp.heat_flux_width(edges, q, 1.0) == pytest.approx(4.0, abs=1e-14)

  def test_no_flux_has_no_width(self):
    assert np.isnan(sp.heat_flux_width(np.arange(4.0), np.zeros(3), 0.63))

  @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
  def test_fraction_outside_unit_interval_is_refused(self, fraction):
    with pytest.raises(ValueError, match="fraction"):
      sp.heat_flux_width(np.arange(3.0), np.ones(2), fraction)


class TestPlateDistance:

  def test_middle_of_each_segment_along_the_face(self):
    face = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 10.0]])  # lengths 5, 6
    np.testing.assert_allclose(sp.plate_distance(face), [2.5, 8.0],
                               rtol=0,
                               atol=1e-15)


def plate_leg(nr: int = 4) -> np.ndarray:
  """A leg block whose theta-upper face (the plate) runs along a polyline
  of segment lengths 5, 6, 5, 6 from the separatrix corner."""
  nodes = block_nodes(nr)
  nodes[:, -1] = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 10.0], [6.0, 14.0],
                           [6.0, 20.0]])[:nr + 1]
  return nodes


class TestPlateProfile:

  def test_midplane_distance_and_lambda_q(self):
    leg, mid = plate_leg(), block_nodes(4)
    # Signed flux at the plate (theta end -1) at two frames; the profile is
    # |q|. The other columns hold values the plate must not read.
    flux = np.full((2, 4, 4), 7.0)
    flux[:, :, -1] = [[-3.0, 1.0, -0.5, 0.0], [0.5, -1.0, 3.0, 0.5]]
    target = Target("upper outer target",
                    leg=0,
                    pf=None,
                    theta_end=-1,
                    mid_block=1,
                    mid_crossing=None,
                    updown="upper",
                    side="outer")
    p = sp.plate_profile(make_layout([leg, mid], targets=[target]), target,
                         values({0: flux}, [2, 5], [1e-6, 2e-6]), 0.63)
    np.testing.assert_array_equal(p.frames, [2, 5])
    np.testing.assert_array_equal(p.time, [1e-6, 2e-6])
    np.testing.assert_array_equal(p.heat_flux,
                                  [[3.0, 1.0, 0.5, 0.0], [0.5, 1.0, 3.0, 0.5]])
    np.testing.assert_array_equal(p.row, [0, 1, 2, 3])
    np.testing.assert_allclose(p.distance,
                               DR * (np.arange(4) + 0.5),
                               rtol=0,
                               atol=1e-14)
    np.testing.assert_allclose(p.plate_distance, [2.5, 8.0, 13.5, 19.0],
                               rtol=0,
                               atol=1e-14)
    # Integrals over rows of width DR. Frame 2: 3, 1, 0.5 (x DR), total
    # 4.5 DR; 0.63 * 4.5 = 2.835 = 3 * x / DR within the first row. Frame 5:
    # 0.5, 1, 3, 0.5, total 5 DR; 0.63 * 5 = 3.15 = 1.5 + 3 (x / DR - 2).
    np.testing.assert_allclose(p.lambda_q,
                               DR * np.array([2.835 / 3.0, 2.55]),
                               rtol=1e-14,
                               atol=0)

  def test_open_row_starts_at_the_lcfs(self):
    leg = plate_leg()
    flux = np.zeros((1, 4, 4))
    flux[0, :, 0] = [9.0, 2.0, 1.0, 1.0]
    target = Target("lower outer target",
                    leg=0,
                    pf=None,
                    theta_end=0,
                    mid_block=0,
                    mid_crossing=None,
                    updown="lower",
                    side="outer",
                    open_row=1)
    p = sp.plate_profile(make_layout([leg], targets=[target]), target,
                         values({0: flux}, [0], [0.0]))
    np.testing.assert_array_equal(p.heat_flux, [[2.0, 1.0, 1.0]])
    np.testing.assert_array_equal(p.row, [1, 2, 3])
    np.testing.assert_allclose(p.distance,
                               DR * (np.arange(3) + 0.5),
                               rtol=0,
                               atol=1e-14)
    # Integrals 2, 1, 1 (x DR), total 4 DR; 0.63 * 4 = 2.52 = 2 + (x/DR - 1).
    np.testing.assert_allclose(p.lambda_q, [DR * 1.52], rtol=0, atol=1e-15)

  def test_without_a_midplane_only_the_plate_places_surfaces(self):
    leg = plate_leg()
    flux = np.ones((2, 4, 4))
    target = Target("upper inner target",
                    leg=0,
                    pf=None,
                    theta_end=-1,
                    mid_block=None,
                    mid_crossing=None,
                    updown="upper",
                    side="inner")
    p = sp.plate_profile(make_layout([leg], targets=[target]), target,
                         values({0: flux}, [0, 1], [0.0, 1.0]))
    assert p.distance is None
    assert p.lambda_q.shape == (2, ) and np.isnan(p.lambda_q).all()
    np.testing.assert_allclose(p.plate_distance, [2.5, 8.0, 13.5, 19.0],
                               rtol=0,
                               atol=1e-14)
    assert "no block of its chain reaches Z = 0" in str(p)


class TestRecords:

  @staticmethod
  def profile(**kw):
    """A two-row profile at two frames, ``kw`` replacing fields."""
    fields = dict(stack=MidplaneStack("outboard", None, 0, False),
                  block=np.array([0, 0]),
                  row=np.array([0, 1]),
                  R=np.array([1.0, 2.0]),
                  R_sep=1.5,
                  frames=np.array([0, 3]),
                  time=np.array([0.0, 1e-6]),
                  value=np.array([[3.0, 4.0], [5.0, 6.0]]))
    return sp.MidplaneProfile(**{**fields, **kw})

  def test_profile_arrays_are_read_only_copies(self):
    R, value = np.array([1.0, 2.0]), np.array([[3.0, 4.0], [5.0, 6.0]])
    p = self.profile(R=R, value=value)
    R[0], value[0, 0] = 9.0, 9.0
    assert p.R[0] == 1.0 and p.value[0, 0] == 3.0
    for field in (p.value, p.frames, p.time):
      with pytest.raises(ValueError):
        field[0] = 0
    np.testing.assert_array_equal(p.distance, [-0.5, 0.5])
    assert "2 frames (0 to 3)" in str(p)

  def test_profile_refuses_unequal_lengths(self):
    with pytest.raises(ValueError, match="same length"):
      self.profile(block=np.array([0]))
    with pytest.raises(ValueError, match="same length"):
      self.profile(time=np.array([0.0]))

  @pytest.mark.parametrize("value", [
      np.array([3.0, 4.0]),
      np.ones((3, 2)),
      np.ones((2, 3)),
      np.ones((2, 2, 1))
  ])
  def test_profile_value_must_be_frames_by_rows(self, value):
    with pytest.raises(ValueError, match=r"\(frames, row\) = \(2, 2\)"):
      self.profile(value=value)

  def test_profile_needs_a_frame(self):
    with pytest.raises(ValueError, match="at least one frame"):
      self.profile(frames=np.array([], int),
                   time=np.array([]),
                   value=np.zeros((0, 2)))

  def test_target_without_midplane_refuses_a_width(self):
    target = Target("t", 0, None, 0, None, None, "lower", "outer")
    with pytest.raises(ValueError, match="midplane distance"):
      sp.TargetProfile(target,
                       row=np.array([0]),
                       plate_distance=np.array([0.1]),
                       distance=None,
                       frames=np.array([0, 1]),
                       time=np.array([0.0, 1.0]),
                       heat_flux=np.array([[1.0], [2.0]]),
                       lambda_q=np.array([np.nan, 0.01]),
                       fraction=0.63)

  def test_target_lambda_q_has_one_value_per_frame(self):
    target = Target("t", 0, None, 0, 1, None, "lower", "outer")
    with pytest.raises(ValueError, match="same length"):
      sp.TargetProfile(target,
                       row=np.array([0]),
                       plate_distance=np.array([0.1]),
                       distance=np.array([0.05]),
                       frames=np.array([0, 1]),
                       time=np.array([0.0, 1.0]),
                       heat_flux=np.array([[1.0], [2.0]]),
                       lambda_q=np.array([0.01]),
                       fraction=0.63)


class TestPublicArguments:

  def test_bare_string_quantity_is_refused(self):
    with pytest.raises(TypeError, match=r"\['M0'\]"):
      sp.midplane_profile("M0", "run", 0)

  def test_bare_string_species_is_refused(self):
    with pytest.raises(TypeError, match=r"\['elc'\]"):
      sp.target_profiles("run", 0, species="elc")


class TestNonMonotoneMidplane:

  def test_decreasing_edges_are_refused(self):
    with pytest.raises(ValueError, match="must not decrease"):
      sp.heat_flux_width(np.array([0.0, 2.0, 1.0]), np.ones(2), 0.63)

  def test_plate_profile_has_no_width_on_folded_rows(self):
    # Node row 3 of the midplane block crosses Z = 0 inside row 2: the gaps
    # from the separatrix are 0, 1, 2, 0.5, 4 (x DR), not increasing.
    leg, mid = plate_leg(), block_nodes(4)
    mid[3, :, 0] -= 2.5 * DR
    target = Target("lower outer target",
                    leg=0,
                    pf=None,
                    theta_end=0,
                    mid_block=1,
                    mid_crossing=None,
                    updown="lower",
                    side="outer")
    p = sp.plate_profile(make_layout([leg, mid], targets=[target]), target,
                         values({0: np.ones((2, 4, 4))}, [0, 1], [0.0, 1.0]))
    np.testing.assert_allclose(p.distance,
                               DR * np.array([0.5, 1.5, 1.25, 2.25]),
                               rtol=0,
                               atol=1e-14)
    assert np.isnan(p.lambda_q).all()
    assert "midplane distances not increasing" in str(p)

  def test_empty_quantity_list_is_refused(self):
    with pytest.raises(ValueError, match="at least one name"):
      sp.midplane_profile([], "run", 0)


@pytest.fixture
def run(tmp_path):
  return str(fx.write_plate_files(fx.write_run(tmp_path / "run")))


def cell_centres(block: int) -> np.ndarray:
  xe, _ = fx.cell_edges(block)
  return 0.5 * (xe[1:] + xe[:-1])


class TestFromRunFiles:
  """The entry points on the fixture run: every frame's files read once per
  frame, the layout once per call."""

  @pytest.mark.parametrize("side,sign", [("outboard", 1.0), ("inboard", -1.0)])
  def test_midplane_density_at_every_frame(self, run, side, sign):
    p = sp.midplane_profile(["M0"],
                            fx.NAME,
                            ":",
                            species=["elc", "ion"],
                            path=run,
                            side=side)
    assert p.stack.blocks == (0, 1)
    np.testing.assert_array_equal(p.frames, [0, 1, 2])
    np.testing.assert_array_equal(p.time, fx.FRAME_TIMES)
    np.testing.assert_array_equal(p.block, [0, 0, 1, 1])
    np.testing.assert_array_equal(p.row, [0, 1, 0, 1])
    # Node rows at r = 0.5, 0.75, 1 (core) and 1, 1.25, 1.5 (SOL), on Z = 0.
    r = np.array([0.625, 0.875, 1.125, 1.375])
    np.testing.assert_allclose(p.R, fx.R0 + sign * r, rtol=0, atol=1e-14)
    assert p.R_sep == pytest.approx(fx.R0 + sign * 1.0, abs=1e-14)
    np.testing.assert_allclose(p.distance, r - 1.0, rtol=0, atol=1e-14)
    # Both species' density a + b x (no y dependence), at the cell centres.
    for k in range(3):
      want = np.concatenate([
          2 * (fx.density(b, k)[0] + fx.density(b, k)[1] * cell_centres(b))
          for b in (0, 1)
      ])
      np.testing.assert_allclose(p.value[k], want, rtol=1e-14, atol=0)

  @pytest.mark.parametrize("frame,want", [("0,2", [0, 2]), ("1:3", [1, 2]),
                                          ([2], [2]), (1, [1])])
  def test_frame_specification(self, run, frame, want):
    p = sp.midplane_profile(["M0"], fx.NAME, frame, species=["elc"], path=run)
    np.testing.assert_array_equal(p.frames, want)
    np.testing.assert_array_equal(p.time, np.array(fx.FRAME_TIMES)[want])
    for i, k in enumerate(want):
      np.testing.assert_allclose(p.value[i, :2],
                                 fx.density(0, k)[0] +
                                 fx.density(0, k)[1] * cell_centres(0),
                                 rtol=1e-14,
                                 atol=0)

  def test_a_species_missing_a_frame_is_refused(self, run):
    (Path(run) / f"{fx.NAME}_b1-ion_M0_2.gkyl").unlink()
    with pytest.raises(ValueError, match="every species at the same frames"):
      sp.midplane_profile(["M0"], fx.NAME, species=["elc", "ion"], path=run)

  def test_blocks_holding_different_frames_are_refused(self, run):
    for s in ("elc", "ion"):
      (Path(run) / f"{fx.NAME}_b1-{s}_M0_2.gkyl").unlink()
    with pytest.raises(ValueError, match="hold different frames"):
      sp.midplane_profile(["M0"], fx.NAME, species=["elc", "ion"], path=run)

  def test_target_heat_flux_and_width_at_every_frame(self, run):
    (p, ) = sp.target_profiles(fx.NAME, ":", species=["elc", "ion"], path=run)
    assert (p.target.name, p.target.leg) == ("lower inner target", fx.LEG)
    np.testing.assert_array_equal(p.frames, [0, 1, 2])
    np.testing.assert_array_equal(p.time, fx.FRAME_TIMES)
    np.testing.assert_array_equal(p.row, [0, 1])
    # Node rows r = 1, 1.25, 1.5: 0.25 apart on Z = 0 and along the plate.
    np.testing.assert_allclose(p.distance, [0.125, 0.375], rtol=0, atol=1e-14)
    np.testing.assert_allclose(p.plate_distance, [0.125, 0.375],
                               rtol=0,
                               atol=1e-14)
    # |h0 + h1 x| at x = 1.25, 1.75 (the plate column's y term vanishes).
    want = [[3.5e6, 2.5e6], [2.0e6, 4.0e6], [3.0e6, 3.0e6]]
    np.testing.assert_allclose(p.heat_flux, want, rtol=1e-14, atol=0)
    # Integrals 0.25 |q| per row: 0.63 * 0.25 * 6e6 = 0.945e6 lies in the
    # second row at frames 0 and 1 (0.25 + (0.945 - 0.875) / 2.5 and
    # 0.25 + (0.945 - 0.5) / 4), and at 0.63 * 0.5 for the flat frame 2.
    np.testing.assert_allclose(p.lambda_q, [0.278, 0.36125, 0.315],
                               rtol=1e-13,
                               atol=0)
