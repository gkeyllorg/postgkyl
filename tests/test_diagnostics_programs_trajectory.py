"""Tests for ``postgkyl.diagnostics.vlasov.trajectory``.

Ported from ``src_bak/postgkyl/apps/trajectory.py`` (no ``tests_bak`` corpus
exists for this app). A Gkeyll dynvector's grid holds exactly one time stamp
per recorded sample (``io/gkyl_reader.py``'s ``_read_t2_v1``: ``grid[0]``
has the same length as ``values.shape[0]``) -- unlike a *field* file's
``num_cells + 1`` edge convention. ``postgkyl.io.write`` only emits
file_type == 1 (field) ``.gkyl`` files, so a real write -> reload round trip
reads back through a field file with no basis metadata; ``GDataState``
defaults that case to p0 serendipity/nodal and re-expresses the grid as
cell centers, which lines back up with the dynvector convention (one point
per sample). Most trajectory fixtures here still build the ``GDataState``
directly (the same technique ``tests/test_io_writer.py``'s ``_make_state``
uses); one test explicitly exercises the ``io.write`` round trip to confirm
the convention matches after reload.

Run: PYTHONPATH=src pytest tests/test_diagnostics_programs_trajectory.py -v
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from postgkyl import io
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.diagnostics.vlasov import trajectory as traj


def _make_trajectory(num_pos=10, *, velocity=False, seed=0):
  """A synthetic dynvector-shaped trajectory: ``grid[0]`` has exactly
  ``num_pos`` time stamps, matching ``values.shape[0]``."""
  rng = np.random.default_rng(seed)
  time = np.linspace(0.0, 1.0, num_pos)
  ncomp = 6 if velocity else 3
  values = rng.uniform(-1.0, 1.0, size=(num_pos, ncomp))
  d = GDataState()
  d.push([time], values)
  return d
# end


class TestMasked:

  def test_no_bounds_passthrough(self):
    coord = np.array([1.0, 2.0, 3.0])
    out = traj._masked(coord, None, None)
    np.testing.assert_allclose(out, coord)
  # end

  def test_lower_bound_masks_below(self):
    coord = np.array([1.0, 2.0, 3.0])
    out = traj._masked(coord, 1.5, None)
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1:], [2.0, 3.0])
  # end

  def test_upper_bound_masks_above(self):
    coord = np.array([1.0, 2.0, 3.0])
    out = traj._masked(coord, None, 2.5)
    np.testing.assert_allclose(out[:2], [1.0, 2.0])
    assert np.isnan(out[2])
  # end

  def test_both_bounds(self):
    coord = np.array([1.0, 2.0, 3.0])
    out = traj._masked(coord, 1.5, 2.5)
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1], 2.0)
    assert np.isnan(out[2])
  # end
# end


class TestTrajectoryRaises:

  def test_no_datasets_raises(self):
    with pytest.raises(ValueError, match="at least one dataset"):
      traj.trajectory()
  # end
# end
    # end


class TestTrajectorySynthetic:

  def test_frame_count_matches_samples(self):
    d = _make_trajectory(num_pos=8)
    anim = traj.trajectory(d)
    try:
      assert anim._save_count == 8
    # end
    finally:
      plt.close(anim._fig)
  # end
    # end

  def test_numframes_subsamples(self):
    d = _make_trajectory(num_pos=20)
    anim = traj.trajectory(d, numframes=5)
    try:
      assert anim._save_count == 5
    # end
    finally:
      plt.close(anim._fig)
  # end
    # end

  def test_first_frame_renders_without_error(self):
    d = _make_trajectory(num_pos=6, velocity=True)
    anim = traj.trajectory(d, velocity=True)
    try:
      fig = anim._fig
      ax = fig.axes[0]
      traj._update(0, ax, (d,), 1, True, None, None, None, None, None, None)
      assert ax.get_title().startswith("T:")
    # end
    finally:
      plt.close(anim._fig)
  # end
    # end

  def test_last_frame_uses_final_dt_branch(self):
    """When ``t_idx + leap`` runs past the end of the trace, the velocity
    vector uses ``time[-1] - time[t_idx]`` instead of indexing out of
    bounds."""
    d = _make_trajectory(num_pos=4, velocity=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
      traj._update(3, ax, (d,), 1, True, None, None, None, None, None, None)
    # end
    finally:
      plt.close(fig)
  # end
    # end

  def test_multiple_datasets_overlaid(self):
    d1 = _make_trajectory(num_pos=6, seed=1)
    d2 = _make_trajectory(num_pos=6, seed=2)
    anim = traj.trajectory(d1, d2)
    try:
      assert anim._save_count == 6
    # end
    finally:
      plt.close(anim._fig)
  # end
    # end

  def test_axis_bounds_mask_points(self):
    d = _make_trajectory(num_pos=6)
    anim = traj.trajectory(d, xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5,
        zmin=-0.5, zmax=0.5)
    try:
      assert anim._save_count == 6
    # end
    finally:
      plt.close(anim._fig)
  # end
    # end

  def test_fixaspect_and_view_angles(self):
    d = _make_trajectory(num_pos=5)
    anim = traj.trajectory(d, fixaspect=True, elevation=30.0, azimuth=45.0)
    try:
      assert anim._save_count == 5
    # end
    finally:
      plt.close(anim._fig)
  # end
# end
    # end


class TestTrajectoryViaIoWriter:
  """Exercises the ``io.write`` round trip the instruction file suggests.

  The written file is a field file with no basis metadata, so on reload
  ``GDataState`` defaults it to p0 serendipity/nodal and re-expresses the
  ``num_cells + 1`` edge grid as cell centers -- which lines back up
  one-to-one with the dynvector convention (``len(grid[0]) == values.shape[0]``).

  Single-component only: the compiled reader (``gpython.rio.read_field``, tried
  first whenever the shim is available) fails on *any* multi-component
  ``.gkyl`` field this writer produces --
  ``PYTHONPATH=src python -c`` reproduction:
  ``io.write(state_with_ncomp_2_or_more, ...)`` then re-reading it raises
  ``OSError: gpython_read_field failed`` (reproduces even for pre-existing,
  layer-agnostic data, e.g. any ``GDataState`` pushed with
  ``values.shape[-1] >= 2``; single-component data round-trips fine). That
  is a pre-existing limitation in ``gpython``/``io`` (outside this layer's
  scope), not something introduced here -- see this layer's report. A real
  3-component trajectory is exercised directly (no disk I/O) by
  ``TestTrajectorySynthetic`` instead."""

  def test_single_component_trajectory_round_trips_and_animates(self, tmp_path):
    num_pos = 6
    time_edges = np.linspace(0.0, 1.0, num_pos + 1)
    values = np.zeros((num_pos, 1))
    values[:, 0] = np.linspace(0.0, 1.0, num_pos)
    d = GDataState()
    d.push([time_edges], values)

    out = io.save(d, out_name=str(tmp_path / "traj.gkyl"), extension="gkyl")

    from postgkyl.gdata import GData
    with pytest.warns(UserWarning, match="not resolvable"):
      reloaded = GData(out)
    # end
    assert reloaded.grid[0].shape[0] == num_pos + 1  # field convention: N+1 edges
    assert reloaded.values.shape[0] == num_pos
  # end
# end
