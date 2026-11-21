"""Tests for ``postgkyl.numerics.frame_match.match_frames``: nearest-time
matching within half of the coarser frame spacing."""

from __future__ import annotations

import numpy as np

from postgkyl.numerics.frame_match import match_frames

DT = 0.25
T = np.arange(12) * DT


def test_identical_times_match_themselves():
  np.testing.assert_array_equal(match_frames(T, T), np.arange(T.size))


def test_frames_past_the_end_have_no_match():
  got = match_frames(T, T[:5])
  np.testing.assert_array_equal(got, np.r_[np.arange(5), [-1] * 7])


def test_shifted_times_match_the_nearest_within_half_a_spacing():
  np.testing.assert_array_equal(match_frames(T + 0.3 * DT, T),
                                np.arange(T.size))
  # 0.6 of a spacing later: the next frame is 0.4 away; the last has none.
  np.testing.assert_array_equal(match_frames(T + 0.6 * DT, T),
                                np.r_[np.arange(1, T.size), -1])


def test_a_tie_goes_to_the_earlier_frame():
  np.testing.assert_array_equal(match_frames([0.5 * DT], T[:2]), [0])


def test_unsorted_frames_return_their_own_indices():
  order = np.array([3, 0, 2, 1])
  got = match_frames(T[:4], T[:4][order])
  np.testing.assert_array_equal(T[:4][order][got], T[:4])


def test_coarser_run_sets_the_tolerance():
  coarse = np.arange(4) * 1.0  # median spacing 1: tolerance 0.5
  fine = np.array([0.0, 0.25, 0.5, 1.45, 1.7])  # median spacing 0.25
  np.testing.assert_array_equal(match_frames(coarse, fine), [0, 3, 4, -1])


def test_empty_inputs():
  assert match_frames([], T).size == 0
  np.testing.assert_array_equal(match_frames(T[:3], []), [-1, -1, -1])
