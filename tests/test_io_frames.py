"""The frame specification shared by ``load --frame``, the GK loaders and the
GUI (``postgkyl.io.select_frames``)."""

from __future__ import annotations

import pytest

from postgkyl.io import select_frame_files, select_frames

AVAILABLE = [0, 2, 4, 6]


@pytest.mark.parametrize(
    "frame, expected",
    [
        (":", [0, 2, 4, 6]),
        ("2:5", [2, 4]),
        ("4:", [4, 6]),
        (":4", [0, 2]),
        ("::4", [0, 4]),
        ("2::4", [2, 6]),
        # A negative number or bound counts back from the last frame.
        ("-1", [6]),
        (-1, [6]),
        ("-2:", [4, 6]),
        (":-1", [0, 2, 4]),
        ("-10:", [0, 2, 4, 6]),  # clamped, like a Python slice
        ("-5:-4", []),
        # Frames given outright are returned, available or not.
        ("7", [7]),
        (7, [7]),
        ("6,0", [6, 0]),
        ([2, -1], [2, 6]),
    ])
def test_select_frames(frame, expected):
  assert select_frames(frame, reversed(AVAILABLE)) == expected


@pytest.mark.parametrize("frame, message", [
    ("a:b", "not a frame number"),
    ("1:2:3:4", "not a frame number"),
    ("::0", "positive integer"),
    ("-5", "counts back past the first"),
])
def test_select_frames_rejects(frame, message):
  with pytest.raises(ValueError, match=message):
    select_frames(frame, AVAILABLE)


def test_a_range_of_no_available_frames_is_empty():
  assert select_frames(":", []) == []


def test_frame_files_skip_restarts_and_keep_every_block():
  paths = [
      "d/s-elc_M0_3.gkyl", "d/s-elc_M0_1.gkyl", "d/s-elc_M0_3_restart.gkyl",
      "d/s-elc_M0.gkyl", "d/s_b1-elc_M0_3.gkyl", "d/s-elc_M0_2.gkyl"
  ]
  last = select_frame_files(paths, "-1")
  assert last == ["d/s-elc_M0_3.gkyl", "d/s_b1-elc_M0_3.gkyl"]
  first = select_frame_files(paths, ":3")
  assert first == ["d/s-elc_M0_1.gkyl", "d/s-elc_M0_2.gkyl"]
