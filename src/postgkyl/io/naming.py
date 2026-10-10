"""Gkeyll's output-file naming convention -- the ONE home for reading a
dataset's *identity* out of its path.

A Gkeyll output file name encodes four facts::

    rt_gk_multib_sheath_1x2v_p1_b2-geo_int_B3.gkyl
    `------------ sim ---------' `bl' `-quantity-'

    gk_lorentzian_mirror-elc_M0_1.gkyl
    `------- sim -------' `-quan-' frame

- **sim**      the simulation name (everything before the last ``'-'``),
- **block**    the multiblock block index, the ``_b<N>`` suffix of the sim
               part; ``None`` for a single-block run,
- **quantity** the output name (species/moment/geometry field),
- **frame**    the trailing ``_<digits>`` of the quantity, when present.

This convention is shared by geometry lookup, output discovery, and frame
grouping. It lives in ``io`` because it describes Gkeyll's files and sits below
every consumer: ``gdatastate`` stamps identity into ``ctx`` at load time and
``diagnostics`` builds quantity discovery on top of it.

Frame numbers are chosen with one specification, :func:`select_frames`,
shared by ``load``'s ``frame`` option, the gyrokinetic loaders and the GUI.

The parser is *pure*: it never touches the filesystem. ``os.path.exists`` is
the caller's business.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import numbers
import os
import re
from dataclasses import dataclass

# The multiblock block index: Gkeyll writes "<sim>_b<N>-<quantity>.gkyl"
# (see the ``name + '_b*-'`` prefix built by ``diagnostics.gk.nodes``
# and main's ``nodes``). Digits are required, so a simulation legitimately
# named e.g. "gk_beta_scan" is never mistaken for block "eta_scan".
_BLOCK_RE = re.compile(r"^(?P<sim>.*)_b(?P<block>\d+)$")

# The frame index: a trailing "_<digits>" run. Requiring *only* digits is what
# keeps a geometry field like "geo_int_B3" (frame-less) from being read as
# quantity "geo_int_B" at frame 3.
_FRAME_RE = re.compile(r"^(?P<quantity>.*)_(?P<frame>\d+)$")

_RESTART_SUFFIX = "_restart"


@dataclass(frozen=True)
class OutputName:
  """The identity of one Gkeyll output file, parsed from its path.

  Attributes:
    directory: The path's directory part ("" for a bare file name).
    sim: The simulation name, with any ``_b<N>`` block suffix removed.
    block: The multiblock block index, or ``None`` for single-block output.
    quantity: The output name, with any frame index and ``_restart`` removed.
    frame: The frame index, or ``None`` when the name carries none.
    restart: True when the name carried a ``_restart`` suffix.
  """

  directory: str
  sim: str
  block: int | None
  quantity: str
  frame: int | None
  restart: bool = False

  @property
  def prefix(self) -> str:
    """The ``'<dir>/<sim>[_b<N>]'`` path every sibling file of this *block*
    shares -- what a geometry lookup appends ``'-geo_int_nodes.gkyl'`` to.

    For single-block output this is the path before the last ``'-'``.
    """
    base = self.sim if self.block is None else f"{self.sim}_b{self.block:d}"
    return os.path.join(self.directory, base) if self.directory else base

  @property
  def stem(self) -> str:
    """``'<sim>[_b<N>]-<quantity>'`` -- the file name with directory, frame
    index, ``_restart`` and extension stripped (``discovery``'s notion of a
    stem)."""
    tail = f"-{self.quantity}" if self.quantity else ""
    return f"{os.path.basename(self.prefix)}{tail}"

  @property
  def field_key(self) -> tuple:
    """What two files of the **same field on different blocks** share.

    Deliberately excludes ``block`` (and the directory): it is the key
    :func:`postgkyl.gdatastate.collection.group_blocks` partitions a working
    set on.
    """
    return (self.sim, self.quantity, self.frame)


def parse_output_name(path: str | None) -> OutputName | None:
  """Parse a Gkeyll output path into its :class:`OutputName` identity.

  Args:
    path: A file path, e.g. ``"data/sim_b2-elc_M0_7.gkyl"``. May be any
      extension, or none.

  Returns:
    The parsed identity, or ``None`` for an empty/absent path (a dataset a
    verb computed rather than read from disk).

  Notes:
    The split is deliberately total -- every non-empty path parses. A name
    with no ``'-'`` at all (out of convention) yields the whole stem as
    ``sim`` and an empty ``quantity``; the frame index is then taken off the
    ``sim``, since that is the only component there is.
  """
  if not path:
    return None
  directory, base = os.path.split(str(path))
  stem = os.path.splitext(base)[0]

  restart = stem.endswith(_RESTART_SUFFIX)
  if restart:
    stem = stem[:-len(_RESTART_SUFFIX)]

  if "-" in stem:
    sim_part, quantity = stem.rsplit("-", 1)
  else:
    sim_part, quantity = stem, ""

  # The frame index trails the *last* component of the name -- the quantity
  # normally, the sim itself for a dash-less name.
  tail = quantity or sim_part
  frame = None
  match = _FRAME_RE.match(tail)
  if match:
    tail = match.group("quantity")
    frame = int(match.group("frame"))
  if quantity:
    quantity = tail
  else:
    sim_part = tail

  block = None
  match = _BLOCK_RE.match(sim_part)
  if match:
    sim_part = match.group("sim")
    block = int(match.group("block"))

  return OutputName(directory=directory,
                    sim=sim_part,
                    block=block,
                    quantity=quantity,
                    frame=frame,
                    restart=restart)


# A frame specification: one frame, frames, or a range of frames.
FrameSpec = int | str | Sequence[int]


def _position(frame: int, available: list[int]) -> int:
  """``frame``, or for a negative one the frame that far from the last."""
  if frame >= 0:
    return frame
  if -frame > len(available):
    raise ValueError(f"frame {frame} counts back past the first of the "
                     f"{len(available)} available frame(s)")
  return available[frame]


def _bound(value: int | None, available: list[int], default: int) -> int:
  """A range bound as a frame number; a negative one counts from the end,
  clamped to the first available frame like a Python slice."""
  if value is None:
    return default
  if value >= 0:
    return value
  return available[max(len(available) + value, 0)]


def select_frames(frame: FrameSpec, available: Iterable[int]) -> list[int]:
  """The frame numbers ``frame`` selects among the ``available`` frames.

  Args:
    frame: A frame number (``7``, ``"7"``), frame numbers (``[0, 2]``,
      ``"0,2"``), or a ``"start:stop[:step]"`` range: the available frames
      ``start <= frame < stop`` whose number is ``start`` plus a multiple of
      ``step``. An empty ``start``/``stop`` means the first/past the last
      available frame, so ``":"`` selects them all. A negative number or
      bound counts back from the last available frame, like a Python index:
      ``-1`` is the last frame and ``"-10:"`` the last ten.
    available: The frame numbers on disk, in any order.

  Returns:
    The selected frame numbers, in increasing order for a range. Explicit
    non-negative frames are returned whether or not they are available.

  Raises:
    ValueError: ``frame`` is malformed, its step is not positive, or a
      negative frame counts back past the first available one.
  """
  available = sorted(available)
  if isinstance(frame, numbers.Integral) and not isinstance(frame, bool):
    return [_position(int(frame), available)]
  if not isinstance(frame, str):
    return [_position(int(f), available) for f in frame]
  text = frame.strip()
  try:
    if ":" not in text:
      listed = [int(f) for f in text.split(",")]
    else:
      parts = text.split(":")
      if len(parts) > 3:
        raise ValueError
      start, stop, step = [int(p) if p.strip() else None
                           for p in parts] + [None] * (3 - len(parts))
  except ValueError:
    raise ValueError(f"frame {frame!r} is not a frame number, a "
                     "comma-separated list, or a 'start:stop[:step]' "
                     "range") from None
  if ":" not in text:
    return [_position(f, available) for f in listed]
  step = 1 if step is None else step
  if step <= 0:
    raise ValueError(f"frame {frame!r}: the step must be a positive integer")
  if not available:
    return []
  lower = _bound(start, available, available[0])
  upper = _bound(stop, available, available[-1] + 1)
  return [
      f for f in available if lower <= f < upper and (f - lower) % step == 0
  ]


def select_frame_files(paths: Iterable[str], frame: FrameSpec) -> list[str]:
  """The ``paths`` holding the frames ``frame`` selects, in frame order.

  Frames are read from the file names; restart files and files without a
  frame number are left out. Every file of a selected frame is kept (one per
  block of a multiblock run).
  """
  by_frame: dict[int, list[str]] = {}
  for path in sorted(paths):
    name = parse_output_name(path)
    if name is not None and name.frame is not None and not name.restart:
      by_frame.setdefault(name.frame, []).append(path)
  return [
      path for chosen in select_frames(frame, by_frame)
      for path in by_frame.get(chosen, [])
  ]
