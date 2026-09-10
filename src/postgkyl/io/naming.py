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

Before this module the convention was re-derived in three places with three
slightly different rules (``diagnostics.gk.rz._file_prefix``'s
``rsplit('-', 1)``, ``diagnostics.discovery.find_output_stems``'s
``_\\d+$`` strip, and the animation operation's port of main's
``utils.set_frame``, which recovered a frame index by diffing the loaded file
names character by character). It lives in ``io`` because it is knowledge
about Gkeyll's *files*, which is what this layer owns, and because ``io`` is
below every consumer (``gdatastate`` stamps it into ``ctx`` at load time;
``diagnostics`` builds directory discovery on top of it).

The parser is *pure*: it never touches the filesystem. ``os.path.exists`` is
the caller's business.
"""

from __future__ import annotations

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

    For single-block output this is exactly the old
    ``rz._file_prefix`` (the part of the path before the last ``'-'``).
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
