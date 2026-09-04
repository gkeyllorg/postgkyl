"""Equation-blind output discovery -- Gkeyll's file-naming convention.

The ONE home for "what outputs does this directory hold" (CLAUDE.md,
diagnostics layer). Every equation loader in ``gk/`` and every
program-scale diagnostic (layer 13) resolves files through here, never with
private ``glob`` logic of its own -- doctrine V, one home per fact.

Ported from ``src_bak/postgkyl/loader.py``'s ``find_output_stems`` plus a new
``available_frames`` helper factored out of
``src_bak/postgkyl/gk/gk_quantities/gkquantity.py``'s ``_avail_frames_src``
(the gyrokinetic quantity registry no longer globs on its own -- see
``diagnostics/gk/quantity.py``).
"""

from __future__ import annotations

import glob
import os

from postgkyl import io


def find_output_stems(extensions: str = "gkyl", path: str = ".") -> dict:
  """Map each extension to the sorted unique Gkeyll filename stems in ``path``.

  Frame indices and a trailing ``_restart`` are stripped from each stem by
  :func:`postgkyl.io.parse_output_name` -- the one home for Gkeyll's naming
  convention -- rather than by a private regex here.

  Args:
    extensions: Comma-separated list of file extensions to scan.
    path: Directory to scan.

  Returns:
    A dict mapping each extension to a sorted list of unique stems.
  """
  result = {}
  for ext in extensions.split(","):
    unique = []
    for fn in glob.glob(f"{path}/*.{ext:s}"):
      stem = io.parse_output_name(os.path.basename(fn)).stem
      if stem not in unique:
        unique.append(stem)
    result[ext] = sorted(unique)
  return result


def available_frames(stem: str, *, frames: list[int] | None = None) -> set[int]:
  """Set of available frame numbers for a ``<stem><frame>.gkyl`` file family.

  Args:
    stem: The file stem, including any trailing separator before the frame
      number (e.g. ``"path/name-elc_M0_"``).
    frames: Restrict the search to these candidate frame numbers instead of
      globbing the whole directory (cheaper when the caller already has a
      short candidate list).

  Returns:
    The set of frame numbers for which ``<stem><frame>.gkyl`` exists.
  """
  found: set[int] = set()
  if frames:
    candidates = (f"{stem}{f}.gkyl" for f in frames
                  if os.path.isfile(f"{stem}{f}.gkyl"))
  else:
    candidates = glob.glob(f"{glob.escape(stem)}*.gkyl")
  for f in candidates:
    suffix = f[len(stem):-5]
    if suffix.isdigit():
      found.add(int(suffix))
  return found
