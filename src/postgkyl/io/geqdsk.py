"""GEQDSK (EFIT) equilibrium files: the header scalars and the limiter.

The header's second to fifth lines hold the magnetic axis and the poloidal
flux at the axis and at the boundary twice. Some files transpose the first
copy (STEP's ``step.geqdsk`` swaps ``simag`` and the axis position on its
third line), and Gkeyll reads the later copies, so these are read here:
``simag`` and ``rmaxis`` on the fourth line, ``zmaxis`` and ``sibry`` on the
fifth.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

_FLOAT = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?")
# Positions of the scalars among the floats of lines 2-6 (five per line):
# line 4 holds current, simag, xdum, rmaxis, xdum; line 5 zmaxis, xdum, sibry.
_HEADER_INDEX = {"simag": 11, "rmaxis": 13, "zmaxis": 15, "sibry": 17}


@dataclass(frozen=True)
class GeqdskHeader:
  """Magnetic axis (m) and poloidal flux at the axis and the boundary.

  ``nan`` where the header is too short to hold the value.
  """

  rmaxis: float
  zmaxis: float
  simag: float
  sibry: float


def read_geqdsk_header(file_name: str) -> GeqdskHeader:
  """The magnetic axis and the axis and boundary flux of a GEQDSK file."""
  with open(file_name) as fh:
    lines = fh.read().split("\n")[1:6]
  floats = [float(x) for line in lines for x in _FLOAT.findall(line)]
  return GeqdskHeader(
      **{
          key: floats[i] if i < len(floats) else np.nan
          for key, i in _HEADER_INDEX.items()
      })


def read_geqdsk_limiter(file_name: str) -> np.ndarray | None:
  """The limiter outline ``(n + 1, 2)`` in (R, Z), closed by repeating its
  first point; ``None`` when the file lists fewer than three limiter points.

  The boundary and limiter point counts are the first line holding exactly
  two integers after the header's arrays; the boundary's ``(R, Z)`` pairs
  follow, then the limiter's.
  """
  with open(file_name) as fh:
    lines = fh.read().split("\n")
  floats: list[float] = []
  counts = None
  for line in lines[1:]:
    if counts is None and len(floats) > 20 and re.fullmatch(
        r"\s*\d+\s+\d+\s*", line):
      counts = (len(floats), [int(x) for x in line.split()])
      continue
    floats += [float(x) for x in _FLOAT.findall(line)]
  if counts is None:
    return None
  start, (nbdy, nlim) = counts
  start += 2 * nbdy
  lim = np.array(floats[start:start + 2 * nlim]).reshape(-1, 2)
  return np.vstack([lim, lim[:1]]) if len(lim) >= 3 else None
