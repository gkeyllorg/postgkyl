"""Natural/numeric sort key for strings (pure)."""

from __future__ import annotations

import re

_CHUNK_RE = re.compile(r"(\d+)")


def natural_sort_key(s: str) -> tuple:
  """Split ``s`` into alternating text/int chunks so embedded digit runs
  compare numerically instead of character-by-character (``'field_2'`` sorts
  before ``'field_10'``, unlike a plain lexicographic string sort)."""
  return tuple(int(c) if c.isdigit() else c for c in _CHUNK_RE.split(s))
