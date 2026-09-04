"""File I/O -- bytes <-> dataset arrays.

A leaf layer: one reader per format, dispatched by ``read()``; ``write()`` for
output. Nothing here imports ``gdatastate``/``operations``; the readers fill a plain ``ctx``
dict and return ``(grid, values)`` so the container can construct itself on top.
"""

from __future__ import annotations

import os.path

from . import mapping
from .naming import OutputName, parse_output_name
from .gkyl_c_reader import GkylCReader
from .gkyl_reader import GkylReader
from .gkyl_h5_reader import GkylH5Reader
from .flash_h5_reader import FlashH5Reader
from .writer import save

# Reader registry -- tried in order; extend by adding (name, reader) entries.
# Order is by *specificity* of ``is_compatible()``, most specific / cheapest
# first, so a file never falls into the wrong reader:
#   1. "gkyl_c"  -- native .gkyl via libg0core; the magic-byte + file-type
#                  check is exact and returns modal data as a GkylArray.
#   2. "gkyl"    -- pure-Python .gkyl fallback (no libg0core, partial loads,
#                  dynvectors); same exact magic-byte check as gkyl_c.
#   3. "h5"      -- legacy Gkeyll HDF5 output (predates the native .gkyl
#                  binary format); is_compatible() requires the Gkeyll-specific
#                  "/StructGridField" or "/DataStruct/data" node, so a FLASH
#                  .h5 file (no such nodes) is correctly declined and falls
#                  through to "flash".
#   4. "flash"   -- FLASH code HDF5 output; is_compatible() requires a
#                  "coordinates" node, disjoint from the Gkeyll h5 layout.
# Because 1-2 are checked with the same fast magic-byte test before 3-4 ever
# touch the (slower) tables importer, a .gkyl file never reaches an h5
# reader, and vice versa.
_READERS = {
    "gkyl_c": GkylCReader,
    "gkyl": GkylReader,
    "h5": GkylH5Reader,
    "flash": FlashH5Reader,
}


def read(file_name: str, ctx: dict | None = None, **kwargs):
  """Read ``file_name`` into ``(grid, values)``, populating ``ctx`` in place.

  The reader is chosen by trying each registered reader's ``is_compatible``
  check. ``ctx`` (a plain dict) is filled with metadata -- ``poly_order``,
  ``basis_type``, ``cells``, ``lower``/``upper``, ``time``/``frame``, ... --
  exactly as the legacy reader did.
  """
  if ctx is None:
    ctx = {}
  if not os.path.exists(file_name):
    raise FileNotFoundError(f"No such file: '{file_name}'")
  for reader_cls in _READERS.values():
    reader = reader_cls(file_name=file_name, ctx=ctx, **kwargs)
    if reader.is_compatible():
      reader.preload()
      return reader.load()
  raise NameError(
      f"'{file_name}' cannot be read with any known reader: {list(_READERS)}")


__all__ = [
    "read", "save", "mapping", "naming", "OutputName", "parse_output_name",
    "GkylCReader", "GkylReader", "GkylH5Reader", "FlashH5Reader"
]
