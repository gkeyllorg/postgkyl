"""The ``extract_input`` verb -- decode the input file embedded in ``ctx``.

Gkeyll output files may carry the original simulation input file as a
base64-encoded string, stashed by the reader under ``ctx['input_file']``.
This verb is *terminal*: unlike every other verb in this module it returns
a plain ``str``, not a dataset (matching the legacy contract).

No current :mod:`postgkyl.io` reader populates ``ctx['input_file']``; this
verb decodes it whenever a reader does provide it, and returns ``""``
otherwise, exactly as the legacy code did when no input file was embedded.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState


def extract_input(data: "GDataState") -> str:
  """Decode the input file embedded in a Gkeyll output file's ``ctx``.

  Args:
    data: the dataset whose embedded input file is decoded.

  Returns:
    The decoded input-file text, or an empty string when none is embedded.
  """
  encoded = data.ctx.get("input_file")
  if encoded:
    return base64.decodebytes(encoded.encode("utf-8")).decode("utf-8")
  return ""
