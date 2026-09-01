"""Deprecated module path plus parser retained for legacy callers."""

import re

from postgkyl.cli.commands import command_named

command = command_named("gyrokinetics-load-gk-quantity")


def _number(value: str):
  try:
    integer = int(value)
    return integer if str(integer) == value else float(value)
  except ValueError:
    try:
      return float(value)
    except ValueError:
      return value
    # end
  # end
# end


def _parse_extra(extra: str | None) -> dict:
  """Parse legacy comma/space separated quantity key/value pairs."""
  if not extra:
    return {}
  # end
  matches = list(re.finditer(r"(?:^|[ ,]+)([A-Za-z_]\w*)=", extra))
  result = {}
  for index, match in enumerate(matches):
    end = matches[index + 1].start() if index + 1 < len(matches) else len(extra)
    raw = extra[match.end():end].strip(" ,")
    values = [_number(item.strip()) for item in raw.split(",") if item.strip()]
    result[match.group(1)] = values[0] if len(values) == 1 else values
  # end
  return result
# end
