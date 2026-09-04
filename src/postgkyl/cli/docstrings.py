"""Strict parser for the Google-style subset used by generated commands."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import re


class DocstringError(ValueError):
  """A canonical callable's documentation cannot be lowered losslessly."""


@dataclass(frozen=True)
class ParsedDocstring:
  summary: str
  long_help: str
  parameters: dict[str, str]


_SECTION = re.compile(r"^([A-Za-z][A-Za-z ]*):\s*$")
_ARG = re.compile(r"^\s{2,}(\*{0,2}[A-Za-z_]\w*)(?:\s*\([^)]*\))?:\s*(.*)$")


def _paragraph(lines: list[str]) -> str:
  out: list[str] = []
  for line in lines:
    if not line.strip() or _SECTION.match(line):
      if out:
        break
      continue
    out.append(line.strip())
  return " ".join(out)


def _narrative(lines: list[str]) -> str:
  """Return prose preceding the first structured docstring section."""
  end = next(
      (index for index, line in enumerate(lines) if _SECTION.match(line)),
      len(lines))
  return "\n".join(lines[:end]).strip()


def parse_docstring(obj,
                    *,
                    required: set[str] | None = None,
                    signature_names: set[str] | None = None) -> ParsedDocstring:
  """Parse and validate one canonical callable's command documentation."""
  qualname = f"{getattr(obj, '__module__', '<unknown>')}.{getattr(obj, '__qualname__', obj)}"
  doc = inspect.getdoc(obj)
  if not doc:
    raise DocstringError(f"{qualname}: missing docstring")
  lines = doc.splitlines()
  summary = _paragraph(lines)
  if not summary:
    raise DocstringError(f"{qualname}: missing first-paragraph description")

  entries: dict[str, str] = {}
  args_sections = [i for i, line in enumerate(lines) if line == "Args:"]
  if len(args_sections) > 1:
    raise DocstringError(f"{qualname}: duplicate Args sections")
  args_at = args_sections[0] if args_sections else None
  if args_at is not None:
    current: str | None = None
    chunks: list[str] = []

    def finish() -> None:
      nonlocal current, chunks
      if current is None:
        return
      text = " ".join(part for part in chunks if part).strip()
      if not text:
        raise DocstringError(
            f"{qualname}: parameter {current!r} has no description")
      if current in entries:
        raise DocstringError(
            f"{qualname}: parameter {current!r} is documented twice")
      entries[current] = text

    for line in lines[args_at + 1:]:
      if line and not line[0].isspace() and _SECTION.match(line):
        break
      match = _ARG.match(line)
      if match:
        finish()
        current = match.group(1).lstrip("*")
        chunks = [match.group(2).strip()]
      elif current is not None and (not line.strip()
                                    or line.startswith("    ")):
        chunks.append(line.strip())
      elif line.strip():
        raise DocstringError(
            f"{qualname}: malformed Args entry: {line.strip()!r}")
    finish()

  signature_names = signature_names or set()
  unknown = set(entries) - signature_names
  if unknown:
    names = ", ".join(sorted(unknown))
    raise DocstringError(
        f"{qualname}: documented parameter(s) absent from signature: {names}")
  for name in sorted(required or ()):
    if name not in entries:
      raise DocstringError(f"{qualname}: parameter {name!r} is undocumented")
  return ParsedDocstring(summary=summary,
                         long_help=_narrative(lines),
                         parameters=entries)


__all__ = ["DocstringError", "ParsedDocstring", "parse_docstring"]
