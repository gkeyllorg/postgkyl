"""Regenerate src/postgkyl/cli/session.pyi from the CLI command models.

IDEs and type checkers cannot see PostgkylSession's runtime methods, so this
stub declares them. The ``session-stub`` pre-commit hook reruns this when
``src/postgkyl`` changes; a test fails while the stub is out of date.
"""

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUB = ROOT / "src/postgkyl/cli/session.pyi"
_YAPF_VERSION = "0.43.0"  # Keep aligned with pyproject.toml and pre-commit.


def format_stub(source: str) -> str:
  """Format generated declarations exactly as the pre-commit YAPF hook does."""
  try:
    installed = version("yapf")
  except PackageNotFoundError as exc:
    raise RuntimeError(f"Stub generation requires YAPF {_YAPF_VERSION}; "
                       "install the project with `pip install -e '.[test]'`.") \
        from exc
  if installed != _YAPF_VERSION:
    raise RuntimeError(f"Stub generation requires YAPF {_YAPF_VERSION}; "
                       f"found {installed}. Install the project with "
                       "`pip install -e '.[test]'`.")
  from yapf.yapflib.yapf_api import FormatCode

  formatted, _ = FormatCode(source,
                            filename=str(STUB),
                            style_config=str(ROOT / ".style.yapf"))
  return formatted


def main():
  argparse.ArgumentParser(description=__doc__).parse_args()
  from postgkyl.cli.session import render_stub

  generated = format_stub(render_stub())
  if STUB.exists() and STUB.read_text() == generated:
    print(f"up to date {STUB}")
    return
  STUB.write_text(generated)
  print(f"wrote {STUB}")


if __name__ == "__main__":
  main()
