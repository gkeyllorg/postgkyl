"""Regenerate src/postgkyl/cli/session.pyi from the CLI command models.

IDEs and type checkers cannot see PostgkylSession's runtime methods, so this
stub declares them. The ``session-stub`` pre-commit hook reruns this when
``src/postgkyl`` changes; a test fails while the stub is out of date.
"""

import argparse
from pathlib import Path

STUB = Path(__file__).resolve().parents[1] / "src/postgkyl/cli/session.pyi"


def main():
  argparse.ArgumentParser(description=__doc__).parse_args()
  from postgkyl.cli.session import render_stub

  STUB.write_text(render_stub())
  print(f"wrote {STUB}")


if __name__ == "__main__":
  main()
