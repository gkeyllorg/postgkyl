"""Run the tutorial scripts and CLI pipelines and compare their actual outputs.

PNG/GIF comparisons use decoded pixels (and GIF timings). Plotly comparisons
use the saved trace data, layout, and configuration, ignoring random HTML IDs.
No cross-machine image snapshots or renderer mocks are used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import numpy as np
from PIL import Image, ImageSequence


def plotly_spec(path: Path) -> list:
  html = path.read_text()
  matches = list(re.finditer(r'Plotly\.newPlot\(\s*"[a-f0-9-]+"\s*,', html))
  if len(matches) != 1:
    raise ValueError(f"Expected one saved Plotly figure in {path}")
  remaining = html[matches[0].end():]
  result = []
  for _ in range(3):
    remaining = remaining.lstrip(" \n\r\t,")
    value, end = json.JSONDecoder().raw_decode(remaining)
    result.append(value)
    remaining = remaining[end:]
  return result


def compare_outputs(python_path: Path, cli_path: Path) -> str:
  if python_path.suffix == ".html":
    if plotly_spec(python_path) != plotly_spec(cli_path):
      raise AssertionError(
          f"Plotly data/layout/configuration differ: {python_path.name}")
    return "Identical Plotly traces, layout, and configuration"
  with Image.open(python_path) as first, Image.open(cli_path) as second:
    if first.n_frames != second.n_frames:
      raise AssertionError(f"Frame counts differ: {python_path.name}")
    for index, (left, right) in enumerate(
        zip(ImageSequence.Iterator(first), ImageSequence.Iterator(second))):
      np.testing.assert_array_equal(
          np.asarray(left.convert("RGBA")),
          np.asarray(right.convert("RGBA")),
          err_msg=f"{python_path.name}: frame {index}")
      if left.info.get("duration") != right.info.get("duration"):
        raise AssertionError(f"Frame timings differ: {python_path.name}")
    return f"Identical pixels ({first.n_frames} frame(s)) and timing"


def run_gallery(root: Path, output: Path) -> dict[str, str]:
  output.mkdir(parents=True, exist_ok=True)
  cli_output = output / "cli"
  cli_output.mkdir(exist_ok=True)
  commands = json.loads((root / "examples/figure_commands.json").read_text())
  env = {**os.environ, "MPLBACKEND": "Agg", "PGKYL_EXAMPLE_OUTPUT": str(output)}
  scripts = sorted((root / "examples/scripts").glob("*.py"))
  results = {}
  for script in scripts:
    if script.name.startswith("_"):
      continue
    subprocess.run([sys.executable, str(script)], cwd=root, env=env, check=True)
    for name, command in commands.get(script.name, {}).items():
      arguments = shlex.split(command)
      if arguments.pop(0) != "pgkyl":
        raise ValueError(f"Expected a pgkyl command for {name}")
      arguments = [
          arg.replace("{output}", str(cli_output)) for arg in arguments
      ]
      subprocess.run([
          sys.executable, "-c", "from postgkyl.cli.app import cli; cli()",
          *arguments
      ],
                     cwd=root,
                     env=env,
                     check=True)
      results[name] = compare_outputs(output / name, cli_output / name)
      print(f"{name}: {results[name]}", flush=True)
  expected = {name for group in commands.values() for name in group}
  if set(results) != expected:
    raise AssertionError(f"Unexecuted figure pairs: {expected - set(results)}")
  (output / "interface-comparison.json"
   ).write_text(json.dumps(results, indent=2) + "\n")
  return results


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output",
                      type=Path,
                      default=Path("build/figure-comparison"))
  args = parser.parse_args()
  run_gallery(Path(__file__).resolve().parents[1], args.output.resolve())
