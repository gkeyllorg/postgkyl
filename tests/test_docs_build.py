"""Exercise the publishable docs, including the downloadable example bundle."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
import runpy
import subprocess
import sys
import zipfile

import pytest

from postgkyl import gpython
from postgkyl.cli.app import cli

pytest.importorskip("sphinx", reason="install the docs extra for website tests")
pytestmark = pytest.mark.skipif(not gpython.available(),
                                reason="needs compiled Gkeyll")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def documentation(tmp_path_factory):
  destination = tmp_path_factory.mktemp("docs")
  source = destination / "source"
  subprocess.run([
      sys.executable,
      str(ROOT / "scripts/build_docs.py"), "--output",
      str(source)
  ],
                 cwd=ROOT,
                 check=True,
                 capture_output=True,
                 text=True)
  subprocess.run([
      sys.executable, "-m", "sphinx", "-W", "--keep-going", "-b", "html", "-c",
      str(ROOT / "docs"),
      str(source),
      str(destination / "html")
  ],
                 check=True,
                 capture_output=True,
                 text=True)
  return destination


def test_published_inventory_and_navigation(documentation):
  inventory = json.loads((documentation / "source/build-info.json").read_text())
  assert set(inventory["commands"]) == set(cli.commands)
  index = (documentation / "html/index.html").read_text()
  assert 'href="examples.html"' in index
  with (documentation /
        "html/.doctrees/environment.pickle").open("rb") as stream:
    environment = pickle.load(stream)
  assert environment.toctree_includes["index"] == [
      "installation", "examples", "reference/api", "reference/cli", "concepts",
      "reference/quantities", "contributing", "provenance"
  ]
  examples = environment.toctree_includes["examples"]
  assert {"cli-tutorial", "interface-equivalence"} <= set(examples)
  pairs = []
  for page in examples:
    assert (documentation / f"html/{page}.html").is_file()
    source = (documentation / f"source/{page}.rst").read_text()
    includes = [
        line for line in source.splitlines()
        if line.startswith(".. include:: _pairs/")
    ]
    assert len(includes) <= 1, page
    pairs.extend(line.split("_pairs/")[1] for line in includes)
  assert sorted(pairs) == sorted(
      path.name for path in (documentation / "source/_pairs").glob("*.inc"))
  assert 'href="reference/cli.html"' in index
  api = (documentation / "html/reference/api.html").read_text()
  assert "postgkyl.load" in api
  for path, page in inventory["api_pages"].items():
    assert (documentation / f"html/reference/{page}.html").is_file(), path
  load_page = inventory["api_pages"]["postgkyl.load"]
  assert "poly_order" in (documentation /
                          f"html/reference/{load_page}.html").read_text()
  pressure_page = inventory["api_pages"][
      "postgkyl.diagnostics.mom.five_moment.pressure"]
  assert "gas_gamma" in (documentation /
                         f"html/reference/{pressure_page}.html").read_text()
  comparison = json.loads(
      (documentation / "source/figures/interface-comparison.json").read_text())
  commands = json.loads((ROOT / "examples/figure_commands.json").read_text())
  assert set(comparison) == {
      name
      for outputs in commands.values()
      for name in outputs
  }
  for prefix in ("python", "cli"):
    assert (documentation /
            f"html/interactive/{prefix}-08_surface.html").is_file()
    assert (documentation /
            f"html/interactive/{prefix}-08_volume.html").is_file()


def test_downloaded_examples_run_outside_repository(documentation, tmp_path):
  with zipfile.ZipFile(documentation /
                       "source/downloads/postgkyl-examples.zip") as bundle:
    bundle.extractall(tmp_path)
  env = {
      **os.environ, "MPLBACKEND": "Agg",
      "PGKYL_EXAMPLE_OUTPUT": str(tmp_path / "figures")
  }
  subprocess.run([
      sys.executable,
      str(tmp_path / "examples/compare_interfaces.py"), "--output",
      str(tmp_path / "figures")
  ],
                 cwd=tmp_path,
                 env=env,
                 check=True,
                 capture_output=True,
                 text=True)
  assert (tmp_path / "figures/06_growth.png").stat().st_size > 0
  assert (tmp_path / "figures/05_gk_rz.png").stat().st_size > 0


def test_comparison_rejects_changed_pixels(tmp_path):
  from PIL import Image
  first, second = tmp_path / "python.png", tmp_path / "cli.png"
  Image.new("RGB", (4, 4), "white").save(first)
  changed = Image.new("RGB", (4, 4), "white")
  changed.putpixel((1, 1), (0, 0, 0))
  changed.save(second)
  compare = runpy.run_path(str(
      ROOT / "examples/compare_interfaces.py"))["compare_outputs"]
  with pytest.raises(AssertionError):
    compare(first, second)


def test_plotly_comparison_ignores_only_html_id(tmp_path):
  first, second = tmp_path / "python.html", tmp_path / "cli.html"
  first.write_text(
      'Plotly.newPlot("aaaa", [{"z":[1,2]}], {"title":"Density"}, {})')
  second.write_text(
      'Plotly.newPlot("bbbb", [{"z":[1,2]}], {"title":"Density"}, {})')
  compare = runpy.run_path(str(
      ROOT / "examples/compare_interfaces.py"))["compare_outputs"]
  assert "Identical Plotly" in compare(first, second)
  second.write_text(second.read_text().replace('[1,2]', '[1,3]'))
  with pytest.raises(AssertionError, match="Plotly"):
    compare(first, second)


def test_animation_comparison_checks_timing(tmp_path):
  from PIL import Image
  first, second = tmp_path / "python.gif", tmp_path / "cli.gif"
  frames = [Image.new("RGB", (4, 4), color) for color in ("white", "black")]
  for path, duration in ((first, 100), (second, 200)):
    frames[0].save(path,
                   save_all=True,
                   append_images=frames[1:],
                   duration=duration)
  compare = runpy.run_path(str(
      ROOT / "examples/compare_interfaces.py"))["compare_outputs"]
  with pytest.raises(AssertionError, match="timings"):
    compare(first, second)


def test_preparation_preserves_unowned_directory(tmp_path):
  existing = tmp_path / "notes.txt"
  existing.write_text("Keep this file")
  prepare = runpy.run_path(str(ROOT / "scripts/build_docs.py"))["prepare"]
  with pytest.raises(ValueError, match="unowned"):
    prepare(ROOT, tmp_path)
  assert existing.read_text() == "Keep this file"
