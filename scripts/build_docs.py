"""Prepare Postgkyl's Sphinx subtree from this checkout and execute its gallery.

Run with Python 3.12 after installing ``.[docs]``. The output is disposable;
an ownership marker prevents deleting a directory that this script did not make.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import textwrap
from types import ModuleType
import zipfile


def public_functions(module: ModuleType, seen_modules: set):
  """Yield public function paths through declared module exports."""
  if module in seen_modules:
    return
  seen_modules.add(module)
  names = getattr(module, "__all__", sorted(vars(module)))
  for name in names:
    if name.startswith("_"):
      continue
    value = getattr(module, name)
    if inspect.isfunction(value) and value.__module__.startswith("postgkyl"):
      yield f"{module.__name__}.{name}", value
    elif module.__name__.startswith("postgkyl.diagnostics") and isinstance(
        value,
        ModuleType) and value.__name__.startswith("postgkyl.diagnostics"):
      yield from public_functions(value, seen_modules)


def write_reference(output: Path) -> dict:
  """Render the real public functions and compiled Click command inventory."""
  import click
  import postgkyl as pg
  from postgkyl.cli.app import cli, COMMAND_SECTIONS, MODELS

  reference = output / "reference"
  reference.mkdir()
  api = [
      "Python API\n==========\n",
      "Signatures and descriptions below come from the installed source.\n"
  ]
  seen_functions = {}
  api_pages = {}
  command_names = {}
  for model in MODELS:
    command_names.setdefault(model.canonical, []).append(model.name)
  roots = [("Core", pg)] + [(f"{name} diagnostics", getattr(
      pg.diagnostics, name)) for name in pg.diagnostics.__all__]
  seen_modules = set()
  for title, module in roots:
    api.extend(
        [f"{title}\n{'-' * len(title)}\n", ".. toctree::\n   :maxdepth: 1\n"])
    aliases = []
    for path, function in public_functions(module, seen_modules):
      if function in seen_functions:
        aliases.append(
            f"``{path}`` is an alias of :func:`{seen_functions[function]}`.\n")
        continue
      seen_functions[function] = path
      page = "api-" + path.replace(".", "-")
      api_pages[path] = page
      (reference / f"{page}.rst"
       ).write_text(f"{path}\n{'=' * len(path)}\n\n.. autofunction:: {path}\n")
      with (reference / f"{page}.rst").open("a") as page_file:
        for name in command_names.get(function, ()):
          page_file.write(f"\n:doc:`CLI: {name} <cli-{name}>`\n")
      api.append(f"   {page}")
    api.extend(["", *aliases])
  api.append(
      "Fluent datasets\n---------------\n\n.. toctree::\n   :maxdepth: 1\n")
  for name in ("GData", "GDataGroup"):
    cls = getattr(pg, name)
    class_page = f"api-{name}"
    api.append(f"   {class_page}")
    members = [
        f"{name}\n{'=' * len(name)}\n", f".. autoclass:: postgkyl.{name}\n\n",
        ".. toctree::\n   :maxdepth: 1\n"
    ]
    aliases = []
    for member, value in inspect.getmembers(cls):
      if member.startswith("_") or not (callable(value)
                                        or isinstance(value, property)):
        continue
      path = f"postgkyl.{name}.{member}"
      if value in seen_functions:
        aliases.append(f"``{path}`` uses :func:`{seen_functions[value]}`.\n")
        continue
      page = "api-" + path.replace(".", "-")
      kind = "autoattribute" if isinstance(value, property) else "automethod"
      (reference / f"{page}.rst"
       ).write_text(f"{path}\n{'=' * len(path)}\n\n.. {kind}:: {path}\n")
      api_pages[path] = page
      members.append(f"   {page}")
    (reference / f"{class_page}.rst").write_text("\n".join(
        [*members, "", *aliases]))
  api.append("")
  (reference / "api.rst").write_text("\n".join(api))

  commands = []
  contents = [
      "Command reference\n=================\n",
      "This inventory and every help block are generated from the "
      "same Click commands used by ``pgkyl --help``.\n"
  ]
  for section, names in COMMAND_SECTIONS.items():
    contents.extend([
        section, "-" * len(section), "", ".. toctree::", "   :maxdepth: 1", ""
    ])
    for name in names:
      command = cli.commands[name]
      with click.Context(command, info_name=f"pgkyl {name}",
                         terminal_width=88) as context:
        help_text = command.get_help(context)
      title = f"pgkyl {name}"
      page = (f"{title}\n{'=' * len(title)}\n\n.. code-block:: text\n\n" +
              textwrap.indent(help_text, "   ") + "\n")
      model = next(model for model in MODELS if model.name == name)
      python_path = seen_functions.get(model.canonical)
      if python_path is not None:
        page += f"\n:doc:`Python API <{api_pages[python_path]}>`\n"
      (reference / f"cli-{name}.rst").write_text(page)
      contents.append(f"   cli-{name}")
      commands.append(name)
    contents.append("")
  (reference / "cli.rst").write_text("\n".join(contents))
  quantities = "\n".join(f"* ``{name}``"
                         for name in pg.gk.available_quantities())
  (reference / "quantities.rst").write_text(
      "Gyrokinetic quantities\n======================\n\n"
      "Names below come from ``pg.gk.available_quantities()``. "
      "See :func:`postgkyl.diagnostics.gk.load_quantity` for loading and "
      "physical parameters.\n\n" + quantities + "\n")
  return {
      "api": list(seen_functions.values()),
      "api_pages": api_pages,
      "commands": commands
  }


def write_figure_pairs(root: Path, output: Path, comparison: dict) -> None:
  """Publish the commands that passed parity, beside the actual two outputs."""
  commands = json.loads((root / "examples/figure_commands.json").read_text())
  pairs = output / "_pairs"
  pairs.mkdir()
  interactive = output / "interactive"
  interactive.mkdir()
  report = [
      "Python and CLI figure comparisons\n=================================\n",
      "Every result below was checked during this documentation build. "
      "Raster comparisons decode pixels and animation timings; Plotly "
      "comparisons check trace data, layout, and configuration. They "
      "do not compare browser-specific rasterization.\n",
      ".. list-table::\n   :header-rows: 1\n\n"
      "   * - Output\n     - Comparison\n"
  ]
  for script, outputs in commands.items():
    snippet = [
        "**Python script**\n", ".. code-block:: bash\n\n"
        f"   python examples/scripts/{script}\n",
        f".. literalinclude:: _inputs/examples/scripts/{script}\n"
        "   :language: python\n", "**Equivalent CLI commands**\n",
        "Run from the extracted example bundle (or repository root). "
        "Create an output directory first:\n",
        ".. code-block:: bash\n\n   mkdir -p output\n"
    ]
    for name, command in outputs.items():
      arguments = [
          arg.replace("{output}", "output") for arg in shlex.split(command)
      ]
      # Wrap only between tokens; shell quoting remains correct for math labels.
      lines = []
      current = ""
      for arg in arguments:
        token = shlex.quote(arg)
        if current and len(current) + len(token) > 90:
          lines.append(current + " \\")
          current = "    " + token
        else:
          current += (" " if current else "") + token
      lines.append(current)
      snippet.append(".. code-block:: bash\n\n" +
                     textwrap.indent("\n".join(lines), "   ") + "\n")
      snippet.append(f"``{name}``: {comparison[name]}.\n")
      report.append(f"   * - ``{name}``\n     - {comparison[name]}\n")
      if name.endswith(".html"):
        for label, source in (("Python", output / "figures" / name),
                              ("CLI", output / "figures/cli" / name)):
          target = f"{label.lower()}-{name}"
          shutil.copyfile(source, interactive / target)
          snippet.append(
              f"{label}: :download:`open interactive figure <interactive/{target}>`.\n"
          )
          snippet.append(
              ".. raw:: html\n\n"
              f'   <iframe src="interactive/{target}" title="{label}: {name}" '
              'width="100%" height="480" loading="lazy" '
              'style="border:0"></iframe>\n')
      else:
        snippet.append(
            ".. list-table:: Python and CLI outputs\n   :widths: 50 50\n\n"
            f"   * - .. image:: figures/{name}\n"
            f"          :alt: Python output for {name}\n"
            "          :width: 100%\n"
            f"     - .. image:: figures/cli/{name}\n"
            f"          :alt: CLI output for {name}\n"
            "          :width: 100%\n")
    (pairs / f"{Path(script).stem}.inc").write_text("\n".join(snippet))
  (output / "interface-equivalence.rst").write_text("\n".join(report))


def prepare(root: Path, output: Path) -> None:
  """Execute this checkout's examples and stage their sources and outputs."""
  import postgkyl as pg
  from postgkyl import gpython

  if not Path(pg.__file__).resolve().is_relative_to(root / "src"):
    raise RuntimeError(
        "Install this checkout with pip install -e '.[docs]' first")
  gpython.require()
  marker = output / ".postgkyl-docs"
  if output.exists():
    if not marker.is_file():
      raise ValueError(
          f"Refusing to replace unowned output directory: {output}")
    shutil.rmtree(output)
  shutil.copytree(root / "docs" / "source", output)
  marker.touch()
  inputs = output / "_inputs"
  shutil.copytree(root / "examples",
                  inputs / "examples",
                  ignore=shutil.ignore_patterns("output", "__pycache__"))

  subprocess.run([sys.executable,
                  str(root / "tests/generate_test_data.py")],
                 cwd=root,
                 check=True,
                 stdout=subprocess.DEVNULL)
  figures = output / "figures"
  figures.mkdir()
  gallery = runpy.run_path(str(root / "examples/compare_interfaces.py"))
  comparison = gallery["run_gallery"](root, figures)
  write_figure_pairs(root, output, comparison)
  shutil.copytree(root / "docs/_ext", output / "_ext")

  readme = (root / "README.md").read_text()
  installation = readme.split("## Installation\n",
                              1)[1].split("## Documentation\n", 1)[0]
  notes = readme.split("### Additional installation notes\n",
                       1)[1].split("## Developing for Postgkyl\n", 1)[0]
  installation = "# Installation\n" + installation + "## Additional notes\n" + notes
  installation = re.sub(r"^#(#{2,} )", r"\1", installation, flags=re.MULTILINE)
  for name in ("environment.yml", "pyproject.toml"):
    installation = installation.replace(
        f"]({name})",
        f"](https://github.com/ammarhakim/postgkyl/blob/main/{name})")
  (output / "installation.md").write_text(installation)

  inventory = write_reference(output)
  revision = subprocess.check_output(
      ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
  inventory.update(version=pg.__version__, revision=revision)
  (output /
   "build-info.json").write_text(json.dumps(inventory, indent=2) + "\n")
  (output / "provenance.rst").write_text(
      "Documentation source\n====================\n\n"
      "The hosted documentation is rebuilt from Postgkyl's ``main`` branch. "
      "Local and pull-request previews use their working checkout.\n\n"
      f"This build used version ``{pg.__version__}``, commit ``{revision}``. "
      "Uncommitted local changes, if present, are included in local previews.\n\n"
      "`Edit the guides in Postgkyl "
      "<https://github.com/ammarhakim/postgkyl/tree/main/docs/source>`_. "
      "API descriptions are edited in the implementing Python functions.\n")

  # One download preserves the paths used by both the scripts and CLI tutorial.
  downloads = output / "downloads"
  downloads.mkdir()
  with zipfile.ZipFile(downloads / "postgkyl-examples.zip",
                       "w",
                       compression=zipfile.ZIP_DEFLATED) as bundle:
    files = list((inputs / "examples").rglob("*"))
    for path in sorted(files):
      if path.is_file():
        bundle.write(path, path.relative_to(inputs))
    for path in sorted((root / "tests/test_data").rglob("*.gkyl")):
      bundle.write(path, path.relative_to(root))
    bundle.write(root / "tests/generate_test_data.py",
                 "tests/generate_test_data.py")
    bundle.write(output / "build-info.json", "build-info.json")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=Path("build/docs/source"))
  args = parser.parse_args()
  prepare(Path(__file__).resolve().parents[1], args.output.resolve())


if __name__ == "__main__":
  main()
