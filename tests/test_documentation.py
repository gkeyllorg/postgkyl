"""Contracts keeping Python hover help and generated CLI help in sync."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import pydoc

import postgkyl as pg
from postgkyl.cli.commands import GENERATED_COMMANDS, MODELS
from postgkyl.cli.docstrings import parse_docstring
from postgkyl.gdata.gdata import GData


def _public_documented_members(cls) -> dict[str, object]:
  members = {}
  for name in dir(cls):
    if name.startswith("_"):
      continue
    # end
    value = getattr(cls, name)
    if callable(value) or isinstance(value, property):
      members[name] = value
    # end
  # end
  return members
# end


def _function_source_doc(function) -> str | None:
  path = Path(inspect.getsourcefile(function))
  tree = ast.parse(path.read_text(), path)
  matches = [node for node in ast.walk(tree)
      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
      and node.name == function.__name__]
  assert len(matches) == 1, f"could not locate {function.__qualname__} in {path}"
  return ast.get_docstring(matches[0], clean=True)
# end


def test_public_python_surfaces_have_hover_documentation():
  missing = []
  for name in pg.__all__:
    value = getattr(pg, name)
    if callable(value) and not inspect.getdoc(value):
      missing.append(f"postgkyl.{name}")
    # end
    if inspect.isfunction(value):
      assert _function_source_doc(value) == inspect.getdoc(value)
    # end
  # end
  for cls in (pg.GData, pg.GDataGroup):
    for name, value in _public_documented_members(cls).items():
      if not inspect.getdoc(value):
        missing.append(f"{cls.__name__}.{name}")
      # end
    # end
  # end
  assert missing == []
# end


def test_fluent_operations_are_static_aliases_to_documented_functions():
  aliases = {
      name: value for name, value in GData.__dict__.items()
      if inspect.isfunction(value) and (
          value.__module__.startswith("postgkyl.operations")
          or value.__module__ == "postgkyl.io.writer")
  }
  assert aliases

  path = Path(inspect.getsourcefile(GData))
  tree = ast.parse(path.read_text(), path)
  class_node = next(node for node in tree.body
      if isinstance(node, ast.ClassDef) and node.name == "GData")
  class_assignments = {
      target.id for node in class_node.body if isinstance(node, ast.Assign)
      for target in node.targets if isinstance(target, ast.Name)
  }
  assert set(aliases) <= class_assignments

  instance = GData()
  for name, function in aliases.items():
    method = getattr(instance, name)
    assert GData.__dict__[name] is function
    assert inspect.getdoc(method) == inspect.getdoc(function)
    assert tuple(inspect.signature(method).parameters) == tuple(
        inspect.signature(function).parameters)[1:]
  # end
# end


def test_shared_functional_and_fluent_spellings_are_one_object():
  exceptions = {"load", "plot", "val2coord"}
  shared = set(pg.__all__) & set(GData.__dict__) - exceptions
  paired = {
      name for name in shared
      if inspect.isfunction(getattr(pg, name))
      and inspect.isfunction(GData.__dict__[name])
  }
  assert paired
  for name in paired:
    assert GData.__dict__[name] is getattr(pg, name)
  # end
# end


def test_cli_help_is_lowered_from_source_docstrings():
  commands = {command.model.name: command for command in GENERATED_COMMANDS}
  for model in MODELS:
    parsed = parse_docstring(model.canonical,
        required=set(inspect.signature(model.canonical).parameters),
        signature_names=set(inspect.signature(model.canonical).parameters))
    source_doc = _function_source_doc(model.canonical)

    assert source_doc == inspect.getdoc(model.canonical)
    assert "Value for ``" not in source_doc
    assert model.help == parsed.summary
    assert model.long_help == parsed.long_help

    command = commands[model.name]
    assert command.short_help == parsed.summary
    assert command.help == parsed.long_help
    click_parameters = {parameter.name: parameter for parameter in command.params}
    for parameter in model.parameters:
      if not parameter.injected:
        assert click_parameters[parameter.name].help == parsed.parameters[parameter.name]
      # end
    # end
  # end
# end


def test_python_help_renders_for_function_and_bound_method():
  summary = "Interpolate DG (modal/nodal) data onto a uniform evaluation mesh."
  assert summary in pydoc.render_doc(pg.interpolate)
  assert summary in pydoc.render_doc(GData().interpolate)
# end


def test_distribution_marks_inline_types_for_editor_tools():
  marker = Path(pg.__file__).with_name("py.typed")
  assert marker.is_file()
# end
