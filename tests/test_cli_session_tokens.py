"""``command_tokens`` inverts the generated CLI parsing, for every command.

The round trip runs over the whole discovered surface, so a new or changed
public verb is covered without editing this file.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pytest

import postgkyl as pg
from postgkyl.cli.app import MODELS, cli
from postgkyl.cli.compiler import CodecKind, _convert, command_tokens

_MODELS = {model.name: model for model in MODELS}

# (Python value, what parsing its tokens gives back) per scalar codec.
_SCALARS = {
    CodecKind.STRING: [("a b $\\alpha$", "a b $\\alpha$"), ("--grid", "--grid"),
                       (2, "2"), (2.0, "2.0"), (-1.5, "-1.5"),
                       (np.float64(0.5), "0.5"), (Path("d/f"), "d/f")],
    CodecKind.INTEGER: [(3, 3), (-2, -2), (np.int64(4), 4)],
    CodecKind.FLOAT: [(2.5, 2.5), (-1.5, -1.5), (2, 2.0),
                      (np.float32(0.5), 0.5)],
    CodecKind.PATH: [("d/f.gkyl", Path("d/f.gkyl"))],
}


def _samples(codec):
  """``(value, parsed)`` pairs covering one codec."""
  if codec.kind in _SCALARS:
    return _SCALARS[codec.kind]
  if codec.kind is CodecKind.BOOLEAN:
    return [(True, True)]
  if codec.kind is CodecKind.CHOICE:
    return [(choice, choice) for choice in codec.choices]
  if codec.kind is CodecKind.ENUM:
    return [(choice, codec.python_type(choice)) for choice in codec.choices]
  if codec.kind is CodecKind.SEQUENCE:
    items = _samples(codec.items[0])[:2]
    return [([value for value, _ in items], [parsed for _, parsed in items])]
  if codec.kind is CodecKind.TUPLE:
    items = [_samples(item)[1] for item in codec.items]
    return [(tuple(value for value, _ in items),
             tuple(parsed for _, parsed in items))]
  if codec.kind is CodecKind.MAPPING:
    return [({"key": "a b", "other": 2}, {"key": "a b", "other": "2"})]
  raise AssertionError(codec.kind)


def _parse(model, tokens):
  """Parse ``model``'s tokens as ``pgkyl``'s chain does, before ``info``."""
  context = click.Context(cli, info_name="pgkyl")
  name, command, rest = cli.resolve_command(context,
                                            [model.name, *tokens, "info"])
  parsed = command.make_context(name,
                                rest,
                                parent=context,
                                allow_extra_args=True,
                                allow_interspersed_args=False)
  assert parsed.args == ["info"], "the next command must stay untouched"
  return parsed.params


def _exposed(model):
  return [parameter for parameter in model.parameters if not parameter.injected]


def _parameter_samples(parameter):
  """A dataset reference is spelled by its tag; anything else by its codec."""
  if parameter.dataset_ref:
    return [("w", "w")]
  return _samples(parameter.codec)


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_tokens_parse_back_to_the_values(name):
  model = _MODELS[name]
  exposed = _exposed(model)
  # Required and positional parameters always get a value: an omitted
  # optional positional would read the following command's name.
  base = {
      parameter.name: _parameter_samples(parameter)[0]
      for parameter in exposed if parameter.required or parameter.argument
  }
  cases = []
  for parameter in exposed:
    for sample in _parameter_samples(parameter):
      cases.append({**base, parameter.name: sample})
  cases.append({
      parameter.name: _parameter_samples(parameter)[0]
      for parameter in exposed
  })
  for case in cases:
    values = {key: value for key, (value, _) in case.items()}
    params = _parse(model, command_tokens(model, values))
    for parameter in exposed:
      if parameter.name not in case:
        continue
      parsed = params[parameter.name]
      if not parameter.dataset_ref:
        parsed = _convert(parsed, parameter.codec)
      assert parsed == case[parameter.name][1], (parameter.name, values)


def test_unset_values_have_no_tokens():
  plot = _MODELS["plot"]
  assert command_tokens(_MODELS["select"], {"z0": None, "comp": None}) == []
  assert command_tokens(plot, {"no_show": False, "dpi": 200}) == []
  assert command_tokens(plot, {"no_show": True}) == ["--no_show"]


def test_options_come_before_positionals_and_a_dash_value_after_dashes():
  evaluate = _MODELS["evaluate"]
  assert command_tokens(evaluate, {
      "chain": "f0 f1 +",
      "tag": "t"
  }) == ["--tag", "t", "f0 f1 +"]
  tokens = command_tokens(evaluate, {"chain": "-1 f0 *"})
  assert tokens == ["--", "-1 f0 *"]
  assert _parse(evaluate, tokens)["chain"] == "-1 f0 *"


def test_floats_keep_their_decimal_point():
  select = _MODELS["select"]
  tokens = command_tokens(select, dict(z0=2.0, z1=2))
  assert tokens == ["--z0", "2.0", "--z1", "2"]


@pytest.mark.parametrize("name, values, reason", [
    ("select", dict(z0=slice(1, 3)), "slice has no string spelling"),
    ("plot", dict(dpi=np.array([1, 2])), "ndarray has no integer spelling"),
    ("plot", dict(dpi=True), "a boolean has no scalar spelling"),
    ("plot", dict(no_show=1), "expected a bool"),
    ("average", dict(dims=1), "expected a list or tuple"),
    ("average", dict(dims="1"), "expected a list or tuple"),
    ("plot", dict(figsize=(1.0, )), "expected 2 values"),
])
def test_values_without_a_spelling_are_refused(name, values, reason):
  with pytest.raises(TypeError, match=reason):
    command_tokens(_MODELS[name], values)


def test_a_dataset_reference_must_be_a_tag():
  data = pg.GData(tag="w")
  with pytest.raises(TypeError, match="referred to by its tag"):
    command_tokens(_MODELS["average"], {"dims": [0], "weight": data})
  assert command_tokens(_MODELS["average"], {
      "dims": [0],
      "weight": "w"
  }) == ["--dims", "0", "--weight", "w"]
