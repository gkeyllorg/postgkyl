"""Compatibility command objects backed by canonical generated models."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import click

from .compat import warn_legacy
from .compiler import execute_model


@dataclass(frozen=True)
class LegacyOption:
  spelling: str
  name: str
  default: object = None
  type: object = str
  required: bool = False
# end


@dataclass(frozen=True)
class LegacyModel:
  target: str
  options: tuple[LegacyOption, ...]
  rename: object
# end


def _option(spelling, name=None, default=None, type=str, required=False):
  return LegacyOption(spelling, name or spelling.removeprefix("--").replace("-", "_"),
      default, type, required)
# end


_TAG_LABEL = (_option("--tag", default=None), _option("--label", default=None))
LEGACY_MODELS = MappingProxyType({
    "velocity": LegacyModel("five-moment-velocity", (
      _option("--density", "density_tag", "density"),
      _option("--momentum", "momentum_tag", "momentum"), *_TAG_LABEL),
      MappingProxyType({"density_tag": "density", "momentum_tag": "momentum"})),
    "agyro": LegacyModel("ten-moment-agyro", (
      _option("--measure", default="frobenius"),
      _option("--pressure", "pressure_tag", "pressure"),
      _option("--bfield", "bfield_tag", "field"), *_TAG_LABEL),
      MappingProxyType({"pressure_tag": "ptensor", "bfield_tag": "bfield"})),
    "current": LegacyModel("multispecies-accumulate-current", (
      _option("--qbym", default=False, type=bool),
      _option("--charge", default=None, type=float),
      _option("--mass", default=None, type=float),
      _option("--use", default=None), *_TAG_LABEL), MappingProxyType({})),
    "energetics": LegacyModel("multispecies-energetics", (
      _option("--elc", "elc_tag", "elc"), _option("--ion", "ion_tag", "ion"),
      _option("--field", "field_tag", "field"),
      _option("--gas-gamma", default=5.0 / 3.0, type=float),
      _option("--num-moms", default=None, type=int), *_TAG_LABEL),
      MappingProxyType({"elc_tag": "elc", "ion_tag": "ion", "field_tag": "field"})),
    "parrotate": LegacyModel("rotations-parrotate", (
      _option("--array", "array_tag", "array"),
      _option("--rotator", "rotator_tag", "rotator"),
      _option("--coords", default="0:3"), *_TAG_LABEL),
      MappingProxyType({"array_tag": "array", "rotator_tag": "rotator"})),
    "perprotate": LegacyModel("rotations-perprotate", (
      _option("--array", "array_tag", "array"),
      _option("--rotator", "rotator_tag", "rotator"),
      _option("--coords", default="0:3"), *_TAG_LABEL),
      MappingProxyType({"array_tag": "array", "rotator_tag": "rotator"})),
    "bparrotate": LegacyModel("rotations-bparrotate", (
      _option("--array", "array_tag", "array"),
      _option("--field", "field_tag", "field"), *_TAG_LABEL),
      MappingProxyType({"array_tag": "array", "field_tag": "field"})),
    "bperprotate": LegacyModel("rotations-bperprotate", (
      _option("--array", "array_tag", "array"),
      _option("--field", "field_tag", "field"), *_TAG_LABEL),
      MappingProxyType({"array_tag": "array", "field_tag": "field"})),
    "transform_frame": LegacyModel("kinetic-transform-frame", (
      _option("--distribution", "distribution_tag", required=True),
      _option("--bulk", "bulk_tag", required=True),
      _option("--cdim", type=int, required=True), *_TAG_LABEL),
      MappingProxyType({"distribution_tag": "distribution", "bulk_tag": "bulk"})),
    "laguerre_compose": LegacyModel("pkpm-laguerre-compose", (
      _option("--distribution", "distribution_tag", "distribution"),
      _option("--variables", "tm_tag", "variables"), *_TAG_LABEL),
      MappingProxyType({"distribution_tag": "distribution", "tm_tag": "variables"})),
})


def build_legacy_command(name: str) -> click.Command:
  """Build a compatibility adapter that invokes one canonical model."""
  spec = LEGACY_MODELS[name]
  from .commands import MODELS

  target = next(model for model in MODELS if model.name == spec.target)
  params = [click.Option([option.spelling, option.name], default=option.default,
      type=option.type, required=option.required) for option in spec.options]

  @click.pass_context
  def callback(click_context, **legacy_values):
    warn_legacy(click_context, name)
    values = {parameter.name: parameter.default for parameter in target.parameters
        if not parameter.injected}
    for old, value in legacy_values.items():
      values[spec.rename.get(old, old)] = value
    # end
    in_place_legacy = "inplace" in values and legacy_values.get("tag") is None
    if in_place_legacy:
      values["inplace"] = True
    # end
    if target.spec.selectable:
      values.setdefault("use", legacy_values.get("use"))
    # end
    result = execute_model(click_context, target, values)
    if in_place_legacy and click_context.obj.datasets \
        and click_context.obj.datasets[-1] is result:
      click_context.obj.datasets.pop()
    # end
    return result
  # end
  return click.Command(name, params=params, callback=callback,
      help=f"Deprecated compatibility adapter for {spec.target}.")
# end


__all__ = ["LEGACY_MODELS", "LegacyModel", "LegacyOption", "build_legacy_command"]
