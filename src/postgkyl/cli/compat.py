"""Declarative, temporary translations from the 2.x CLI vocabulary."""

from __future__ import annotations

from types import MappingProxyType

import click

from .compiler import CodecKind, CommandModel


COMMAND_ALIASES = MappingProxyType({
    "agyro": "ten-moment-agyro",
    "bparrotate": "rotations-bparrotate",
    "bperprotate": "rotations-bperprotate",
    "current": "multispecies-accumulate-current",
    "dg_local_poly": "local-poly",
    "energetics": "multispecies-energetics",
    "evalatcoordproj": "eval-at-coord-proj",
    "extractinput": "extract-input",
    "gk_distf": "gyrokinetics-load-gk-distf",
    "gk_energy_balance": "gyrokinetics-gk-energy-balance",
    "gk_fluxsurf": "gk-fluxsurf",
    "gk_load_quantity": "gyrokinetics-load-gk-quantity",
    "gk_particle_balance": "gyrokinetics-gk-particle-balance",
    "gk_rz": "gk-rz",
    "gkyl_pkpm": "pkpm-load-pkpm",
    "laguerre_compose": "pkpm-laguerre-compose",
    "listoutputs": "discovery-find-output-stems",
    "parrotate": "rotations-parrotate",
    "perprotate": "rotations-perprotate",
    "plotly_animate": "plotly-animate",
    "transform_frame": "kinetic-transform-frame",
    "velocity": "five-moment-velocity",
    "pl": "plot",
    "ev": "evaluate",
})

# Option aliases are presentation-only translations. Values still pass
# through the generated option's canonical codec and invocation model.
OPTION_ALIASES = MappingProxyType({
    "load": MappingProxyType({"--basis": "--basis-type"}),
    "save": MappingProxyType({
        "--out": "--out-name", "-o": "--out-name",
        "--format": "--extension", "-f": "--extension",
    }),
    "local-poly": MappingProxyType({
        "--num-points": "--npoints", "-n": "--npoints"}),
    "plot": MappingProxyType({"-m": "--multiblock"}),
    "map": MappingProxyType({"--mapping-file": "--mapping"}),
    "val2coord": MappingProxyType({"-x": "--x", "-y": "--y"}),
    "ten-moment-agyro": MappingProxyType({"--pressure": "--ptensor"}),
    "gk-rz": MappingProxyType({"--zaxis": "--z-axis", "--phi": "--phi-tor"}),
    "gyrokinetics-load-gk-distf": MappingProxyType({
        "-n": "--name", "-s": "--species", "-f": "--frame",
        "--c2p": "--use-c2p-vel", "--block": "--block-idx"}),
    "gyrokinetics-load-gk-quantity": MappingProxyType({
        "-q": "--quantity", "-s": "--species", "-n": "--name",
        "-f": "--frame", "-p": "--path", "--dir": "--direction"}),
    "pkpm-load-pkpm": MappingProxyType({
        "-n": "--name", "-s": "--species", "-i": "--idx",
        "-p": "--poly-order"}),
})

# Old flag-only spellings which lower to an ordinary canonical option/value
# pair.  Keeping the implied value in this table makes the translation fully
# declarative and keeps it out of the execution adapters.
FLAG_OPTIONS = MappingProxyType({
    "plot": MappingProxyType({"--no-show": ("--show", "False")}),
    "animate": MappingProxyType({
        "--no-show": ("--show", "False"),
        "--float": ("--fixed-range", "False"),
    }),
    "plotly": MappingProxyType({"--no-show": ("--show", "False")}),
    "plotly-animate": MappingProxyType({"--no-show": ("--show", "False")}),
    "pyvista": MappingProxyType({
        "--no-show": ("--show", "False"),
        "--no-spin": ("--spin", "False"),
    }),
    "average": MappingProxyType({
        **{f"--z{dimension}": ("--dims", str(dimension))
            for dimension in range(6)},
    }),
})

# Each legacy coordinate flag carried both the direction in its spelling and
# the coordinate in its following value.  Canonically those are two repeated
# API options, so the table records the lossless paired expansion.
PAIRED_OPTIONS = MappingProxyType({
    "eval-at-coord-proj": MappingProxyType({
        **{f"--z{dimension}": ("--eval-dirs", str(dimension),
            "--eval-coords") for dimension in range(6)},
    }),
})

# The old chained parser had four scientific positional values. Canonical
# syntax makes every one a long option.
POSITIONAL_OPTIONS = MappingProxyType({
    "fit": "--fit-type",
    "evaluate": "--chain",
    "map": "--mapping",
    "integrate-axis": "--axis",
})


def warn_legacy(ctx, spelling: str) -> None:
  """Emit at most one deprecation warning for a command invocation."""
  root = ctx.find_root()
  if root.meta.get("postgkyl_legacy_warning"):
    return
  # end
  root.meta["postgkyl_legacy_warning"] = True
  click.echo(
      f"Deprecation warning: legacy CLI spelling {spelling!r}; use canonical "
      "long-option syntax.", err=True)
# end


def _takes_value(options: set[str], token: str) -> bool:
  return token in options or any(token.startswith(option + "=") for option in options)
# end


def rewrite_argv(model: CommandModel, ctx, argv: list[str]) -> list[str]:
  """Translate one legacy argv fragment to canonical generated options."""
  aliases = OPTION_ALIASES.get(model.name, {})
  flag_options = FLAG_OPTIONS.get(model.name, {})
  paired_options = PAIRED_OPTIONS.get(model.name, {})
  boolean_options = {
      "--" + parameter.name.replace("_", "-")
      for parameter in model.parameters
      if not parameter.injected and parameter.codec is not None
      and parameter.codec.kind is CodecKind.BOOLEAN
  }
  known_options = {
      "--" + parameter.name.replace("_", "-")
      for parameter in model.parameters if not parameter.injected
  } | ({"--use"} if model.spec.selectable else set())

  rewritten: list[str] = []
  changed = False
  i = 0
  while i < len(argv):
    token = argv[i]
    if token in flag_options:
      rewritten.extend(flag_options[token])
      changed = True
      i += 1
      continue
    # end
    if token in paired_options and i + 1 < len(argv):
      rewritten.extend((*paired_options[token], argv[i + 1]))
      changed = True
      i += 2
      continue
    # end
    replacement = aliases.get(token, token)
    changed |= replacement != token
    rewritten.append(replacement)
    if replacement in boolean_options and (
        i + 1 == len(argv) or argv[i + 1].startswith("--")
        or argv[i + 1] not in ("true", "false", "True", "False", "1", "0")):
      rewritten.append("True")
      changed = True
    # end
    i += 1
  # end

  positional = POSITIONAL_OPTIONS.get(model.name)
  if positional and positional not in rewritten:
    # Options and their values are left untouched; the first free token is
    # the legacy positional value. This is deliberately bounded to the four
    # historical forms above.
    value_indices: set[int] = set()
    for index, token in enumerate(rewritten[:-1]):
      if _takes_value(known_options, token) and "=" not in token:
        value_indices.add(index + 1)
      # end
    # end
    free = next((index for index, token in enumerate(rewritten)
        if not token.startswith("-") and index not in value_indices), None)
    if free is not None:
      value = rewritten.pop(free)
      rewritten[free:free] = [positional, value]
      changed = True
    # end
  # end

  if model.spec.execution.name == "LOAD" and model.name == "load" \
      and "--file-name" not in rewritten and ctx.obj.in_data_strings:
    rewritten[:0] = ["--file-name", ctx.obj.in_data_strings.pop(0)]
  # end
  if model.name == "load" and "--basis-type" in rewritten:
    index = rewritten.index("--basis-type")
    codes = {
        "ms": ("serendipity", "modal"), "ns": ("serendipity", "nodal"),
        "mo": ("maximal-order", "modal"), "mt": ("tensor", "modal"),
        "gkhyb": ("gkhybrid", "modal"), "pkpmhyb": ("hybrid", "modal"),
    }
    if index + 1 < len(rewritten) and rewritten[index + 1] in codes:
      basis, value_form = codes[rewritten[index + 1]]
      rewritten[index + 1] = basis
      if "--value-form" not in rewritten:
        rewritten.extend(["--value-form", value_form])
      # end
      changed = True
    # end
  # end
  if changed:
    warn_legacy(ctx, model.name)
  # end
  return rewritten
# end


__all__ = [
    "COMMAND_ALIASES", "FLAG_OPTIONS", "OPTION_ALIASES", "PAIRED_OPTIONS",
    "POSITIONAL_OPTIONS", "rewrite_argv", "warn_legacy",
]
