"""Deterministic discovery of command metadata from public API roots."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from types import ModuleType

import postgkyl
from postgkyl.command_spec import command_spec, hidden_spec


class SurfaceClassificationError(ValueError):
  """A public CLI candidate is neither exposed nor explicitly hidden."""
# end


@dataclass(frozen=True)
class DiscoveredCallable:
  name: str
  callable: object
  public_path: str
# end


def _classify(obj, path: str) -> bool:
  exposed = command_spec(obj) is not None
  excluded = hidden_spec(obj) is not None
  if exposed == excluded:
    state = "both exposed and hidden" if exposed else "unclassified"
    raise SurfaceClassificationError(f"{path}: public callable is {state}")
  # end
  return exposed
# end


def _functions(module: ModuleType):
  public = getattr(module, "__all__", None)
  names = public if public is not None else sorted(
      name for name, value in vars(module).items()
      if not name.startswith("_") and inspect.isfunction(value)
      and value.__module__ == module.__name__)
  for name in names:
    value = getattr(module, name)
    if inspect.isfunction(value):
      yield name, value
    # end
  # end
# end


def _diagnostic_modules(root: ModuleType):
  seen: set[int] = set()

  def visit(module):
    if id(module) in seen:
      return
    # end
    seen.add(id(module))
    yield module
    for name in getattr(module, "__all__", ()):
      value = getattr(module, name)
      if isinstance(value, ModuleType) and value.__name__.startswith(
          "postgkyl.diagnostics"):
        yield from visit(value)
      # end
    # end
  # end
  yield from visit(root)
# end


def discover_public_surface(facade=postgkyl) -> tuple[DiscoveredCallable, ...]:
  """Walk the declared public roots and return all exposed callables."""
  found: list[DiscoveredCallable] = []
  classified: set[tuple[int, str]] = set()

  def consider(obj, path: str, name: str) -> None:
    key = (id(obj), name)
    if key in classified:
      return
    # end
    classified.add(key)
    if _classify(obj, path):
      found.append(DiscoveredCallable(name, obj, path))
    # end
  # end

  # Fluent methods are the authoritative inventory of per-dataset commands.
  for cls_name in ("GData", "GDataGroup"):
    cls = getattr(facade, cls_name)
    for name, value in sorted(cls.__dict__.items()):
      if name.startswith("_") or not callable(value):
        continue
      # end
      consider(value, f"{cls.__module__}.{cls.__qualname__}.{name}",
          name.replace("_", "-"))
    # end
  # end

  # Public facade functions not already reached through their fluent view.
  for name in facade.__all__:
    value = getattr(facade, name)
    if inspect.isfunction(value):
      path = f"postgkyl.{name}"
      if value.__module__.startswith("postgkyl.diagnostics"):
        _classify(value, path)
      # end
      else:
        consider(value, path, name.replace("_", "-"))
      # end
    # end
  # end

  diagnostics = facade.diagnostics
  for module in _diagnostic_modules(diagnostics):
    if module is diagnostics:
      continue
    # end
    relative = module.__name__.removeprefix("postgkyl.diagnostics.")
    namespace = relative.split(".", 1)[0].replace("_", "-")
    for name, value in _functions(module):
      if not value.__module__.startswith("postgkyl.diagnostics"):
        # Compatibility re-exports owned by a lower layer keep that owner's
        # canonical command (for example operations.gyrokinetics.gk_rz ->
        # ``gk-rz``); the diagnostic alias is classified but does not invent
        # a second command or move it into the wrong help section.
        _classify(value, f"{module.__name__}.{name}")
        continue
      # end
      consider(value, f"{module.__name__}.{name}",
          f"{namespace}-{name.replace('_', '-')}")
    # end

    # Registry values are public vocabulary roots too. Identity de-duplication
    # means aliases do not invent additional command names.
    variables = getattr(module, "VARIABLES", None)
    if isinstance(variables, dict):
      for value in variables.values():
        if inspect.isfunction(value):
          consider(value, f"{module.__name__}.VARIABLES[{value.__name__!r}]",
              f"{namespace}-{value.__name__.replace('_', '-')}")
        # end
      # end
    # end
  # end

  # A fluent view and facade function may be aliases of the same operation.
  # Keep one deterministic view.
  unique: dict[tuple[str, int], DiscoveredCallable] = {}
  for item in found:
    key = (item.name, id(item.callable))
    unique.setdefault(key, item)
  # end
  return tuple(sorted(unique.values(), key=lambda item: (item.name, item.public_path)))
# end


__all__ = ["DiscoveredCallable", "SurfaceClassificationError", "discover_public_surface"]
