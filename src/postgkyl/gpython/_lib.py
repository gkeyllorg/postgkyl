"""Load the compiled ``_gpython`` extension -- the single capability switch.

The foreign floor is the CPython extension ``postgkyl.gpython._gpython``, built by
setuptools against ``gkyl_gpython_core.h`` -- the gpython shim, which lives
in the gkeyll repo (``core/zero/gpython_core.c``) and is compiled INTO
``libg0core.so`` by Gkeyll's own build (GKEYLL_C_SHIM.md). There are no
runtime signature declarations and no struct mirrors here: the contract is
enforced by the C compiler at the producer. The one runtime check left is
the ``GPYTHON_API_VERSION`` handshake, which catches a stale ``_gpython.so``
paired with a newer shim header (or vice versa).

If the extension is missing, :func:`available` returns False and
:func:`require` raises with build guidance; importing postgkyl never fails.

An incompatible NumPy ABI can raise ``ValueError`` during import, so it is
caught alongside ``ImportError`` and reported by :func:`require`.
"""

from __future__ import annotations

import json
import pathlib

try:
  from . import _gpython as _mod
  if _mod.api_version() != _mod.GPYTHON_API_VERSION:
    raise ImportError(
        f"gpython shim version mismatch: _gpython.so was built for API "
        f"{_mod.api_version()}, postgkyl expects {_mod.GPYTHON_API_VERSION}; "
        "rebuild with python -m pip install -e .")
  _ERROR = None
except (ImportError, ValueError) as exc:
  _mod = None
  _ERROR = (f"{exc}\nBuild the compiled bridge from a Postgkyl checkout with "
            "`python -m pip install -e .`. If NumPy reports an incompatible "
            "ABI, reinstall Postgkyl in the active Python environment.")


def available() -> bool:
  """True when the compiled Gkeyll bridge is loaded (the capability switch)."""
  return _mod is not None


def require():
  """The ``_gpython`` module, or a RuntimeError explaining how to build it."""
  if _mod is None:
    raise RuntimeError(f"postgkyl's Gkeyll bridge is unavailable: {_ERROR}")
  return _mod


def lib_path() -> pathlib.Path | None:
  """Path of the loaded extension (which is rpath-bound to its libg0core)."""
  return pathlib.Path(_mod.__file__) if _mod is not None else None


def build_info() -> dict[str, str | list[str]] | None:
  """Build provenance bundled beside the extension, or None if absent."""
  path = pathlib.Path(__file__).with_name("_build_info.json")
  if not path.is_file():
    return None
  return json.loads(path.read_text())


def module_api_versions() -> dict[str, int]:
  """Call every compiled module's shim and return its checked ABI version."""
  return require().module_api_versions()


def require_module(module: str, function: str):
  """Require a compiled model module and explain how to enable it."""
  lib = require()
  if module not in lib.module_api_versions():
    raise RuntimeError(
        f"Cannot call {function}: this function requires Gkeyll's {module} "
        f"module, but Gkeyll was built without that module. To call this "
        f"function, rebuild Postgkyl with the {module} layer (or higher) "
        f"from a Postgkyl checkout: "
        f"POSTGKYL_GKEYLL_MODULE={module} python -m pip install -e .")
  return lib
