"""Load the compiled ``_gpython`` extension -- the single capability switch.

The foreign floor is the CPython extension ``postgkyl.gpython._gpython``, built by
``scripts/build_gpython.sh`` against ``gkyl_gpython.h`` -- the gpython shim, which lives
in the gkeyll repo (``core/zero/gpython.c``) and is compiled INTO
``libg0core.so`` by Gkeyll's own build (GKEYLL_C_SHIM.md). There are no
runtime signature declarations and no struct mirrors here: the contract is
enforced by the C compiler at the producer. The one runtime check left is
the ``GPYTHON_API_VERSION`` handshake, which catches a stale ``_gpython.so``
paired with a newer shim header (or vice versa).

If the extension is missing, :func:`available` returns False and
:func:`require` raises with build guidance; importing postgkyl never fails.

A NumPy ABI mismatch (`_gpython.so` compiled against a different NumPy than
the one installed -- e.g. pip's isolated build environment resolving a
different NumPy than the target environment did) surfaces as a ``ValueError``
from NumPy's own ``import_array()`` check, not an ``ImportError``; it is
caught alongside the missing-extension case for the same reason -- a broken
bridge must degrade to "unavailable", never take down ``import postgkyl``.
"""

from __future__ import annotations

import pathlib

try:
  from . import _gpython as _mod
  if _mod.api_version() != _mod.GPYTHON_API_VERSION:
    raise ImportError(
        f"gpython shim version mismatch: _gpython.so was built for API "
        f"{_mod.api_version()}, postgkyl expects {_mod.GPYTHON_API_VERSION}; "
        "rebuild with scripts/build_gpython.sh")
  _ERROR = None
except (ImportError, ValueError) as exc:
  _mod = None
  _ERROR = (f"{exc}\nBuild the compiled bridge with scripts/build_gkeyll.sh "
            "(or scripts/build_gpython.sh if libg0core.so already exists). "
            "A 'numpy.dtype size changed' error means _gpython.so was "
            "compiled against a different NumPy than the one installed here "
            "-- reinstall with `pip install -e . --no-build-isolation` so "
            "the build step and the installed environment use the same "
            "NumPy, then rebuild the bridge.")


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


def build_info() -> dict[str, str] | None:
  """Metadata about the vendored Gkeyll build this bridge was compiled from.

  None when the bridge has never been built (scripts/build_gkeyll.sh never
  ran): ``_build_info`` is a generated build artifact, not part of the
  source tree (see scripts/build_gpython.sh, .gitignore).
  """
  try:
    from . import _build_info as _bi
  except ImportError:
    return None
  return {
      "gkeyll_commit": _bi.GKEYLL_COMMIT,
      "gkeyll_commit_date": _bi.GKEYLL_COMMIT_DATE,
      "gkeyll_branch": _bi.GKEYLL_BRANCH,
      "postgkyl_build_commit": _bi.POSTGKYL_BUILD_COMMIT,
      "build_date": _bi.BUILD_DATE,
      "build_cc": _bi.BUILD_CC,
      "build_arch_flags": _bi.BUILD_ARCH_FLAGS
  }
