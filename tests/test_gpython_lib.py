"""Tests for ``postgkyl.gpython._lib`` -- the capability-switch handshake.

Run:  PYTHONPATH=src pytest tests/test_gpython_lib.py -v
"""

import importlib.util
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)  # dedup harmless across the shared test session

from postgkyl import gpython  # noqa: E402
from postgkyl.gpython import _lib  # noqa: E402

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


@needs_gkeyll
def test_available_true_when_extension_loaded():
  assert _lib.available() is True


@needs_gkeyll
def test_require_returns_the_extension_module():
  mod = _lib.require()
  assert mod is sys.modules["postgkyl.gpython._gpython"]


@needs_gkeyll
def test_lib_path_points_at_the_loaded_extension():
  p = _lib.lib_path()
  assert p is not None
  assert p.name.startswith("_gpython")
  assert p.exists()


@needs_gkeyll
def test_handshake_version_matches():
  g0 = _lib.require()
  assert g0.api_version() == g0.GPYTHON_API_VERSION


def test_available_false_when_extension_absent(monkeypatch):
  """Simulate a no-library install by monkeypatching the module attributes
  (the pattern the layer instructions call out explicitly) rather than
  reloading the real module in place -- `monkeypatch` guarantees the original
  ``_mod``/``_ERROR`` are restored even if an assertion below fails, so this
  can never leak a broken capability switch into the rest of the suite."""
  monkeypatch.setattr(_lib, "_mod", None)
  monkeypatch.setattr(_lib, "_ERROR", "simulated: no _gpython.so found")
  assert _lib.available() is False
  with pytest.raises(RuntimeError, match="simulated: no _gpython.so found"):
    _lib.require()
  assert _lib.lib_path() is None


def _exec_independent_lib_copy():
  """Execute a fresh, independent copy of _lib.py's module code.

  Distinct from `postgkyl.gpython._lib` (a different module object entirely) so
  mutating its state can never affect `postgkyl.gpython.available`/`require`,
  which are bound to the real module's original functions. Its relative
  `from . import _gpython` still resolves against the real `postgkyl.gpython`
  package, which the caller controls via `sys.modules['postgkyl.gpython._gpython']`
  for the duration of the call.
  """
  spec = importlib.util.spec_from_file_location(
      "postgkyl.gpython._lib_independent_copy", _lib.__file__)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


class _patched_gpython:
  """Context manager that makes `from . import _gpython` see `replacement`.

  `from package import submodule` tries `getattr(package, submodule)`
  BEFORE consulting `sys.modules`, and the real `postgkyl.gpython` package
  object already carries a `_gpython` attribute (set as a side effect of the
  real import at process start) -- so patching `sys.modules` alone is not
  enough. Both are patched here and restored unconditionally.
  """

  def __init__(self, replacement):
    self._replacement = replacement

  def __enter__(self):
    self._pkg = sys.modules["postgkyl.gpython"]
    self._had_attr = hasattr(self._pkg, "_gpython")
    self._old_attr = getattr(self._pkg, "_gpython", None)
    self._old_sys_mod = sys.modules.get("postgkyl.gpython._gpython")
    if self._had_attr:
      delattr(self._pkg, "_gpython")
    sys.modules["postgkyl.gpython._gpython"] = self._replacement

  def __exit__(self, *exc):
    if self._had_attr:
      setattr(self._pkg, "_gpython", self._old_attr)
    if self._old_sys_mod is not None:
      sys.modules["postgkyl.gpython._gpython"] = self._old_sys_mod
    else:
      del sys.modules["postgkyl.gpython._gpython"]
    return False


def test_import_error_when_extension_missing():
  """The actual `try: from . import _gpython / except ImportError` branch."""
  with _patched_gpython(None):  # sentinel: forces ImportError
    copy = _exec_independent_lib_copy()

  assert copy.available() is False
  with pytest.raises(RuntimeError, match="Build the compiled bridge"):
    copy.require()
  assert copy.lib_path() is None
  # The real package's bindings must be entirely unaffected by the above.
  assert gpython.available() is True
  assert isinstance(gpython.require(), types.ModuleType)


@needs_gkeyll
def test_patched_gpython_cleans_up_sys_modules_when_never_previously_imported():
  """``_patched_gpython.__exit__``'s cleanup has two cases: restore whatever was
  in ``sys.modules`` before (exercised by every other test here, since the
  real ``_gpython`` is always already imported in this environment), or delete
  the key entirely when there was nothing to restore. Simulate the latter by
  removing the real module first and restoring it manually afterward."""
  real = sys.modules.pop("postgkyl.gpython._gpython")
  try:
    with _patched_gpython(types.SimpleNamespace()):
      assert "postgkyl.gpython._gpython" in sys.modules
    assert "postgkyl.gpython._gpython" not in sys.modules
  finally:
    sys.modules["postgkyl.gpython._gpython"] = real


@needs_gkeyll
def test_version_mismatch_degrades_like_missing():
  """A stale `_gpython.so` (wrong GPYTHON_API_VERSION) must degrade the same way."""
  real = sys.modules["postgkyl.gpython._gpython"]
  fake = types.SimpleNamespace(
      api_version=lambda: real.GPYTHON_API_VERSION + 1000,
      GPYTHON_API_VERSION=real.GPYTHON_API_VERSION)
  with _patched_gpython(fake):
    copy = _exec_independent_lib_copy()

  assert copy.available() is False
  with pytest.raises(RuntimeError, match="version mismatch"):
    copy.require()
  # Unaffected real bindings.
  assert gpython.available() is True
  assert gpython.require() is real
