"""Shared pytest configuration for the postgkyl test suite.

No-GUI guarantee
-----------------
This test session must never put a window or a browser tab on the desktop --
doing so can crash/hang the sandboxed environment this suite runs in. Two
independent guards enforce that, both applied before any test module (or its
imports) can run:

- ``matplotlib.use("Agg")`` is forced at import time, below, before anything
  else gets a chance to trigger Matplotlib's own backend auto-detection
  (which, given a display, could pick an interactive GUI backend). Agg is a
  pure-raster, no-window backend, so ``plt.show()`` is always a no-op under it.
- ``_block_gui_popups`` (autouse, session-scoped) monkeypatches
  ``webbrowser.open`` -- the mechanism ``render.plotly``'s ``open_preview``
  (default-off; see its docstring) uses to show a Plotly figure -- and, if
  PyVista is installed, ``pyvista.Plotter.show`` -- the analogous mechanism
  for a PyVista render window (see ``render.pyvista.pyvista``'s ``no_show``
  parameter). Plotly defaults ``show`` to ``False``; interactive-by-default
  PyVista calls in tests pass ``no_show=True``. This fixture is the backstop
  for any call that forgets to select its renderer's headless setting.

Test data generation
--------------------
``pytest_configure`` writes synthetic .gkyl files to
``tests/test_data/generated/`` (gitignored -- every test that reads from
that directory depends on this running first). It is a hook, not a
session-scoped autouse fixture, specifically so it runs exactly once in the
true parent process before collection or any forking begins -- see its own
comment for why a fixture is the wrong tool once macOS CI's --forked is in
play. Without it, a clean checkout (e.g. CI) has no fixtures to read; only a
machine where someone has run ``python tests/generate_test_data.py`` (or a
prior pytest session) before would happen to have them already on disk.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from generate_test_data import generate_all

GEN_DIR = Path(__file__).parent / "test_data" / "generated"

# macOS-only escape hatch: ``postgkyl``'s facade (__init__.py -> render ->
# render.pyvista) unconditionally imports PyVista's VTK bindings, so every
# test process here loads the full native VTK stack regardless of whether
# any PyVista test runs. On macOS CI (observed on Python 3.10, 3.11, and
# 3.12 alike -- not Python-version-specific, so not the intermittent
# matplotlib font-rendering issue tracked elsewhere in this suite) VTK's own
# C++ global/static destructors reproducibly SIGSEGV during CPython's
# interpreter finalization, always *after* pytest has already run every
# test and printed its full (passing) summary. Nothing of value happens in
# that teardown window, so once pytest has reported its result, exit the
# process immediately via ``os._exit`` -- bypassing the interpreter
# finalization that walks into VTK's broken destructors -- instead of
# letting a native-library crash overwrite an already-successful result
# with a misleading CI failure.
_exit_status: list[int] = []


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
  _exit_status.append(int(exitstatus))


def pytest_unconfigure(config: pytest.Config) -> None:
  if sys.platform == "darwin" and _exit_status:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exit_status[0])


@pytest.fixture(scope="session", autouse=True)
def _block_gui_popups():
  """Backstop: no test may open a browser tab or a native render window."""
  import webbrowser

  def _no_browser(*_args, **_kwargs):
    raise AssertionError(
        "webbrowser.open() was called during tests -- a figure/preview would "
        "have popped up on the desktop. Pass show=False for render.plotly or "
        "no_show=True for render.pyvista, or mock the call being tested.")

  webbrowser.open = _no_browser
  webbrowser.open_new = _no_browser
  webbrowser.open_new_tab = _no_browser

  try:
    import pyvista

    def _no_plotter_show(*_args, **_kwargs):
      raise AssertionError(
          "pyvista.Plotter.show() was called during tests -- a render window "
          "would have popped up on the desktop. Pass no_show=True or mock "
          "the call being tested.")

    pyvista.Plotter.show = _no_plotter_show
  except ImportError:
    pass


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
  """Release every pyplot-managed figure after each test."""
  yield

  import matplotlib.pyplot as plt
  plt.close("all")


def pytest_configure(config: pytest.Config) -> None:
  # A session-scoped autouse *fixture* only actually runs on first request,
  # which lands inside whichever test forks first under macOS CI's --forked
  # (see test.yml) -- pytest's "already cached" bookkeeping then lives in
  # that child's forked copy of the session, never propagating back to the
  # parent, so every subsequent forked test re-triggers it too (measured:
  # 40 re-generations for 40 tests instead of one). pytest_configure runs
  # exactly once, in the true parent process, before collection or any
  # forking begins, so this is immune to that regardless of --forked.
  generate_all(GEN_DIR)
