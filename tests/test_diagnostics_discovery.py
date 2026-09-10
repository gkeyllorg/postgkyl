"""Tests for postgkyl.diagnostics.discovery -- the equation-blind
output-stem/frame discovery shared by every equation loader.

No dedicated ``find_output_stems``/``.outputs()`` tests exist in
``tests_bak`` (``tests_bak/test_loader.py`` tests the ``pg.load``
callable/namespace instead -- see ``test_diagnostics_gk_load.py``'s
``TestResolveFrames`` for the pieces of that file that do belong to this
layer), so this is a fresh corpus targeting ``find_output_stems`` and the new
``available_frames`` helper directly.
"""

from __future__ import annotations

from postgkyl.diagnostics import discovery


def _touch(tmp_path, *names):
  for name in names:
    (tmp_path / name).touch()


class TestFindOutputStems:

  def test_single_extension_single_stem(self, tmp_path):
    _touch(tmp_path, "elc_M0_0.gkyl", "elc_M0_1.gkyl", "elc_M0_2.gkyl")
    out = discovery.find_output_stems("gkyl", str(tmp_path))
    assert out == {"gkyl": ["elc_M0"]}

  def test_multiple_stems_sorted(self, tmp_path):
    _touch(tmp_path, "ion_M0_0.gkyl", "elc_M0_0.gkyl", "field_0.gkyl")
    out = discovery.find_output_stems("gkyl", str(tmp_path))
    assert out["gkyl"] == ["elc_M0", "field", "ion_M0"]

  def test_multiple_extensions(self, tmp_path):
    _touch(tmp_path, "elc_M0_0.gkyl", "elc_M0_0.h5")
    out = discovery.find_output_stems("h5,gkyl", str(tmp_path))
    assert out == {"h5": ["elc_M0"], "gkyl": ["elc_M0"]}

  def test_strips_restart_suffix(self, tmp_path):
    _touch(tmp_path, "elc_M0_0_restart.gkyl")
    out = discovery.find_output_stems("gkyl", str(tmp_path))
    assert out["gkyl"] == ["elc_M0"]

  def test_no_frame_number_kept_as_is(self, tmp_path):
    _touch(tmp_path, "geo_int_jacobtot_inv.gkyl")
    out = discovery.find_output_stems("gkyl", str(tmp_path))
    assert out["gkyl"] == ["geo_int_jacobtot_inv"]

  def test_empty_directory(self, tmp_path):
    out = discovery.find_output_stems("gkyl", str(tmp_path))
    assert out == {"gkyl": []}

  def test_default_extensions_and_path(self, tmp_path, monkeypatch):
    _touch(tmp_path, "a_0.gkyl")
    monkeypatch.chdir(tmp_path)
    out = discovery.find_output_stems()
    assert out == {"gkyl": ["a"]}


class TestAvailableFrames:

  def test_discovers_all_frames(self, tmp_path):
    stem = str(tmp_path / "sim-ion_M0_")
    _touch(tmp_path, "sim-ion_M0_0.gkyl", "sim-ion_M0_1.gkyl",
           "sim-ion_M0_5.gkyl")
    assert discovery.available_frames(stem) == {0, 1, 5}

  def test_restricted_to_candidate_frames(self, tmp_path):
    stem = str(tmp_path / "sim-ion_M0_")
    _touch(tmp_path, "sim-ion_M0_0.gkyl", "sim-ion_M0_1.gkyl",
           "sim-ion_M0_5.gkyl")
    assert discovery.available_frames(stem, frames=[0, 5, 99]) == {0, 5}

  def test_no_matching_files(self, tmp_path):
    stem = str(tmp_path / "sim-ion_M0_")
    assert discovery.available_frames(stem) == set()

  def test_non_numeric_suffix_ignored(self, tmp_path):
    stem = str(tmp_path / "sim-ion_M0_")
    _touch(tmp_path, "sim-ion_M0_0.gkyl", "sim-ion_M0_restart.gkyl")
    assert discovery.available_frames(stem) == {0}
