"""Tests for ``postgkyl.diagnostics.compare`` -- the equation-blind
baseline-vs-candidate simulation comparison diagnostic.

Dynvector-only scenarios (ranking, selection, multiblock overlay, batch
mode) are built from real on-disk ``.gkyl`` dynvector files via
``generate_test_data.write_gkyl_dynvector``, so they exercise the actual
``pg.load``/``plot`` path without needing the compiled Gkeyll bridge.
Scenarios that also compare a spatial field (``interpolate()``) are marked
``needs_gkeyll`` and use ``write_gkyl_field``.

Run: PYTHONPATH=src pytest tests/test_diagnostics_compare.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from postgkyl import gpython
from postgkyl.diagnostics import compare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "generate_test_data", os.path.join(ROOT, "tests", "generate_test_data.py"))
generate_test_data = importlib.util.module_from_spec(spec)
sys.modules.setdefault("generate_test_data", generate_test_data)
spec.loader.exec_module(generate_test_data)

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


def _write_dynvec(path, name, quantity, time, values):
  generate_test_data.write_gkyl_dynvector(
      os.path.join(path, f"{name}-{quantity}.gkyl"), np.asarray(time),
      np.asarray(values))


def _write_field(path, name, quantity, cells, values, *, poly_order=1,
                 basis_type="serendipity", frame=0):
  generate_test_data.write_gkyl_field(
      os.path.join(path, f"{name}-{quantity}_{frame}.gkyl"), cells,
      [0.0] * len(cells), [1.0] * len(cells), values, poly_order, basis_type,
      frame=frame)


def _sim(tmp_path, subdir, name="sim", *, n=5, energy_scale=1.0,
        moms_scale=1.0):
  """A minimal two-quantity dynvector-only sim: field_energy and
  elc_integrated_moms, both present under both sims by default."""
  path = str(tmp_path / subdir)
  os.makedirs(path, exist_ok=True)
  time = np.linspace(0.0, 1.0, n)
  _write_dynvec(path, name, "field_energy", time,
               (np.linspace(1.0, 2.0, n) * energy_scale)[:, None])
  moms = np.stack([
      np.linspace(1.0, 2.0, n), np.linspace(2.0, 3.0, n),
      np.linspace(3.0, 4.0, n)
  ], axis=-1) * moms_scale
  _write_dynvec(path, name, "elc_integrated_moms", time, moms)
  return path


def _only(reports):
  """The single :class:`CompareSimsReport` out of a non-batch ``sims()``
  call's one-entry dict."""
  assert len(reports) == 1, reports
  return next(iter(reports.values()))


def _pdf_page_count(path) -> int:
  """Count pages in a PDF by scanning its raw bytes for ``/Type /Page``
  (excluding ``/Type /Pages``) -- neither ``pypdf`` nor ``pdfminer`` is
  installed in this environment, but this is enough to verify page counts
  against a real ``matplotlib.backends.backend_pdf.PdfPages`` file."""
  import re
  with open(path, "rb") as f:
    data = f.read()
  return len(re.findall(rb"/Type\s*/Page(?!s)", data))


class TestDefaultLabels:

  def test_shares_no_suffix_uses_basenames(self, tmp_path):
    labels = compare._default_labels(str(tmp_path / "a"), str(tmp_path / "b"))
    assert labels == ["a", "b"]

  def test_identical_basenames_use_parent_to_disambiguate(self, tmp_path):
    labels = compare._default_labels(
        str(tmp_path / "baseline" / "sim"), str(tmp_path / "candidate" / "sim"))
    assert labels == [os.path.join("baseline", "sim"),
                      os.path.join("candidate", "sim")]


class TestIsExcluded:

  def test_nothing_excluded_by_default(self):
    assert compare._DEFAULT_EXCLUDE_PREFIXES == ()
    assert not compare._is_excluded("geo_int_bmag",
                                    compare._DEFAULT_EXCLUDE_PREFIXES)
    assert not compare._is_excluded("bc_zlower_vcutsq",
                                    compare._DEFAULT_EXCLUDE_PREFIXES)
    assert not compare._is_excluded("bflux_xlower_HamiltonianMoments",
                                    compare._DEFAULT_EXCLUDE_PREFIXES)
    assert not compare._is_excluded("field_energy",
                                    compare._DEFAULT_EXCLUDE_PREFIXES)

  def test_extra_prefix(self):
    prefixes = compare._DEFAULT_EXCLUDE_PREFIXES + ("dt",)
    assert compare._is_excluded("dt", prefixes)


class TestStemsByQuantity:

  def test_maps_quantity_to_one_stem(self, tmp_path):
    path = _sim(tmp_path, "s")
    quantities = compare._stems_by_quantity(path)
    assert quantities["field_energy"] == "sim-field_energy"
    assert quantities["elc_integrated_moms"] == "sim-elc_integrated_moms"

  def test_multiblock_stem_keeps_one_block(self, tmp_path):
    path = str(tmp_path)
    time = np.linspace(0.0, 1.0, 3)
    for block in range(3):
      _write_dynvec(path, f"sim_b{block}", "field_energy", time,
                   np.ones((3, 1)))
    quantities = compare._stems_by_quantity(path)
    assert quantities["field_energy"] == "sim_b0-field_energy"


class TestSimsDynvectorOnly:

  def test_plots_every_shared_quantity_within_num_plots(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=5))
    assert set(report.plotted) == {"field_energy", "elc_integrated_moms"}
    assert report.only_in_0 == ()
    assert report.only_in_1 == ()
    assert report.skipped == {}
    for quantity in report.plotted:
      assert os.path.isfile(
          os.path.join(str(tmp_path / "figs"), f"{quantity}.png"))

  def test_ranks_larger_difference_first(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", energy_scale=1.001, moms_scale=2.0)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=5))
    by_quantity = {d.quantity: d for d in report.compared}
    assert (by_quantity["elc_integrated_moms"].max_rel_diff
           > by_quantity["field_energy"].max_rel_diff)
    assert report.compared[0].quantity == "elc_integrated_moms"

  def test_num_plots_limits_auto_selection(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1))
    assert report.plotted == ("elc_integrated_moms",)

  def test_must_include_is_always_plotted(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1, must_include=["field_energy"]))
    assert "field_energy" in report.plotted
    assert len(report.plotted) == 1

  def test_must_include_missing_from_one_sim_raises(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate")
    with pytest.raises(ValueError, match="must_include"):
      compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                  must_include=["not_a_real_quantity"])

  def test_only_in_one_sim_reported_not_errored(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate")
    _write_dynvec(path0, "sim", "baseline_only", np.linspace(0.0, 1.0, 3),
                 np.ones((3, 1)))
    report = _only(compare.sims(path0, path1, out_dir=str(tmp_path / "figs")))
    assert report.only_in_0 == ("baseline_only",)
    assert report.only_in_1 == ()

  def test_shape_mismatch_is_skipped_unless_must_include(self, tmp_path):
    path0 = _sim(tmp_path, "baseline", n=5)
    path1 = _sim(tmp_path, "candidate", n=4)
    report = _only(compare.sims(path0, path1, out_dir=str(tmp_path / "figs")))
    assert "field_energy" in report.skipped
    assert "field_energy" not in report.plotted

    with pytest.raises(ValueError, match="shape"):
      compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                  must_include=["field_energy"])

  def test_exclude_skips_matching_prefix(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    exclude=["elc_"]))
    assert "elc_integrated_moms" not in [d.quantity for d in report.compared]

  def test_custom_labels_used_in_legend(self, tmp_path):
    # No assertion on rendered pixels; only that a custom label list doesn't
    # raise and is accepted end-to-end.
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    labels=["before", "after"]))
    assert report.plotted

  def test_format_controls_saved_extension(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    format="jpg"))
    for quantity in report.plotted:
      assert os.path.isfile(
          os.path.join(str(tmp_path / "figs"), f"{quantity}.jpg"))


class TestSimsMultiblockDynvector:

  def _multiblock_sim(self, tmp_path, subdir, scale=1.0, num_blocks=3):
    path = str(tmp_path / subdir)
    os.makedirs(path, exist_ok=True)
    time = np.linspace(0.0, 1.0, 4)
    for block in range(num_blocks):
      _write_dynvec(path, f"sim_b{block}", "field_energy", time,
                   (np.linspace(1.0, 2.0, 4) * scale * (block + 1))[:, None])
    return path

  def test_plots_one_file_per_block(self, tmp_path):
    path0 = self._multiblock_sim(tmp_path, "baseline")
    path1 = self._multiblock_sim(tmp_path, "candidate", scale=1.2)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1))
    assert report.plotted == ("field_energy",)
    pngs = sorted(
        f for f in os.listdir(str(tmp_path / "figs")) if f.endswith(".png"))
    assert pngs == [
        "field_energy_b0.png", "field_energy_b1.png", "field_energy_b2.png"
    ]


@needs_gkeyll
class TestSimsField:

  def _field_sim(self, tmp_path, subdir, offset=0.0):
    path = str(tmp_path / subdir)
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    values = rng.standard_normal((4, 4, 4)) + offset
    _write_field(path, "sim", "field", [4, 4], values, frame=0)
    return path

  def test_2d_field_plots_side_by_side(self, tmp_path):
    path0 = self._field_sim(tmp_path, "baseline")
    path1 = self._field_sim(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1, must_include=["field"]))
    assert report.plotted == ("field",)
    assert os.path.isfile(os.path.join(str(tmp_path / "figs"), "field.png"))

  def _field_sim_3d(self, tmp_path, subdir, offset=0.0):
    path = str(tmp_path / subdir)
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    # serendipity p1 in 3D packs 8 basis coefficients per cell.
    values = rng.standard_normal((4, 4, 4, 8)) + offset
    _write_field(path, "sim", "field", [4, 4, 4], values, frame=0)
    return path

  def test_3d_field_defaults_to_y_midpoint(self, tmp_path):
    path0 = self._field_sim_3d(tmp_path, "baseline")
    path1 = self._field_sim_3d(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1, must_include=["field"]))
    assert report.plotted == ("field",)
    assert os.path.isfile(os.path.join(str(tmp_path / "figs"), "field.png"))

  def test_3d_field_honors_explicit_z1(self, tmp_path):
    path0 = self._field_sim_3d(tmp_path, "baseline")
    path1 = self._field_sim_3d(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1, must_include=["field"], z1="0"))
    assert report.plotted == ("field",)

  def _field_sim_4d(self, tmp_path, subdir, offset=0.0):
    path = str(tmp_path / subdir)
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    # serendipity p1 in 4D packs 16 basis coefficients per cell; a 4-D
    # spatial field still has 3 dimensions left after the y-midpoint
    # default, so it needs further selectors to plot.
    values = rng.standard_normal((2, 2, 2, 2, 16)) + offset
    _write_field(path, "sim", "field", [2, 2, 2, 2], values, frame=0)
    return path

  def test_4d_field_is_unplottable_without_more_selectors(self, tmp_path):
    path0 = self._field_sim_4d(tmp_path, "baseline")
    path1 = self._field_sim_4d(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=5))
    assert "field" not in report.plotted
    by_quantity = {d.quantity: d for d in report.compared}
    assert by_quantity["field"].plottable is False

  def test_4d_field_as_must_include_raises(self, tmp_path):
    path0 = self._field_sim_4d(tmp_path, "baseline")
    path1 = self._field_sim_4d(tmp_path, "candidate", offset=0.5)
    with pytest.raises(ValueError):
      compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                  must_include=["field"])

  def test_4d_field_plots_with_explicit_selectors(self, tmp_path):
    path0 = self._field_sim_4d(tmp_path, "baseline")
    path1 = self._field_sim_4d(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    must_include=["field"], z1="0", z2="0"))
    assert report.plotted == ("field",)

  def _multiblock_field_sim(self, tmp_path, subdir, num_blocks=3, offset=0.0):
    path = str(tmp_path / subdir)
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    for block in range(num_blocks):
      values = rng.standard_normal((4, 4, 4)) + offset
      _write_field(path, f"sim_b{block}", "field", [4, 4], values, frame=0)
    return path

  def test_multiblock_field_contributes_one_page_per_block(self, tmp_path):
    path0 = self._multiblock_field_sim(tmp_path, "baseline")
    path1 = self._multiblock_field_sim(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, single_pdf=True, num_plots=1,
                    must_include=["field"], out_dir=str(tmp_path / "out")))
    assert report.plotted == ("field",)
    pdf_path = os.path.join(str(tmp_path / "out"),
                            "postgkyl_compare_sims.pdf")
    # Title page + one page per block (3 blocks), no per-quantity image files.
    assert _pdf_page_count(pdf_path) == 1 + 3


class TestSimsBatch:

  def test_compares_every_shared_subdirectory(self, tmp_path):
    base = tmp_path / "baseline"
    cand = tmp_path / "candidate"
    for sim_name in ("sim_a", "sim_b"):
      _sim(base.parent, base.name + f"/{sim_name}", name=sim_name)
      _sim(cand.parent, cand.name + f"/{sim_name}", name=sim_name,
          moms_scale=1.5)
    reports = compare.sims(str(base), str(cand), batch=True,
                           out_dir=str(tmp_path / "figs"))
    assert set(reports) == {"sim_a", "sim_b"}
    for name, report in reports.items():
      assert report.plotted
      assert os.path.isdir(os.path.join(str(tmp_path / "figs"), name))

  def test_raises_when_no_subdirectory_is_shared(self, tmp_path):
    base = tmp_path / "baseline"
    cand = tmp_path / "candidate"
    _sim(base.parent, base.name + "/sim_a", name="sim_a")
    _sim(cand.parent, cand.name + "/sim_b", name="sim_b")
    with pytest.raises(ValueError, match="no simulation subdirectory"):
      compare.sims(str(base), str(cand), batch=True,
                  out_dir=str(tmp_path / "figs"))

  def test_single_sim_dict_key_is_path0_basename(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    reports = compare.sims(path0, path1, out_dir=str(tmp_path / "figs"))
    assert set(reports) == {"baseline"}


class TestOutDirDefault:

  def test_out_dir_omitted_uses_cwd_subfolder(self, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(compare.sims(path0, path1))
    expected = os.path.join(str(tmp_path), "postgkyl_compare_sims")
    assert report.out_dir == expected
    assert os.path.isfile(os.path.join(expected, "field_energy.png"))

  def test_out_dir_given_is_honored(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "custom")))
    assert report.out_dir == str(tmp_path / "custom")


class TestSinglePdf:

  def test_defaults_to_cwd_filename(self, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(compare.sims(path0, path1, single_pdf=True))
    pdf_path = os.path.join(str(tmp_path), "postgkyl_compare_sims.pdf")
    assert os.path.isfile(pdf_path)
    # Title page + one page per plotted quantity; no individual image files.
    assert _pdf_page_count(pdf_path) == 1 + len(report.plotted)
    assert not os.path.isdir(
        os.path.join(str(tmp_path), "postgkyl_compare_sims"))

  def test_out_dir_given_no_pdf_name(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    compare.sims(path0, path1, single_pdf=True,
                out_dir=str(tmp_path / "out"))
    assert os.path.isfile(
        os.path.join(str(tmp_path / "out"), "postgkyl_compare_sims.pdf"))

  def test_pdf_name_given_no_out_dir(self, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    compare.sims(path0, path1, single_pdf=True, pdf_name="report.pdf")
    assert os.path.isfile(os.path.join(str(tmp_path), "report.pdf"))

  def test_pdf_name_and_out_dir_given(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    compare.sims(path0, path1, single_pdf=True, pdf_name="report.pdf",
                out_dir=str(tmp_path / "out"))
    assert os.path.isfile(os.path.join(str(tmp_path / "out"), "report.pdf"))

  def test_pdf_name_without_single_pdf_raises(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    with pytest.raises(ValueError, match="pdf_name requires single_pdf"):
      compare.sims(path0, path1, pdf_name="report.pdf")

  def test_format_is_ignored(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    compare.sims(path0, path1, single_pdf=True, format="jpg",
                out_dir=str(tmp_path / "out"))
    assert os.path.isfile(
        os.path.join(str(tmp_path / "out"), "postgkyl_compare_sims.pdf"))

  def test_multiblock_dynvector_contributes_one_page_per_block(self, tmp_path):
    path0 = str(tmp_path / "baseline")
    path1 = str(tmp_path / "candidate")
    os.makedirs(path0, exist_ok=True)
    os.makedirs(path1, exist_ok=True)
    time = np.linspace(0.0, 1.0, 4)
    for path, scale in ((path0, 1.0), (path1, 1.2)):
      for block in range(3):
        _write_dynvec(path, f"sim_b{block}", "field_energy", time,
                     (np.linspace(1.0, 2.0, 4) * scale * (block + 1))[:, None])
    report = _only(
        compare.sims(path0, path1, single_pdf=True, num_plots=1,
                    out_dir=str(tmp_path / "out")))
    assert report.plotted == ("field_energy",)
    pdf_path = os.path.join(str(tmp_path / "out"),
                            "postgkyl_compare_sims.pdf")
    assert _pdf_page_count(pdf_path) == 1 + 3  # title page + one per block

  def test_batch_writes_one_combined_pdf(self, tmp_path):
    base = tmp_path / "baseline"
    cand = tmp_path / "candidate"
    for sim_name in ("sim_a", "sim_b"):
      _sim(base.parent, base.name + f"/{sim_name}", name=sim_name)
      _sim(cand.parent, cand.name + f"/{sim_name}", name=sim_name,
          moms_scale=1.5)
    reports = compare.sims(str(base), str(cand), batch=True, single_pdf=True,
                           out_dir=str(tmp_path / "out"))
    assert set(reports) == {"sim_a", "sim_b"}
    pdf_path = os.path.join(str(tmp_path / "out"),
                            "postgkyl_compare_sims.pdf")
    assert os.path.isfile(pdf_path)
    total_plotted = sum(len(r.plotted) for r in reports.values())
    assert _pdf_page_count(pdf_path) == 1 + total_plotted
    # No per-simulation subfolders when single_pdf is set.
    assert not os.path.isdir(os.path.join(str(tmp_path / "out"), "sim_a"))
    assert not os.path.isdir(os.path.join(str(tmp_path / "out"), "sim_b"))


class TestTitlePageFigure:

  def test_non_batch_names_both_paths(self):
    fig = compare._title_page_figure("/a/baseline", "/a/candidate",
                                     batch=False)
    text = fig.texts[0].get_text()
    assert "/a/baseline" in text
    assert "/a/candidate" in text

  def test_batch_lists_every_simulation(self):
    fig = compare._title_page_figure(
        "/a/baseline", "/a/candidate", batch=True,
        sim_names=["sim_a", "sim_b", "sim_c"])
    text = fig.texts[0].get_text()
    assert "/a/baseline" in text
    assert "/a/candidate" in text
    for name in ("sim_a", "sim_b", "sim_c"):
      assert name in text
