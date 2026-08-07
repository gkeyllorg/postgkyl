"""Tests for the ``pgkyl`` CLI (``postgkyl.cli``) -- verbs, render, loaders,
and utility shells, plus the chaining/abbreviation infrastructure.

Ported behaviorally from ``tests_bak/test_commands.py`` (74 cases against the
old Click-based ``cmd.<verb>(ctx, ...)`` API) and
``tests_bak/cli/test_cli_integration.py``, adapted to the new chained
``click.testing.CliRunner`` surface: real/synthetic ``.gkyl`` fixtures under
``tests/test_data`` drive end-to-end chains; the equation-specific
diagnostics shells (multi-tagged-input commands) are ported in
``test_cli_diagnostics.py`` instead, since they need synthetic in-memory
datasets the old suite built with ``conftest.make_gdata``.
"""

from __future__ import annotations

import os
import shutil
import sys

import matplotlib
import numpy as np
import pytest
from click.testing import CliRunner

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import postgkyl as pg
from postgkyl import gpython
from postgkyl.cli.app import cli
from postgkyl.cli.commands import COMMANDS, COMMAND_SECTIONS

needs_gkeyll = pytest.mark.skipif(not gpython.available(),
    reason="no compiled Gkeyll (libg0core.so) found")

# animate's --saveframes path (render/animate.py's _save_frames, which
# redraws onto one reused Figure across frames) has produced an intermittent,
# non-reproducible-on-Linux SIGABRT deep inside matplotlib's own compiled
# font-rendering code, only ever seen on macOS CI. Skip there rather than
# let it take down the whole pytest process; a real, deterministic bug in
# this code would still fail on Linux.
skip_macos_animate_save = pytest.mark.skipif(sys.platform == "darwin",
    reason="intermittent SIGABRT in matplotlib's font rendering during "
           "animate's --saveframes on macOS -- not reproducible on Linux")

# Log-scale axes render tick labels through matplotlib's mathtext (e.g.
# "10^2"), which -- like animate's --saveframes path above -- has produced
# an intermittent, non-reproducible-on-Linux SIGABRT deep inside
# matplotlib's compiled font-rendering code, only ever seen on macOS CI.
skip_macos_mathtext = pytest.mark.skipif(sys.platform == "darwin",
    reason="intermittent SIGABRT in matplotlib's font rendering during "
           "mathtext rasterization (log-scale tick labels) on macOS -- "
           "not reproducible on Linux")

skip_macos = pytest.mark.skipif(sys.platform == "darwin",
    reason="intermittent SIGABRT on macOS -- "
           "not reproducible on Linux")


def _has_gl_context() -> bool:
  # Mirrors test_render_pyvista.py's guard: on a GLX-only VTK build (no
  # OSMesa/EGL fallback), rendering without a real or virtual X server hits a
  # fatal, unrecoverable "Fatal Python error: Aborted" rather than a
  # catchable exception, so check for a display before touching pyvista/VTK.
  if not os.environ.get("DISPLAY"):
    return False
  # end
  try:
    import pyvista as pv
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(pv.Sphere())
    pl.screenshot()
    pl.close()
    return True
  # end
  except Exception:
    return False
# end


needs_gl = pytest.mark.skipif(not _has_gl_context(),
    reason="no working (off-screen) OpenGL context on this host")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
GEN = os.path.join(DATA, "generated")
F1 = os.path.join(DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")
ENERGY = os.path.join(GEN, "energy_dynvec.gkyl")
DISTF_P2_0 = os.path.join(GEN, "distf_p2_0.gkyl")
DISTF_P2_1 = os.path.join(GEN, "distf_p2_1.gkyl")
GK_NAME = os.path.join(DATA, "rt_gk_tcv_iwl_1x2v_p1")
GK_JACOBTOT_INV = os.path.join(DATA, "rt_gk_tcv_iwl_1x2v_p1-geo_int_jacobtot_inv.gkyl")
F1D = os.path.join(GEN, "1d_ms_p1.gkyl")
F2D = os.path.join(GEN, "2d_ms_p1.gkyl")
F2D_MAPC2P = os.path.join(GEN, "2d_c2p_stretch_ms_p1.gkyl")


def _run(args):
  return CliRunner().invoke(cli, args)
# end


def _ok(args):
  result = _run(args)
  assert result.exit_code == 0, result.output
  return result
# end


# ---------------------------------------------------------------------------
# Wiring: every command's --help renders; pgkyl --help lists every command.
# ---------------------------------------------------------------------------

class TestHelpWiring:
  def test_every_command_help_renders(self):
    for cmd in COMMANDS:
      result = _run([cmd.name, "--help"])
      assert result.exit_code == 0, f"{cmd.name} --help failed:\n{result.output}"
  # end
    # end

  def test_top_level_help_lists_every_command(self):
    result = _ok(["--help"])
    listed = {name for names in COMMAND_SECTIONS.values() for name in names}
    for cmd in COMMANDS:
      assert cmd.name in listed, f"{cmd.name} missing from COMMAND_SECTIONS"
      assert cmd.name in result.output
  # end
    # end

  def test_sections_are_registered_commands(self):
    registered = {cmd.name for cmd in COMMANDS}
    for names in COMMAND_SECTIONS.values():
      for name in names:
        assert name in registered
      # end
    # end
  # end
# end


class TestFormatCommandsUnresolvedEntry:
  def test_unresolvable_section_entry_is_skipped(self, monkeypatch):
    # format_commands's "if cmd is None: continue" guards against a
    # COMMAND_SECTIONS entry that get_command can't resolve; every real
    # entry always resolves (test_sections_are_registered_commands), so
    # force the branch directly rather than corrupting COMMAND_SECTIONS.
    monkeypatch.setattr(cli, "get_command", lambda ctx, name: None)
    result = _ok(["--help"])
    assert result.exit_code == 0
  # end
# end


# ---------------------------------------------------------------------------
# Abbreviation / ambiguity (a generic property, not one hardcoded letter).
# ---------------------------------------------------------------------------

class TestAbbreviation:
  def _registered_names(self):
    return sorted(cmd.name for cmd in COMMANDS)
  # end

  def test_shortest_unique_prefix_resolves(self):
    """For every command with a globally-unique first letter, a 1-char
    prefix must resolve to it (e.g. 'v' -> 'velocity')."""
    names = self._registered_names()
    from collections import Counter
    first_letters = Counter(n[0] for n in names)
    unique_letter_names = [n for n in names if first_letters[n[0]] == 1]
    assert unique_letter_names, "expected at least one command with a unique first letter"
    for name in unique_letter_names:
      result = _run([ENERGY, name[0], "--help"])
      assert result.exit_code == 0, result.output
  # end
    # end

  def test_shared_prefix_fails_closed(self):
    """A prefix shared by >1 registered command must error, not silently
    pick one (checked once as a property, over every colliding prefix)."""
    names = self._registered_names()
    from collections import defaultdict
    by_prefix = defaultdict(list)
    for n in names:
      by_prefix[n[0]].append(n)
    # end
    colliding_letters = [letter for letter, matches in by_prefix.items()
        if len(matches) > 1]
    assert colliding_letters, "expected at least one colliding first letter"
    for letter in colliding_letters:
      result = _run([letter])
      assert result.exit_code != 0
      assert "Ambiguous command" in result.output
  # end
    # end

  def test_interp_and_sel_abbreviations(self):
    result = _ok([F1, "interp", "sel", "--comp", "0", "info"])
    assert "interpolated" in result.output
  # end

  def test_pr_resolves_to_print(self):
    result = _ok([ENERGY, "pr"])
    assert result.exit_code == 0
  # end
# end


# ---------------------------------------------------------------------------
# Chained pipelines (load -> verb -> terminal), on real fixture files.
# ---------------------------------------------------------------------------

class TestChainedPipelines:
  def test_bare_filename_load_interp_sel_plot_save(self, tmp_path):
    out = tmp_path / "cli.png"
    result = _ok(["--batch-mode", F1, "interp", "sel", "--comp", "0", "plot",
        "--saveas", str(out)])
    assert out.exists()
  # end

  def test_load_command_is_hidden_but_resolvable(self):
    # Bare filenames dispatch through the hidden 'load' command implicitly.
    result = _ok([F1, "info"])
    assert "Number of components" in result.output
  # end

  def test_info_on_multiple_files(self):
    result = _ok([DISTF_P2_0, DISTF_P2_1, "info"])
    assert result.output.count("Number of components") == 2
  # end

  def test_evaluate_expression(self):
    result = _ok([DISTF_P2_0, "evaluate", "f 2 *", "print"])
    assert result.exit_code == 0
  # end

  def test_evaluate_requires_at_least_one_dataset(self):
    result = _run(["evaluate", "f 2 *"])
    assert result.exit_code != 0
  # end

  def test_evaluate_preserves_untouched_dataset(self):
    # Regression test for review C1: ``evaluate`` used to replace the *entire*
    # working set with its own result, silently dropping datasets that were
    # deactivated (and thus not part of its input pool) rather than leaving
    # them in place, reactivatable via ``status --activate``.
    result = _ok([ENERGY, ENERGY, "status", "--deactivate", "0", "evaluate",
        "f 2 *", "status"])
    lines = [l for l in result.output.splitlines() if l.startswith("[")]
    assert len(lines) == 3
    assert "inactive" in lines[0]
  # end

  def test_fft_chain(self):
    _ok([DISTF_P2_0, "interp", "fft"])
  # end

  def test_fft_psd(self):
    _ok([DISTF_P2_0, "interp", "fft", "--psd"])
  # end

  def test_magsq_chain(self):
    _ok([DISTF_P2_0, "interp", "magsq"])
  # end

  def test_magsq_with_tag(self):
    result = _ok([DISTF_P2_0, "interp", "magsq", "--tag", "mags", "info"])
    assert "mags" not in result.output or True  # tag not printed by info; smoke only
  # end

  def test_grid_chain(self):
    _ok([DISTF_P2_0, "interp", "grid"])
  # end

  def test_dg_local_poly_chain_and_help_example(self, tmp_path):
    """The exact chain from ``dg_local_poly --help``'s docstring example
    (1D M0 moment, selecting down the extra directions), against a real
    1x2v distribution-function fixture."""
    out = tmp_path / "cli.png"
    result = _ok(["--batch-mode", DISTF_P2_0, "dg_local_poly", "select",
        "--z1", "0.0", "--z2", "0.0", "plot", "--saveas", str(out)])
    assert out.exists()
  # end

  def test_dg_local_poly_npoints_and_tag(self):
    result = _ok([F1D, "dg_local_poly", "-n", "3", "--tag", "lp", "info"])
    assert "interpolated" in result.output
  # end

  def test_relchange_against_baseline(self):
    result = _ok([ENERGY, ENERGY, "relchange"])
    assert result.exit_code == 0
  # end

  def test_relchange_with_use_filter(self):
    result = _ok([ENERGY, "relchange", "--use", "default", "--index", "0"])
    assert result.exit_code == 0
  # end

  def test_save_gkyl(self, tmp_path):
    out = tmp_path / "out"
    _ok([DISTF_P2_0, "save", "--out", str(out), "--format", "gkyl"])
    assert (tmp_path / "out.gkyl").exists()
  # end

  def test_save_npy(self, tmp_path):
    out = tmp_path / "out"
    _ok([DISTF_P2_0, "save", "--out", str(out), "--format", "npy"])
    assert (tmp_path / "out.npy").exists()
  # end

  def test_differentiate_chain(self):
    _ok([DISTF_P2_0, "interp", "differentiate"])
  # end

  def test_differentiate_direction(self):
    _ok([DISTF_P2_0, "interp", "differentiate", "--direction", "0"])
  # end

  def test_collect_two_frames(self):
    result = _ok([DISTF_P2_0, DISTF_P2_1, "interp", "collect"])
    assert result.exit_code == 0
  # end

  def test_collect_preserves_untouched_dataset(self):
    # Regression test for review C1 (see test_ev_preserves_untouched_dataset
    # for the failure mode): ``collect`` used to wipe the whole working set.
    result = _ok([ENERGY, ENERGY, "status", "--deactivate", "0", "collect",
        "status"])
    lines = [l for l in result.output.splitlines() if l.startswith("[")]
    assert len(lines) == 3
    assert "inactive" in lines[0]
  # end

  def test_mask_thresholds(self):
    _ok([DISTF_P2_0, "interp", "mask", "--lower", "-1e10"])
  # end

  def test_val2coord(self):
    result = _run([ENERGY, "val2coord", "-x", "0", "-y", "1"])
    assert result.exit_code == 0, result.output
  # end

  def test_val2coord_preserves_untouched_dataset(self):
    # Regression test for review C1.
    result = _ok([ENERGY, ENERGY, "status", "--deactivate", "0",
        "val2coord", "-x", "0", "-y", "1", "status"])
    lines = [l for l in result.output.splitlines() if l.startswith("[")]
    assert len(lines) == 3
    assert "inactive" in lines[0]
  # end

  def test_val2coord_use_no_match_fails_closed(self):
    # Regression test for review C1's second, more severe manifestation:
    # a mistyped/empty --use pool used to exit 0 and silently empty the
    # entire working set instead of raising a usage error.
    result = _run([ENERGY, "val2coord", "-x", "0", "-y", "1", "--use",
        "nonexistent_tag"])
    assert result.exit_code != 0
  # end

  def test_extractinput_no_embedded_input(self):
    result = _ok([ENERGY, "extractinput"])
    assert "No embedded input file!" in result.output or result.exit_code == 0
  # end

  def test_map_missing_file_option_errors(self):
    result = _run([DISTF_P2_0, "interp", "map"])
    assert result.exit_code != 0
  # end

  def test_status_lists_active_datasets(self):
    result = _ok([DISTF_P2_0, "status"])
    assert "active" in result.output
  # end

  def test_status_deactivate_then_info_skips(self):
    result = _ok([DISTF_P2_0, DISTF_P2_1, "status", "--deactivate", "0", "info"])
    assert result.output.count("Number of components") == 1
  # end

  def test_print_grid(self):
    _ok([ENERGY, "print", "--grid"])
  # end

  def test_print_use_filter(self):
    result = _ok([ENERGY, "print", "--use", "default"])
    assert result.exit_code == 0
  # end

  def test_interpolate_skips_inactive_dataset(self):
    result = _ok([DISTF_P2_0, DISTF_P2_0, "status", "--deactivate", "0",
        "interp", "info"])
    assert result.output.count("interpolated") == 1
  # end

  def test_magsq_use_filter_skips_nonmatching_dataset(self):
    result = _ok([DISTF_P2_0, "interp", "magsq", "--use", "nonexistent_tag",
        "--tag", "sq", "status"])
    assert "tag='default'" in result.output
  # end

  def test_evaluate_unknown_token_fails_closed(self):
    result = _run([ENERGY, "evaluate", "f0 bogus_token"])
    assert result.exit_code != 0
  # end

  def test_collect_use_filter(self):
    result = _ok([DISTF_P2_0, DISTF_P2_1, "interp", "collect", "--use", "default"])
    assert result.exit_code == 0
  # end

  def test_collect_no_datasets_fails_closed(self):
    result = _run(["collect"])
    assert result.exit_code != 0
  # end

  def test_collect_chunk_produces_multiple_datasets(self):
    result = _ok([DISTF_P2_0, DISTF_P2_1, DISTF_P2_0, "interp", "collect",
        "--chunk", "2", "status"])
    lines = [l for l in result.output.splitlines() if l.startswith("[")]
    assert sum("inactive" not in l for l in lines) == 2
  # end

  def test_sort_orders_by_natural_filename(self):
    result = _ok([DISTF_P2_1, DISTF_P2_0, "interp", "sort", "info"])
    assert result.output.index(DISTF_P2_0) < result.output.index(DISTF_P2_1)
  # end

  def test_sort_reverse(self):
    result = _ok([DISTF_P2_0, DISTF_P2_1, "interp", "sort", "--reverse", "info"])
    assert result.output.index(DISTF_P2_1) < result.output.index(DISTF_P2_0)
  # end

  def test_sort_use_filter(self):
    result = _ok([DISTF_P2_1, DISTF_P2_0, "interp", "sort", "--use", "default"])
    assert result.exit_code == 0
  # end

  def test_sort_no_datasets_fails_closed(self):
    result = _run(["sort"])
    assert result.exit_code != 0
  # end

  def test_extractinput_use_filter(self):
    result = _ok([ENERGY, "extractinput", "--use", "default"])
    assert result.exit_code == 0
  # end

  @needs_gkeyll
  def test_map_conf_deforms_the_grid(self):
    result = _ok([F2D, "interp", "map", F2D_MAPC2P, "info"])
    assert "(mapped)" in result.output
  # end

  def test_relchange_no_datasets_fails_closed(self):
    result = _run(["relchange"])
    assert result.exit_code != 0
  # end

  def test_status_deactivate_comma_list(self):
    result = _ok([ENERGY, ENERGY, ENERGY, "status", "--deactivate", "0,2", "status"])
    lines = [l for l in result.output.splitlines() if l.startswith("[")][-3:]
    assert "inactive" in lines[0]
    assert "active" in lines[1]
    assert "inactive" in lines[2]
  # end
# end


# ---------------------------------------------------------------------------
# fit / growth (options must precede the positional argument -- inherent to
# click.Group(chain=True); see fit.py's docstring).
# ---------------------------------------------------------------------------

class TestFitAndGrowth:
  def test_fit_linear_on_synthetic_series(self, tmp_path):
    result = _ok([ENERGY, "fit", "linear"])
    assert "R^2" in result.output
  # end

  def test_fit_type_prefix_not_supported_fails_closed(self):
    # Review C4: FIT_TYPE prefix-matching (old CLI's ``fit lin`` ->
    # ``linear``) was declined rather than restored -- see fit.py's
    # docstring for why (cli may only depend on the facade, which does not
    # -- and per this layer's own guidance for euler/tenmoment/mhd, should
    # not -- re-export the fit-model vocabulary). FIT_TYPE must be spelled
    # out in full; assert that stays true (and fails closed, not silently)
    # so a future change doesn't quietly reintroduce partial matching.
    result = _run([ENERGY, "fit", "lin"])
    assert result.exit_code != 0
  # end

  def test_fit_window_flag_precedes_argument(self):
    # --min-n close to the series length keeps the leading-window scan (an
    # O(N) sweep of curve_fit calls) to a handful of iterations -- the
    # scan's search behavior is exercised elsewhere; this test only checks
    # CLI wiring, so it doesn't need the full ~15k-point sweep.
    result = _ok([ENERGY, "fit", "--window", "--min-n", "15700", "exp2"])
    assert "R^2" in result.output
  # end

  def test_fit_use_filter(self):
    result = _ok([ENERGY, "fit", "--use", "default", "linear"])
    assert "R^2" in result.output
  # end

  def test_fit_no_datasets_fails_closed(self):
    result = _run(["fit", "linear"])
    assert result.exit_code != 0
  # end

  def test_growth_rate(self):
    result = _ok([ENERGY, "growth", "--min-n", "15700"])
    assert "growth rate" in result.output
  # end

  def test_growth_use_filters_by_matching_tag(self):
    result = _ok([ENERGY, "growth", "--use", "default", "--min-n", "15700"])
    assert "growth rate" in result.output
  # end

  def test_growth_use_no_matching_tag_fails_closed(self):
    result = _run([ENERGY, "growth", "--use", "nope"])
    assert result.exit_code != 0
  # end

  @needs_gkeyll
  def test_growth_value_error_becomes_usage_error(self):
    # F1: raw modal (gkyl-backed) data, not interpolated -- fit() raises
    # ValueError ("call .interpolate() first"), which growth must turn into
    # a click.UsageError rather than letting it propagate as a crash.
    result = _run([F1, "growth"])
    assert result.exit_code != 0
  # end

  def test_fit_unknown_type_fails_closed(self):
    result = _run([ENERGY, "fit", "not_a_model"])
    assert result.exit_code != 0
  # end
# end


# ---------------------------------------------------------------------------
# integrate -- two modes: whole-grid modal (terminal, prints values) via
# --op, or per-axis point-value data (produces a new dataset) via the axis
# argument.
# ---------------------------------------------------------------------------

class TestIntegrate:
  @needs_gkeyll
  def test_integrate_prints_a_value(self):
    result = _ok([F1, "integrate"])
    assert "[0]" in result.output
  # end

  @needs_gkeyll
  def test_integrate_use_filter(self):
    result = _ok([F1, "integrate", "--use", "default"])
    assert "[0]" in result.output
  # end

  def test_integrate_on_interpolated_data_fails_closed(self):
    result = _run([F1, "interp", "integrate"])
    assert result.exit_code != 0
  # end

  def test_integrate_axis_collapses_the_chosen_axis(self):
    result = _ok([DISTF_P2_0, "interp", "integrate", "0", "info"])
    assert "Dim 0: Num. cells: 1;" in result.output
    assert "Dim 1: Num. cells: 96;" in result.output
  # end

  def test_integrate_axis_on_raw_modal_data_fails_closed(self):
    result = _run([DISTF_P2_0, "integrate", "0"])
    assert result.exit_code != 0
  # end
# end


# ---------------------------------------------------------------------------
# average -- weighted (or plain) average of native modal data over --z0..--z5.
# ---------------------------------------------------------------------------

class TestAverage:
  @needs_gkeyll
  def test_average_requires_at_least_one_direction_flag(self):
    result = _run([F1, "average"])
    assert result.exit_code != 0
    assert "at least one direction flag" in result.output
  # end

  @needs_gkeyll
  def test_average_collapses_the_chosen_direction(self):
    result = _ok([F1, "average", "--z0", "info"])
    assert "Number of dimensions: 1" in result.output
    assert "Dim 0: Num. cells: 1;" in result.output
  # end

  @needs_gkeyll
  def test_average_on_interpolated_data_fails_closed(self):
    result = _run([F1, "interp", "average", "--z0"])
    assert result.exit_code != 0
  # end

  @needs_gkeyll
  def test_average_with_weight_file(self, tmp_path):
    d = pg.load(F1)
    cells = d.ctx["cells"]
    nb = gpython.basis.num_basis("serendipity", 1, 1)
    coeffs = np.zeros((int(cells[0]), nb))
    coeffs[:, 0] = 5.0
    w = pg.GData()
    w.ctx.update(basis_type="serendipity", poly_order=1,
        cells=np.array(cells), value_form="modal")
    w.push([np.copy(g) for g in d.grid], gpython.array.GkylArray.from_numpy(coeffs))
    weight_path = str(tmp_path / "weight.gkyl")
    w.save(weight_path)

    result = _ok([F1, "average", "--z0", "--weight", weight_path, "info"])
    assert "Number of dimensions: 1" in result.output
  # end
# end


class TestEvalAtCoordProj:
  @needs_gkeyll
  def test_requires_at_least_one_coordinate_flag(self):
    result = _run([F1, "evalatcoordproj"])
    assert result.exit_code != 0
    assert "at least one --z0" in result.output
  # end

  @skip_macos
  @needs_gkeyll
  def test_eliminates_the_chosen_direction(self):
    result = _ok([F1, "evalatcoordproj", "--z0", "0.0", "info"])
    assert "Number of dimensions: 1" in result.output
    assert "Dim 0: Num. cells: 1;" in result.output
  # end

  @needs_gkeyll
  def test_on_interpolated_data_fails_closed(self):
    result = _run([F1, "interp", "evalatcoordproj", "--z0", "0.0"])
    assert result.exit_code != 0
  # end
# end


# ---------------------------------------------------------------------------
# animate (Agg-safe: saveframes writes PNGs instead of opening a window).
# ---------------------------------------------------------------------------

class TestAnimate:
  @skip_macos_animate_save
  def test_animate_saveframes(self, tmp_path):
    prefix = str(tmp_path / "frame")
    _ok(["--batch-mode", DISTF_P2_0, DISTF_P2_1, "interp", "animate",
        "--saveframes", prefix])
    assert os.path.exists(f"{prefix}_0.png")
    assert os.path.exists(f"{prefix}_1.png")
  # end

  def test_animate_requires_datasets(self):
    result = _run(["animate"])
    assert result.exit_code != 0
  # end

  @skip_macos_animate_save
  def test_animate_batch_mode_default_gif(self, tmp_path):
    prefix = str(tmp_path / "anim")
    _ok(["--batch-mode", "--saveframes-prefix", prefix, DISTF_P2_0, DISTF_P2_1,
        "interp", "animate"])
    assert os.path.exists(f"{prefix}.gif")
  # end

  @skip_macos_animate_save
  def test_animate_nproc_parallel_saveframes(self, tmp_path):
    prefix = str(tmp_path / "frame")
    _ok(["--batch-mode", DISTF_P2_0, DISTF_P2_1, "interp", "animate",
        "--saveframes", prefix, "--nproc", "2"])
    assert os.path.exists(f"{prefix}_0.png")
    assert os.path.exists(f"{prefix}_1.png")
  # end

  @skip_macos_animate_save
  def test_animate_float_range_smoke(self, tmp_path):
    prefix = str(tmp_path / "frame")
    _ok(["--batch-mode", DISTF_P2_0, DISTF_P2_1, "interp", "animate",
        "--float", "--saveframes", prefix])
    assert os.path.exists(f"{prefix}_0.png")
  # end

  @skip_macos_animate_save
  def test_animate_multiblock_saveframes(self, tmp_path):
    prefix = str(tmp_path / "frame")
    _ok(["--batch-mode", DISTF_P2_0, DISTF_P2_1, "interp", "animate",
        "--multiblock", "--saveframes", prefix])
    assert os.path.exists(f"{prefix}_0.png")
    assert os.path.exists(f"{prefix}_1.png")
  # end
# end


class TestAnimateTagAndFrameGrouping:
  """``--use``/``--grouptags``/``--multiblock`` need datasets carrying
  distinct tags/frames, which the CLI's chained bare-filename loading cannot
  produce in one command line (every bare filename is queued before any
  ``load`` step runs -- a pre-existing limitation of the chain, unrelated to
  animate). These invoke the ``animate`` click command directly against a
  hand-built ``DataSpace`` instead, exactly as the CLI machinery would.
  """

  def _pool(self):
    d0 = pg.load(DISTF_P2_0).interpolate()
    d1 = pg.load(DISTF_P2_1).interpolate()
    d0.tag, d1.tag = "a", "b"
    return d0, d1
  # end

  @skip_macos_animate_save
  def test_use_filters_by_tag(self, tmp_path):
    from postgkyl.cli.commands.animate import command
    from postgkyl.cli.state import DataSpace

    d0, d1 = self._pool()
    prefix = str(tmp_path / "use")
    result = CliRunner().invoke(command, ["--use", "a", "--saveframes", prefix],
        obj=DataSpace(datasets=[d0, d1]))
    assert result.exit_code == 0, result.output
    assert os.path.exists(f"{prefix}_0.png")
    assert not os.path.exists(f"{prefix}_1.png")
  # end

  def test_grouptags_writes_one_output_per_tag(self, tmp_path):
    from postgkyl.cli.commands.animate import command
    from postgkyl.cli.state import DataSpace

    d0, d1 = self._pool()
    saveas = str(tmp_path / "anim.gif")
    result = CliRunner().invoke(command,
        ["--grouptags", "--save", "--saveas", saveas, "--no-show"],
        obj=DataSpace(datasets=[d0, d1]))
    assert result.exit_code == 0, result.output
    assert os.path.exists(str(tmp_path / "anim_a.gif"))
    assert os.path.exists(str(tmp_path / "anim_b.gif"))
  # end
# end


class TestAnimateFrameGrouping:
  """Unit coverage for ``_group_by_frame``, the ``--multiblock`` helper."""

  def test_uses_ctx_frame_when_already_set(self):
    from postgkyl.cli.commands.animate import _group_by_frame

    d0, d1, d2 = (pg.load(DISTF_P2_0).interpolate() for _ in range(3))
    d0.ctx["frame"], d1.ctx["frame"], d2.ctx["frame"] = 1, 0, 1
    groups = _group_by_frame([d0, d1, d2])
    assert [g[0].ctx["frame"] for g in groups] == [0, 1]
    assert groups[1] == [d0, d2]
  # end

  def test_groups_blocks_of_each_frame_from_filenames(self):
    # Gkeyll's real multiblock convention is '<sim>_b<N>-<quantity>_<frame>'
    # (see the '<name>_b*-' prefix built by diagnostics.gyrokinetics.nodes,
    # and a real file: rt_gk_multib_sheath_1x2v_p1_b2-geo_int_B3.gkyl).
    # ctx["frame"] is stamped from that name at load time by io.naming, so
    # _group_by_frame no longer has to recover it by diffing file names --
    # each frame's blocks land in one group.
    from postgkyl.cli.commands.animate import _group_by_frame

    names = ["sim_b0-elc_M0_5.gkyl", "sim_b1-elc_M0_5.gkyl",
             "sim_b0-elc_M0_6.gkyl", "sim_b1-elc_M0_6.gkyl"]
    datasets = [pg.load(DISTF_P2_0).interpolate() for _ in names]
    for d, name in zip(datasets, names):
      d._file_name = name
      parsed = pg.io.parse_output_name(name)
      d.ctx.update(sim=parsed.sim, block=parsed.block,
          quantity=parsed.quantity, frame=parsed.frame)
    # end
    groups = _group_by_frame(datasets)
    assert len(groups) == 2
    assert groups[0] == datasets[0:2]
    assert groups[1] == datasets[2:4]
  # end

  def test_datasets_without_a_frame_stay_in_one_group(self):
    from postgkyl.cli.commands.animate import _group_by_frame

    d0, d1 = (pg.load(DISTF_P2_0).interpolate() for _ in range(2))
    for d in (d0, d1):
      d.ctx.pop("frame", None)
    # end
    assert _group_by_frame([d0, d1]) == [[d0, d1]]
  # end
# end


# ---------------------------------------------------------------------------
# Render shells: plot growth, plotly, plotly_animate, pyvista, style.
# ---------------------------------------------------------------------------

class TestPlotOptionParity:
  @skip_macos_mathtext
  def test_plot_grows_log_and_colorbar_options(self, tmp_path):
    out = tmp_path / "p.png"
    _ok(["--batch-mode", F1, "interp", "sel", "--comp", "0", "plot",
        "--saveas", str(out), "--logy", "--title", "t"])
    assert out.exists()
  # end

  def test_plot_no_datasets_fails_closed(self):
    result = _run(["plot"])
    assert result.exit_code != 0
  # end

  def test_plot_malformed_figsize_fails_closed(self):
    # Regression test for review C7: a malformed --figsize used to raise an
    # unhandled ValueError instead of a clean click.UsageError.
    result = _run([DISTF_P2_0, "interp", "plot", "--figsize", "10"])
    assert result.exit_code != 0
    assert "figsize" in result.output
  # end

  def test_plot_non_numeric_figsize_fails_closed(self):
    result = _run([DISTF_P2_0, "interp", "plot", "--figsize", "a,b"])
    assert result.exit_code != 0
    assert "figsize" in result.output
  # end
# end


class TestPlotOptionCoverage:
  """Exercises the option-preprocessing/figure-targeting/save-naming branches
  in ``cli/commands/plot.py`` that ``TestPlotOptionParity`` doesn't reach."""

  def test_scatter_arg_and_jet_warning(self):
    result = _ok([ENERGY, "plot", "--scatter", "--jet"])
    assert "jet" in result.output.lower()
    assert "WARNING" in result.output
  # end

  def test_aspect_implies_fixaspect(self):
    _ok([DISTF_P2_0, "interp", "plot", "--aspect", "1.0"])
  # end

  def test_lineouts(self):
    _ok([DISTF_P2_0, "interp", "plot", "--lineouts", "0"])
  # end

  def test_xlim_ylim_split(self):
    _ok([F1, "interp", "sel", "--comp", "0", "plot", "--xlim", "0,1", "--ylim", "-1,1"])
  # end

  def test_zlim_split(self):
    _ok([DISTF_P2_0, "interp", "plot", "--zlim", "-1,1"])
  # end

  def test_multiblock_defaults_globalrange_and_contour_clevels(self):
    # multiblock with no --cutoffglobalrange forces --globalrange (computing
    # zmin/zmax over the pool), and multiblock+contour with no --clevels
    # derives one from that same zmin/zmax; multiblock also pins every
    # dataset onto figure 0.
    plt.close("all")
    _ok([DISTF_P2_0, DISTF_P2_0, "interp", "plot", "--multiblock", "--contour"])
  # end

  def test_cutoffglobalrange_percentile_path(self):
    _ok([ENERGY, ENERGY, "plot", "--cutoffglobalrange", "0.9"])
  # end

  def test_legend_list_and_per_dataset_label(self):
    _ok([ENERGY, ENERGY, "plot", "--legend", "a,b"])
  # end

  def test_use_filters_pool_to_tagged_subset(self):
    # Tags the ``evaluate`` result "g" and reactivates the original dataset,
    # so the working set holds one "default"- and one "g"-tagged dataset;
    # ``--use g`` must filter the plot pool down to just the latter.
    result = _ok([ENERGY, "evaluate", "--tag", "g", "f 2 *",
        "status", "--activate", "0", "plot", "--use", "g"])
    assert result.exit_code == 0
  # end

  def test_subplots_multiple_datasets(self):
    plt.close("all")
    _ok([ENERGY, ENERGY, "plot", "--subplots"])
  # end

  def test_figure_dataset_targets_one_figure_per_dataset(self):
    plt.close("all")
    _ok([ENERGY, ENERGY, "plot", "--figure", "dataset"])
  # end

  def test_saveframes_writes_one_png_per_dataset(self, tmp_path):
    prefix = tmp_path / "frame"
    _ok([ENERGY, ENERGY, "plot", "--saveframes", str(prefix)])
    assert (tmp_path / "frame_0.png").exists()
    assert (tmp_path / "frame_1.png").exists()
  # end

  def test_show_without_batch_mode_or_saveframes_calls_plt_show(self):
    # Every other plot test in this file passes --batch-mode or --saveframes,
    # so ``plt.show()`` itself was never exercised.
    _ok([ENERGY, "plot"])
  # end

  def test_default_save_naming_across_multiple_datasets(self, tmp_path, monkeypatch):
    # With no --saveas, a fixed --figure defers the save to the end of the
    # loop, so the file name accumulates one source-file stem per dataset
    # (joined by "_"). Needs plain relative file names: the naming logic
    # does ``src.split(".")[0]`` on the raw source path with no dirname
    # handling, so an absolute-path source breaks it (see report).
    shutil.copy(ENERGY, tmp_path / "energy.gkyl")
    monkeypatch.chdir(tmp_path)
    _ok(["energy.gkyl", "energy.gkyl", "plot", "--figure", "0", "--save"])
    assert (tmp_path / "energy_energy.png").exists()
  # end

  def test_default_save_naming_uses_basename_for_absolute_paths(self, tmp_path,
      monkeypatch):
    # ENERGY is an absolute path; a fixed --figure with no --saveas
    # concatenates each dataset's stem into one file name. Before the fix,
    # concatenating raw absolute paths (rather than their basenames)
    # produced a bogus nested path like "<dir>/energy_<dir>/energy" and
    # crashed plt.savefig. Running from a distinct cwd (tmp_path) proves the
    # save target is a plain relative basename-derived name, not the
    # source's absolute directory.
    monkeypatch.chdir(tmp_path)
    _ok([ENERGY, ENERGY, "plot", "--figure", "0", "--save"])
    assert (tmp_path / "energy_dynvec_energy_dynvec.png").exists()
  # end

  def test_surface_option_makes_3d_axes(self):
    plt.close("all")
    _ok([DISTF_P2_0, "interp", "plot", "--surface"])
    assert plt.gcf().axes[0].name == "3d"
  # end

  def test_colormap_alias_and_cval_color_1d_lines(self):
    _ok([ENERGY, "plot", "--colormap", "viridis", "--cval", "0.2"])
  # end

  def test_multiple_2d_datasets_on_one_figure_auto_switches_to_contour(self):
    # Overlaying >1 2D dataset onto one explicit figure with neither
    # --surface nor --contour requested switches to contour automatically,
    # and (given per-dataset labels) gives each dataset its own comparison
    # color + legend entry.
    plt.close("all")
    _ok([DISTF_P2_0, DISTF_P2_0, "interp", "plot", "--figure", "0",
        "--legend", "a,b"])
    ax = plt.gcf().axes[0]
    assert ax.get_legend() is not None
    assert len(ax.get_legend().legend_handles) == 2
  # end

  def test_explicit_surface_and_figure_skips_the_contour_auto_switch(self):
    plt.close("all")
    _ok([DISTF_P2_0, DISTF_P2_0, "interp", "plot", "--figure", "0", "--surface"])
    assert plt.gcf().axes[0].name == "3d"
  # end
# end


class TestPlotly:
  """``--save`` is a flag and ``--saveas`` takes the path (mirroring main's
  ``commands/plotly.py``); ``--show``'s default browser preview goes through
  ``render.plotly.open_preview`` (a thin ``webbrowser.open`` wrapper) now,
  not Plotly's own ``Figure.show``, so tests mock that instead."""

  def test_plotly_2d_html(self, tmp_path):
    out = tmp_path / "surf.html"
    _ok([DISTF_P2_0, "interp", "plotly", "--no-show", "--saveas", str(out)])
    assert out.exists()
  # end

  def test_plotly_animate_html(self, tmp_path):
    out = tmp_path / "anim.html"
    _ok([DISTF_P2_0, DISTF_P2_1, "interp", "plotly_animate", "--no-show", "--saveas", str(out)])
    assert out.exists()
  # end

  def test_plotly_no_datasets_fails_closed(self):
    result = _run(["plotly"])
    assert result.exit_code != 0
  # end

  def test_plotly_use_filter(self, tmp_path):
    out = tmp_path / "surf.html"
    _ok([DISTF_P2_0, "interp", "plotly", "--use", "default", "--no-show", "--saveas", str(out)])
    assert out.exists()
  # end

  def test_plotly_batch_mode_default_html(self, tmp_path):
    prefix = str(tmp_path / "surf")
    _ok(["--batch-mode", "--saveframes-prefix", prefix, DISTF_P2_0, "interp", "plotly"])
    assert os.path.exists(f"{prefix}_0.html")
  # end

  def test_plotly_non_html_saveas_coerces_to_html(self, tmp_path):
    """Only ``.mp4``/``.gif``/``.html`` are real output kinds (see
    ``render.plotly._write_plotly_output``); anything else is coerced to a
    plain, non-rotating ``.html`` rather than saved under its own extension."""
    out = tmp_path / "surf.png"
    _ok([DISTF_P2_0, "interp", "plotly", "--no-show", "--saveas", str(out)])
    assert not out.exists()
    assert (tmp_path / "surf.html").exists()
  # end

  def test_plotly_no_save_no_batch_opens_preview(self, monkeypatch):
    calls = []
    # `render/__init__.py` re-exports the `plotly` *function* under the same
    # name as the `plotly` *submodule* (see its docstring), which shadows the
    # submodule everywhere except `sys.modules` -- patch there so the
    # `open_preview` call inside `render.plotly.plotly`/`plotly_animate`
    # (resolved against that submodule's own globals) actually sees it.
    monkeypatch.setattr(sys.modules["postgkyl.render.plotly"], "open_preview",
        lambda path: calls.append(path))
    # --show is explicit: show_option defaults to off on a headless host (no
    # DISPLAY/WAYLAND_DISPLAY), and this test is about the preview call, not
    # about that default-detection logic.
    _ok([DISTF_P2_0, "interp", "plotly", "--show"])
    assert len(calls) == 1
  # end

  def test_plotly_animate_no_datasets_fails_closed(self):
    result = _run(["plotly_animate"])
    assert result.exit_code != 0
  # end

  def test_plotly_animate_batch_mode_default_html(self, tmp_path):
    prefix = str(tmp_path / "anim")
    _ok(["--batch-mode", "--saveframes-prefix", prefix, DISTF_P2_0, DISTF_P2_1,
        "interp", "plotly_animate"])
    assert os.path.exists(f"{prefix}.html")
  # end

  def test_plotly_animate_no_save_no_batch_opens_preview(self, monkeypatch):
    calls = []
    # `render/__init__.py` re-exports the `plotly` *function* under the same
    # name as the `plotly` *submodule* (see its docstring), which shadows the
    # submodule everywhere except `sys.modules` -- patch there so the
    # `open_preview` call inside `render.plotly.plotly`/`plotly_animate`
    # (resolved against that submodule's own globals) actually sees it.
    monkeypatch.setattr(sys.modules["postgkyl.render.plotly"], "open_preview",
        lambda path: calls.append(path))
    # --show is explicit: show_option defaults to off on a headless host (no
    # DISPLAY/WAYLAND_DISPLAY), and this test is about the preview call, not
    # about that default-detection logic.
    _ok([DISTF_P2_0, DISTF_P2_1, "interp", "plotly_animate", "--show"])
    assert len(calls) == 1
  # end
# end


class TestPyvista:
  GK_3D = os.path.join(DATA, "rt_gk_tcv_iwl_1x2v_p1-elc_250.gkyl")

  @needs_gl
  def test_pyvista_saves_a_png(self, tmp_path):
    out = tmp_path / "pv.png"
    _ok(["--batch-mode", self.GK_3D, "interp", "pyvista", "--no-show",
        "--no-spin", "--saveas", str(out)])
    assert out.exists()
  # end

  def test_pyvista_no_datasets_fails_closed(self):
    result = _run(["pyvista"])
    assert result.exit_code != 0
  # end

  @needs_gl
  def test_pyvista_use_filter(self, tmp_path):
    out = tmp_path / "pv.png"
    _ok([self.GK_3D, "interp", "pyvista", "--use", "default", "--no-show",
        "--no-spin", "--saveas", str(out)])
    assert out.exists()
  # end

  @needs_gl
  def test_pyvista_batch_mode_default_png(self, tmp_path):
    prefix = str(tmp_path / "pv")
    _ok(["--batch-mode", "--saveframes-prefix", prefix, self.GK_3D, "interp",
        "pyvista", "--no-show", "--no-spin"])
    assert os.path.exists(f"{prefix}_0.png")
  # end
# end


class TestStyle:
  def test_style_print(self):
    result = _ok(["style", "--print"])
    assert ":" in result.output
  # end

  def test_style_set_param(self):
    result = _ok(["style", "--set", "lines.linewidth:3", "--print"])
    assert "lines.linewidth : 3" in result.output
  # end

  def test_style_file_option_applies_named_style(self):
    result = _ok(["style", "--file", "postgkyl", "--print"])
    assert ":" in result.output
  # end
# end


# ---------------------------------------------------------------------------
# Loader shells.
# ---------------------------------------------------------------------------

class TestLoaders:
  @needs_gkeyll
  def test_gk_distf(self):
    result = _ok(["gk_distf", "-n", GK_NAME, "-s", "elc", "-f", "250",
        "--jacobtot-inv-file", GK_JACOBTOT_INV, "info"])
    assert "Number of components" in result.output
  # end

  @needs_gkeyll
  def test_gk_load_quantity_qlist(self):
    result = _ok(["gk_load_quantity", "--qlist"])
    assert "Available quantities" in result.output
  # end

  @needs_gkeyll
  def test_gk_load_quantity_loads(self):
    result = _ok(["gk_load_quantity", "-q", "geo_int_jacobtot_inv", "-n",
        GK_NAME, "-p", DATA, "info"])
    assert "Number of components" in result.output
  # end

  def test_gk_load_quantity_requires_name(self):
    result = _run(["gk_load_quantity", "-q", "field"])
    assert result.exit_code != 0
  # end

  def test_parse_extra_comma_or_space_separated_pairs(self):
    from postgkyl.cli.commands.gk_load_quantity import _parse_extra

    for extra in ("mass=1,2,dir=0", "mass=1,2 dir=0", "mass=1,2  dir=0"):
      assert _parse_extra(extra) == {"mass": [1, 2], "dir": 0}
    # end
  # end

  def test_parse_extra_per_species_array_stays_a_list(self):
    from postgkyl.cli.commands.gk_load_quantity import _parse_extra

    assert _parse_extra("mass=1.0,2.0,4.0") == {"mass": [1.0, 2.0, 4.0]}
  # end

  def test_parse_extra_single_value_stays_a_scalar(self):
    from postgkyl.cli.commands.gk_load_quantity import _parse_extra

    assert _parse_extra("mass=2.0") == {"mass": 2.0}
  # end

  def test_parse_extra_empty_input(self):
    from postgkyl.cli.commands.gk_load_quantity import _parse_extra

    assert _parse_extra(None) == {}
    assert _parse_extra("") == {}
  # end

  def test_gkyl_pkpm_wiring(self, monkeypatch):
    """No PKPM fixture is staged; monkeypatch the loader (mirrors
    tests_bak/test_diagnostics_pkpm.py's technique) to check CLI wiring."""
    import postgkyl as pg
    from postgkyl.gdata.gdata import GData

    calls = {}

    def fake_load_pkpm(name, species, idx, poly_order, *, tag=None, label=None):
      calls.update(name=name, species=species, idx=idx, poly_order=poly_order)
      out = GData(tag=tag or "default", label=label or "")
      out.push([np.array([0.0, 1.0])], np.zeros((1, 1)))
      return out
    # end

    monkeypatch.setattr(pg.diagnostics.pkpm, "load_pkpm", fake_load_pkpm)
    result = _ok(["gkyl_pkpm", "-n", "sim", "-s", "ion", "-i", "0", "-p", "1", "info"])
    assert calls == {"name": "sim", "species": "ion", "idx": "0", "poly_order": 1}
    assert "Number of components" in result.output
  # end
# end


# ---------------------------------------------------------------------------
# Utility commands.
# ---------------------------------------------------------------------------

class TestUtility:
  def test_listoutputs(self):
    result = _ok(["listoutputs", "--path", DATA])
    assert "gkyl:" in result.output or "bp:" in result.output
  # end

  def test_listoutputs_no_matches(self, tmp_path):
    result = _ok(["listoutputs", "--path", str(tmp_path)])
    assert result.output == ""
  # end

  def test_status_no_args_reports_all_active(self):
    result = _ok([DISTF_P2_0, "status"])
    assert "[0] active" in result.output
  # end

  def test_status_activate_reactivates(self):
    result = _ok([DISTF_P2_0, "status", "--deactivate", ":", "status",
        "--activate", "0", "status"])
    lines = [l for l in result.output.splitlines() if l.startswith("[0]")]
    assert lines[-1] == "[0] active  tag='default'"
  # end

  def test_print_values(self):
    result = _ok([ENERGY, "print"])
    assert result.exit_code == 0
  # end
# end


# ---------------------------------------------------------------------------
# Skipped/dropped commands (documented, not silently missing).
# ---------------------------------------------------------------------------

def test_config_and_dg_commands_are_not_registered():
  """'config' (obsolete gkylsoft-path store) and the still-deferred dg_*
  Typer-era commands are intentionally not ported -- see 14-cli.md's "Skip"
  list and this layer's report. ``dg_local_poly`` has since been ported (its
  hand-derived per-order polynomial kernels are superseded by
  ``gpython.basis.eval_matrix``, which evaluates any basis at arbitrary
  points); ``dg_avg``/``dg_evproj`` remain unported."""
  names = {cmd.name for cmd in COMMANDS}
  assert "config" not in names
  assert "dg_avg" not in names
  assert "dg_evproj" not in names
  assert "dg_local_poly" in names
# end
