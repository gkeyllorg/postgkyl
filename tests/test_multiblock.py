"""Native multiblock support: identity, partition, and one-figure terminals.

Gkeyll writes a decomposed-domain run as one file per block,
``'<sim>_b<N>-<quantity>_<frame>.gkyl'``. Those files are *one field*, so
postgkyl must (a) recognize the block index without being told, (b) keep
verbs acting blockwise, and (c) have terminal verbs act on the field as a
whole -- one figure, one color scale, one colorbar.

The fixtures are the ``mb_sim_b{0,1,2}-elc_M0_{0,1}`` family written by
``generate_test_data.generate_all``: three blocks tiling the x axis into
abutting disjoint domains, two frames each.
"""

from __future__ import annotations

import glob
import os

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest
from click.testing import CliRunner

import postgkyl as pg
from postgkyl.cli.app import cli
from postgkyl.io import parse_output_name

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "tests", "test_data", "generated")
MB_GLOB = os.path.join(GEN, "mb_sim_b*-elc_M0_0.gkyl")
MB_GLOB_ALL_FRAMES = os.path.join(GEN, "mb_sim_b*-elc_M0_*.gkyl")


def _run(args):
  return CliRunner().invoke(cli, args)


def _ok(args):
  result = _run(args)
  assert result.exit_code == 0, result.output
  return result


def _blocks(frame: int = 0):
  return [
      pg.load(fn) for fn in sorted(
          glob.glob(os.path.join(GEN, f"mb_sim_b*-elc_M0_{frame}.gkyl")))
  ]


# ============================================================ the parser
class TestParseOutputName:

  @pytest.mark.parametrize(
      "name, sim, block, quantity, frame",
      [
          # The real file the convention was confirmed against.
          ("rt_gk_multib_sheath_1x2v_p1_b2-geo_int_B3.gkyl",
           "rt_gk_multib_sheath_1x2v_p1", 2, "geo_int_B3", None),
          ("sim_b10-elc_M0_7.gkyl", "sim", 10, "elc_M0", 7),
          ("gk_lorentzian_mirror-elc_M0_1.gkyl", "gk_lorentzian_mirror", None,
           "elc_M0", 1),
          ("sim-dt.gkyl", "sim", None, "dt", None),
      ])
  def test_parses_the_convention(self, name, sim, block, quantity, frame):
    parsed = parse_output_name(name)
    assert (parsed.sim, parsed.block, parsed.quantity,
            parsed.frame) == (sim, block, quantity, frame)

  def test_frame_requires_all_digits(self):
    # 'geo_int_B3' must not be read as quantity 'geo_int_B' at frame 3 --
    # the trailing run has to be digits only.
    assert parse_output_name("sim_b0-geo_int_B3.gkyl").frame is None

  def test_block_requires_digits_so_a_sim_named__b_is_safe(self):
    # A simulation legitimately named '..._beta' must not be read as block
    # 'eta': _b<N> needs digits.
    parsed = parse_output_name("gk_beta-elc_M0_0.gkyl")
    assert parsed.block is None
    assert parsed.sim == "gk_beta"

  def test_prefix_is_per_block_so_geometry_resolves_per_block(self):
    assert parse_output_name("d/sim_b2-elc_M0_3.gkyl").prefix == "d/sim_b2"
    assert parse_output_name("d/sim-elc_M0_3.gkyl").prefix == "d/sim"

  def test_restart_suffix_is_stripped(self):
    parsed = parse_output_name("sim-elc_5_restart.gkyl")
    assert (parsed.quantity, parsed.frame, parsed.restart) == ("elc", 5, True)

  def test_empty_path_has_no_identity(self):
    assert parse_output_name("") is None
    assert parse_output_name(None) is None


# ====================================================== identity in ctx
class TestBlockIdentityIsStamped:

  def test_load_stamps_sim_block_quantity_frame(self):
    data = pg.load(os.path.join(GEN, "mb_sim_b1-elc_M0_0.gkyl"))
    assert data.ctx["sim"] == "mb_sim"
    assert data.ctx["block"] == 1
    assert data.ctx["quantity"] == "elc_M0"
    assert data.ctx["frame"] == 0

  def test_single_block_data_has_block_none(self):
    data = pg.load(os.path.join(GEN, "distf_p2_0.gkyl"))
    assert data.ctx["block"] is None

  def test_identity_survives_verbs(self):
    # clone() copies ctx, so a family is still recognizable downstream --
    # this is what lets 'interp ... plot' still draw the blocks together.
    out = pg.load(os.path.join(GEN, "mb_sim_b2-elc_M0_1.gkyl")).interpolate()
    assert out.ctx["block"] == 2
    assert out.ctx["frame"] == 1

  def test_header_frame_wins_over_the_file_name(self):
    # The reader stamps frame from the file's msgpack metadata; the parsed
    # name only fills a gap, never overrides.
    data = pg.load(os.path.join(GEN, "mb_sim_b0-elc_M0_1.gkyl"))
    assert data.ctx["frame"] == 1

  def test_identity_is_not_written_into_saved_files(self, tmp_path):
    # The identity comes from the *path*, so it must never be stored in the
    # file: saving block 1's data under another name and reloading it would
    # otherwise find a stale block index in the header -- which, because
    # header metadata wins over the parsed name, would silently stick.
    data = pg.load(os.path.join(GEN, "mb_sim_b1-elc_M0_0.gkyl")).interpolate()
    out = pg.save(data, out_name=str(tmp_path / "plain-thing_0.gkyl"))
    reloaded = pg.load(out)
    assert reloaded.ctx["block"] is None
    assert reloaded.ctx["sim"] == "plain"
    assert reloaded.ctx["quantity"] == "thing"

  def test_info_reports_the_block(self):
    out = pg.load(os.path.join(GEN, "mb_sim_b1-elc_M0_0.gkyl")).info()
    assert "Block: 1" in out
    # The identity keys must not also fall through to info's generic ctx dump.
    assert "├─ block:" not in out
    assert "├─ sim:" not in out


# ========================================================== the partition
class TestGroupBlocks:

  def test_one_family_per_field(self):
    families = pg.group_blocks(_blocks())
    assert len(families) == 1
    assert [d.ctx["block"] for d in families[0]] == [0, 1, 2]

  def test_frames_are_separate_families(self):
    both = _blocks(0) + _blocks(1)
    families = pg.group_blocks(both)
    assert len(families) == 2
    assert {d.ctx["frame"] for d in families[0]} == {0}
    assert {d.ctx["frame"] for d in families[1]} == {1}

  def test_family_is_sorted_by_block_index(self):
    shuffled = list(reversed(_blocks()))
    assert [d.ctx["block"] for d in pg.group_blocks(shuffled)[0]] == [0, 1, 2]

  def test_single_block_data_is_all_singletons(self):
    # The property that keeps every pre-existing pipeline unchanged.
    frames = [pg.load(os.path.join(GEN, f"distf_p2_{i}.gkyl")) for i in (0, 1)]
    assert pg.group_blocks(frames) == [[frames[0]], [frames[1]]]

  def test_differently_tagged_results_do_not_merge(self):
    blocks = _blocks()
    tagged = [d.interpolate(tag="rz") for d in blocks]
    families = pg.group_blocks(blocks + tagged)
    assert len(families) == 2
    assert {d.tag for d in families[0]} == {"default"}
    assert {d.tag for d in families[1]} == {"rz"}


# ================================================= terminals: one figure
class TestOneFigurePerField:

  def test_group_plot_is_one_figure(self):
    group = pg.GDataGroup([d.interpolate() for d in _blocks()])
    fig = group.plot(no_show=True)
    assert isinstance(fig, matplotlib.figure.Figure)

  def test_operations_plot_takes_many_datasets(self):
    fig = pg.plot(*[d.interpolate() for d in _blocks()], no_show=True)
    assert isinstance(fig, matplotlib.figure.Figure)

  def test_blocks_share_one_color_scale_and_one_colorbar(self):
    # Each block's values are offset by its block index, so per-dataset
    # normalization would give three different scales for one field.
    blocks = [d.interpolate() for d in _blocks()]
    fig = pg.plot(*blocks, no_show=True)
    ax = fig.axes[0]
    meshes = ax.collections
    assert len(meshes) == 3
    clims = {m.get_clim() for m in meshes}
    assert len(clims) == 1, f"blocks drew on different color scales: {clims}"

    expected = (min(float(np.nanmin(d.values)) for d in blocks),
                max(float(np.nanmax(d.values)) for d in blocks))
    assert clims.pop() == pytest.approx(expected)

    # One colorbar for the panel, not one per block: the pcolormesh panel
    # plus a single appended colorbar axes.
    assert len(fig.axes) == 2

  def test_explicit_zlim_still_wins(self):
    blocks = [d.interpolate() for d in _blocks()]
    fig = pg.plot(*blocks, zmin=-1.0, zmax=1.0, no_show=True)
    for mesh in fig.axes[0].collections:
      assert mesh.get_clim() == pytest.approx((-1.0, 1.0))


# ================================== per-block geometry (gk_rz/gk_fluxsurf)
class TestPerBlockGeometry:

  def test_geometry_prefix_is_per_block(self):
    from postgkyl.diagnostics.gk import rz

    assert rz.geometry_prefix("d/sim_b2-elc_M0_3.gkyl") == "d/sim_b2"
    assert rz.geometry_prefix("d/sim-elc_M0_3.gkyl") == "d/sim"
    assert rz.geometry_prefix("") is None

  def test_explicit_geometry_path_substitutes_the_block_index(self):
    from postgkyl.diagnostics.gk.rz import per_block_path

    assert per_block_path("geo_b*.gkyl", 3) == "geo_b3.gkyl"
    assert per_block_path("geo.gkyl", 3) == "geo.gkyl"  # no '*' -> as given
    assert per_block_path("geo_b*.gkyl", None) == "geo_b*.gkyl"  # single block
    assert per_block_path(None, 3) is None

  def test_each_block_resolves_its_own_geometry(self, monkeypatch):
    # The bug this replaces: geometry was resolved once, from the first
    # dataset, and that one projection was applied to every block -- drawing
    # every block at block 0's position.
    from postgkyl.diagnostics.gk import rz

    seen = []
    monkeypatch.setattr(
        rz, "resolve_geometry",
        lambda file_name, **kw: seen.append(file_name) or file_name)
    monkeypatch.setattr(rz, "resolve_rz_projection", lambda first, geo, **kw:
                        ("projection", geo))

    blocks = _blocks(0) + _blocks(1)  # 3 blocks x 2 frames
    projections = rz.rz_projections(blocks)

    # One geometry read per block, not per dataset and not just one overall.
    assert len(seen) == 3
    assert set(projections) == {
        os.path.join(GEN, f"mb_sim_b{b}")
        for b in (0, 1, 2)
    }
    for data in blocks:
      assert rz.projection_for(
          projections, data) is projections[rz.geometry_prefix(data.file_name)]

  def test_interpolated_grid_values_is_idempotent(self):
    # 'pgkyl ... interp gk_rz' must not interpolate twice: the second pass
    # would run the DG evaluation matrix over values that are already point
    # values, silently producing garbage instead of raising.
    from postgkyl.diagnostics.gk import utils

    raw = pg.load(os.path.join(GEN, "mb_sim_b0-elc_M0_0.gkyl"))
    once = utils.interpolated_grid_values(raw)
    twice = utils.interpolated_grid_values(raw.interpolate())
    assert np.allclose(once[2], twice[2])


# ================================================================== CLI
class TestMultiblockCli:

  @staticmethod
  def _plot_calls(monkeypatch):
    """Record figures made by the one canonical plot call."""
    calls = []
    real = plt.figure

    def spy(*args, **kwargs):
      figure = real(*args, **kwargs)
      calls.append(figure)
      return figure

    monkeypatch.setattr(plt, "figure", spy)
    return calls

  def test_plot_draws_all_blocks_on_one_figure(self, monkeypatch):
    calls = self._plot_calls(monkeypatch)
    _ok([MB_GLOB, "interp", "plot", "--no_show"])
    assert len(calls) == 1
    assert len(calls[0].axes[0].collections) == 3

  def test_two_frames_give_two_figures(self, monkeypatch):
    calls = self._plot_calls(monkeypatch)
    _ok([MB_GLOB_ALL_FRAMES, "interp", "plot", "--no_show"])
    assert len(calls) == 2
    assert all(len(figure.axes[0].collections) == 3 for figure in calls)

  def test_single_block_data_still_gets_a_figure_per_dataset(self, monkeypatch):
    calls = self._plot_calls(monkeypatch)
    _ok([os.path.join(GEN, "distf_p2_*.gkyl"), "interp", "plot", "--no_show"])
    assert len(calls) == 2

  def test_multiblock_flag_forces_everything_onto_one_figure(self, monkeypatch):
    calls = self._plot_calls(monkeypatch)
    _ok([MB_GLOB_ALL_FRAMES, "interp", "plot", "--multiblock", "--no_show"])
    assert len(calls) == 1
    assert len(calls[0].axes[0].collections) == 6

  def test_load_orders_blocks_naturally(self):
    # A lexicographic sort would put _b10 before _b2; the working set must
    # be in block order.
    result = _ok([MB_GLOB, "info"])
    blocks = [
        int(line.split(":")[1].split("(")[0])
        for line in result.output.splitlines() if line.startswith("├─ Block:")
    ]
    assert blocks == [0, 1, 2]

  def test_explicit_render_options_save_one_png_for_the_field(self, tmp_path):
    out = tmp_path / "mb"
    _ok([MB_GLOB, "interp", "plot", "--no_show", "--saveas", str(out)])
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["mb.png"]
