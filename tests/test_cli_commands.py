"""Integration tests for the uniformly generated chained CLI."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from postgkyl.cli.app import (
    COMMAND_ALIASES, COMMANDS, COMMAND_SECTIONS, MODELS, cli,
)


ROOT = Path(__file__).parents[1]
DATA = ROOT / "tests" / "test_data"
FIELD = DATA / "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl"
DISTF = DATA / "generated" / "distf_p2_0.gkyl"
ENERGY = DATA / "generated" / "energy_dynvec.gkyl"
FIELD_3D = DATA / "generated" / "3d_ms_p1.gkyl"


def _run(*args):
  return CliRunner().invoke(cli, [str(arg) for arg in args])
# end


def _ok(*args):
  result = _run(*args)
  assert result.exit_code == 0, result.output
  return result
# end


def test_every_registered_subcommand_is_generated():
  assert len(COMMANDS) == len(MODELS)
  assert {command.name for command in COMMANDS} == {model.name for model in MODELS}
  assert set(cli.commands) == {model.name for model in MODELS}
# end


def test_help_groups_the_generated_inventory():
  result = _ok("--help")
  for section, names in COMMAND_SECTIONS.items():
    assert f"{section}:" in result.output
    assert names
  # end
  assert "rotations_bparrotate" in result.output
  assert "local_poly" in result.output
# end


def test_removed_manual_commands_and_legacy_spellings_are_rejected():
  for spelling in (
      "bparrotate", "euler", "tenmoment", "status", "print", "dg_local_poly",
      "gk-load-quantity", "extractinput", "plotly-animate",
  ):
    assert _run(spelling).exit_code != 0
  # end
# end


def test_aliases_only_add_spellings():
  assert dict(COMMAND_ALIASES) == {"pl": "plot", "ev": "evaluate"}
  assert cli.get_command(None, "pl") is cli.get_command(None, "plot")
  assert cli.get_command(None, "ev") is cli.get_command(None, "evaluate")
# end


def test_bare_filename_is_a_spelling_for_canonical_load():
  bare = _ok(FIELD, "info")
  explicit = _ok("load", "--file_name", FIELD, "info")
  assert bare.output == explicit.output
# end


def test_fluent_chain_uses_api_command_and_option_names():
  result = _ok(FIELD, "interpolate", "select", "--z0", "0", "--comp", "0",
      "info")
  assert "Number of components: 1" in result.output
# end


def test_select_gets_only_conflict_free_short_options():
  select = next(command for command in COMMANDS if command.name == "select")
  options = {option.name: option.opts for option in select.params}
  assert options == {
      "comp": ["--comp", "-c"],
      "z0": ["--z0"],
      "z1": ["--z1"],
      "z2": ["--z2"],
      "z3": ["--z3"],
      "z4": ["--z4"],
      "z5": ["--z5"],
      "inplace": ["--inplace", "-i"],
      "tag": ["--tag", "-t"],
      "label": ["--label", "-l"],
  }
  result = _ok(FIELD, "interpolate", "select", "-c", "0", "-t", "chosen",
      "info")
  assert result.output.startswith("(chosen#0)")
# end


def test_api_underscores_are_the_only_cli_spellings():
  assert _run(DISTF, "local-poly").exit_code != 0
  assert _run("load", "--file-name", FIELD, "info").exit_code != 0
  _ok(DISTF, "local_poly", "--npoints", "3", "info")
# end


def test_boolean_options_preserve_the_script_signature():
  assert _run(DISTF, "interpolate", "fft", "--psd").exit_code != 0
  _ok(DISTF, "interpolate", "fft", "--psd", "True", "info")
# end


def test_declared_cli_arguments_are_positional():
  _ok(ENERGY, "ev", "f0 2 *", "info")
  assert _run(ENERGY, "evaluate", "--chain", "f 2 *").exit_code != 0

  _ok(FIELD_3D, "integrate", "2", "info")
  assert _run(FIELD_3D, "integrate", "--axis", "2").exit_code != 0
  assert _run(FIELD_3D, "integrate_axis", "2").exit_code != 0
# end


def test_generated_save_options_match_python_parameter_names(tmp_path):
  output = tmp_path / "field"
  _ok(DISTF, "save", "--out_name", output, "--extension", "npy")
  assert _run(DISTF, "save", "--out-name", output).exit_code != 0
  assert output.with_suffix(".npy").is_file()
  assert _run(DISTF, "save", "--out", output).exit_code != 0
# end


def test_plot_uses_generated_render_options(tmp_path):
  output = tmp_path / "field.png"
  _ok(FIELD, "interpolate", "select", "--comp", "0", "plot", "--show",
      "False", "--grid_indices", "True", "--saveas", output)
  assert output.is_file()
# end


def test_manual_session_render_options_are_not_registered():
  assert _run("--batch_mode", FIELD, "info").exit_code != 0
  assert _run("--saveframes-prefix", "frame", FIELD, "info").exit_code != 0
# end


def test_version_and_unknown_command_edges():
  version = _ok("--version")
  assert "postgkyl" in version.output
  assert _run("definitely-not-a-command").exit_code != 0
# end
