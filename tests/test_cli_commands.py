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
  assert "rotations-bparrotate" in result.output
  assert "local-poly" in result.output
# end


def test_removed_manual_commands_and_legacy_spellings_are_rejected():
  for spelling in (
      "bparrotate", "euler", "tenmoment", "status", "print", "dg_local_poly",
      "gk_load_quantity", "extractinput", "plotly_animate",
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
  explicit = _ok("load", "--file-name", FIELD, "info")
  assert bare.output == explicit.output
# end


def test_fluent_chain_uses_dashed_command_and_option_names():
  result = _ok(FIELD, "interpolate", "select", "--z0", "0", "--comp", "0",
      "info")
  assert "Number of components: 1" in result.output
# end


def test_api_underscores_are_not_cli_spellings():
  assert _run(DISTF, "local_poly").exit_code != 0
  assert _run(DISTF, "local-poly", "--num_points", "3").exit_code != 0
  _ok(DISTF, "local-poly", "--npoints", "3", "info")
# end


def test_boolean_options_preserve_the_script_signature():
  assert _run(DISTF, "interpolate", "fft", "--psd").exit_code != 0
  _ok(DISTF, "interpolate", "fft", "--psd", "True", "info")
# end


def test_required_script_parameters_are_required_options():
  assert _run(ENERGY, "evaluate", "f 2 *").exit_code != 0
  _ok(ENERGY, "evaluate", "--chain", "f 2 *", "info")
# end


def test_generated_save_options_match_python_parameter_names(tmp_path):
  output = tmp_path / "field"
  _ok(DISTF, "save", "--out-name", output, "--extension", "npy")
  assert output.with_suffix(".npy").is_file()
  assert _run(DISTF, "save", "--out", output).exit_code != 0
# end


def test_plot_uses_generated_render_options(tmp_path):
  output = tmp_path / "field.png"
  _ok(FIELD, "interpolate", "select", "--comp", "0", "plot", "--show",
      "False", "--saveas", output)
  assert output.is_file()
# end


def test_manual_session_render_options_are_not_registered():
  assert _run("--batch-mode", FIELD, "info").exit_code != 0
  assert _run("--saveframes-prefix", "frame", FIELD, "info").exit_code != 0
# end


def test_version_and_unknown_command_edges():
  version = _ok("--version")
  assert "postgkyl" in version.output
  assert _run("definitely-not-a-command").exit_code != 0
# end
