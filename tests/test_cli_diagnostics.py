"""Generated CLI coverage for equation-specific diagnostics."""

from __future__ import annotations

import click
from click.testing import CliRunner
import numpy as np
import pytest

import postgkyl as pg
from postgkyl.cli.app import cli, COMMANDS
from postgkyl.cli.state import DataSpace
from postgkyl import diagnostics
from postgkyl.diagnostics.gyrokinetics import distf
from postgkyl.gdata.gdata import GData


GRID1D = [np.array([0.0, 1.0])]
COMMAND_BY_NAME = {command.name: command for command in COMMANDS}


def _make(values, tag="default"):
  data = GData(tag=tag)
  data.push(GRID1D, np.asarray(values))
  return data
# end


def _invoke(name, datasets, **kwargs):
  command = COMMAND_BY_NAME[name]
  space = DataSpace(datasets=list(datasets))
  with click.Context(command, obj=space) as context:
    context.invoke(command, **kwargs)
  # end
  return space
# end


def test_diagnostics_follow_gkeyll_model_families():
  assert diagnostics.__all__ == [
      "gyrokinetics", "vlasov", "pkpm", "moments", "discovery"]
  assert diagnostics.moments.five_moment.__name__.endswith(
      ".moments.five_moment")
  assert diagnostics.moments.enstrophy.__name__.endswith(
      ".moments.enstrophy")
  assert diagnostics.vlasov.kinetic.__name__.endswith(".vlasov.kinetic")
  assert diagnostics.vlasov.trajectory.__name__.endswith(".vlasov.trajectory")
# end


def test_gyrokinetic_diagnostics_use_concise_python_names():
  functions = (
      diagnostics.gyrokinetics.energy_balance,
      diagnostics.gyrokinetics.nodes,
      diagnostics.gyrokinetics.particle_balance,
      diagnostics.gyrokinetics.load_distf,
      diagnostics.gyrokinetics.load_quantity,
  )
  assert tuple(function.__name__ for function in functions) == (
      "energy_balance", "nodes", "particle_balance", "load_distf",
      "load_quantity",
  )
  assert pg.gyrokinetics is diagnostics.gyrokinetics
  assert pg.gyrokinetics.available_quantities() == (
      diagnostics.gyrokinetics.available_quantities())
  for bare_name in ("load_distf", "load_quantity", "available_gk_quantities"):
    assert not hasattr(pg, bare_name)
  # end
  for old_name in (
      "gk_energy_balance", "gk_nodes", "gk_particle_balance",
      "load_gk_distf", "load_gk_quantity",
  ):
    assert not hasattr(diagnostics.gyrokinetics, old_name)
    assert not hasattr(pg, old_name)
  # end
# end


def test_only_canonical_diagnostic_names_are_registered():
  assert "rotations-bparrotate" in COMMAND_BY_NAME
  assert "five-moment-pressure" in COMMAND_BY_NAME
  assert "ten-moment-agyro" in COMMAND_BY_NAME
  assert "multispecies-energetics" in COMMAND_BY_NAME
  assert "kinetic-transform-frame" in COMMAND_BY_NAME
  assert "pkpm-laguerre-compose" in COMMAND_BY_NAME
  for name in (
      "gyrokinetics-energy-balance", "gyrokinetics-nodes",
      "gyrokinetics-particle-balance", "gyrokinetics-load-distf",
      "gyrokinetics-load-quantity",
  ):
    assert name in COMMAND_BY_NAME
  # end
  for old_name in (
      "bparrotate", "agyro", "energetics", "euler", "tenmoment", "mhd",
      "transform_frame", "laguerre_compose",
      "gyrokinetics-gk-energy-balance", "gyrokinetics-gk-nodes",
      "gyrokinetics-gk-particle-balance", "gyrokinetics-load-gk-distf",
      "gyrokinetics-load-gk-quantity",
  ):
    assert old_name not in COMMAND_BY_NAME
  # end
# end


def test_load_distf_frame_is_text_and_cli_accepts_all_frames(tmp_path,
    monkeypatch):
  command = COMMAND_BY_NAME["gyrokinetics-load-distf"]
  frame_option = next(option for option in command.params
      if option.name == "frame")
  assert isinstance(frame_option.type, click.types.StringParamType)

  calls = []

  def fake_load_distf_frame(*, frame, tag, **kwargs):
    calls.append(frame)
    data = GData(tag=tag, ctx={"frame": frame})
    data.push([np.array([0.0, 1.0])], np.array([[float(frame)]]))
    return data
  # end

  monkeypatch.setattr(distf, "_load_distf_frame", fake_load_distf_frame)
  for frame in (0, 2):
    (tmp_path / f"sim-ion_fdot_{frame}.gkyl").touch()
  # end
  monkeypatch.chdir(tmp_path)

  result = CliRunner().invoke(cli, [
      "gyrokinetics-load-distf", "--name", "sim", "--species", "ion",
      "--frame", ":", "--suffix", "fdot",
  ])
  assert result.exit_code == 0, result.output
  assert calls == [0, 2]
# end


def test_bparrotate_is_compiled_directly_from_the_script_callable():
  array = _make([[1.0, 0.0, 0.0]], tag="array")
  field = _make([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], tag="field")

  space = _invoke("rotations-bparrotate", [array, field],
      array="array", field="field", inplace=False, tag="parallel", label=None)

  assert len(space.datasets) == 1
  assert space.datasets[0].tag == "parallel"
  np.testing.assert_allclose(space.datasets[0].values, [[1.0, 0.0, 0.0]])
# end


def test_dataset_parameters_are_tag_options_with_exact_dashed_names():
  command = COMMAND_BY_NAME["rotations-bparrotate"]
  assert {option.opts[0] for option in command.params} == {
      "--array", "--field", "--inplace", "--tag", "--label",
  }
# end


def test_generated_map_diagnostic_uses_the_callable_signature():
  gamma = 5.0 / 3.0
  rho, velocity, pressure = 2.0, 0.5, 0.8
  energy = pressure / (gamma - 1.0) + 0.5 * rho * velocity**2
  moments = _make([[rho, rho * velocity, 0.0, 0.0, energy]])

  space = _invoke("five-moment-pressure", [moments], gas_gamma=gamma,
      num_moms=None, inplace=False, tag="pressure", label=None)

  assert len(space.datasets) == 1
  assert space.datasets[0].tag == "pressure"
  np.testing.assert_allclose(space.datasets[0].values, [[pressure]])
# end


def test_missing_dataset_tag_fails_closed():
  array = _make([[1.0, 0.0, 0.0]], tag="array")
  with pytest.raises(click.UsageError, match="field"):
    _invoke("rotations-bparrotate", [array], array="array", field="field",
        inplace=False, tag=None, label=None)
  # end
# end
