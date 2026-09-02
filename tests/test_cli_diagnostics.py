"""Generated CLI coverage for equation-specific diagnostics."""

from __future__ import annotations

import click
import numpy as np
import pytest

from postgkyl.cli.app import COMMANDS
from postgkyl.cli.state import DataSpace
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


def test_only_canonical_diagnostic_names_are_registered():
  assert "rotations-bparrotate" in COMMAND_BY_NAME
  assert "five-moment-pressure" in COMMAND_BY_NAME
  assert "ten-moment-agyro" in COMMAND_BY_NAME
  assert "multispecies-energetics" in COMMAND_BY_NAME
  assert "kinetic-transform-frame" in COMMAND_BY_NAME
  assert "pkpm-laguerre-compose" in COMMAND_BY_NAME
  for old_name in (
      "bparrotate", "agyro", "energetics", "euler", "tenmoment", "mhd",
      "transform_frame", "laguerre_compose",
  ):
    assert old_name not in COMMAND_BY_NAME
  # end
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
