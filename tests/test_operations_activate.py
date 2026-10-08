"""``activate``: keep datasets by tag (the CLI's working-set choice)."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
import numpy as np

import postgkyl as pg
from postgkyl.cli.app import cli

DATA = Path(__file__).parent / "test_data"
M0_3X = DATA / "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl"


def _tagged(*tags):
  return [pg.GData(tag=tag) for tag in tags]


def test_keeps_the_datasets_carrying_a_tag_in_order():
  a, w, b = _tagged("data", "weight", "data")
  assert pg.activate(a, w, b, tags=["data"]) == [a, b]
  assert pg.activate([a, w, b], tags=["weight", "data"]) == [a, w, b]


def test_without_tags_keeps_everything():
  datasets = _tagged("data", "weight")
  assert pg.activate(*datasets) == datasets
  assert pg.activate(*datasets, tags=[]) == datasets


def _printed(args):
  result = CliRunner().invoke(cli, [str(arg) for arg in args])
  assert result.exit_code == 0, result.output
  return result.output


def test_a_set_aside_weight_is_used_but_never_transformed_or_shown():
  weighted = pg.load(M0_3X).average([1], weight=pg.load(M0_3X))
  output = _printed([
      M0_3X, "--tag", "weight", M0_3X, "activate", "--tags", "default",
      "average", "--dims", "1", "--weight", "weight", "print"
  ])
  # Only the averaged data is printed: the weight was neither averaged nor
  # left in the working set.
  assert output == np.array2string(weighted.values.squeeze(),
                                   precision=16) + "\n"
