"""Info distinguishes source-file facts from the assumptions used to load them."""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import postgkyl as pg
from postgkyl import gpython, io
from postgkyl.cli.app import cli
from generate_test_data import write_gkyl_field


@pytest.fixture(params=[io.GkylReader, io.GkylCReader])
def reader(request, monkeypatch):
  if request.param is io.GkylCReader and not gpython.available():
    pytest.skip("compiled Gkeyll unavailable")
  monkeypatch.setattr(io, "_READERS", {"test": request.param})


@pytest.fixture
def source(tmp_path):

  def write(*, no_metadata=False, metadata=None):
    path = tmp_path / "sim-field_7.gkyl"
    write_gkyl_field(path, [2], [-1.0], [1.0],
                     np.ones((2, 2)),
                     poly_order=1,
                     basis_type="serendipity",
                     metadata=metadata,
                     no_metadata=no_metadata)
    return str(path)

  return write


@pytest.mark.parametrize("options, verbose", [([], False), (["--all"], True),
                                              (["--all", "False"], False)])
def test_info_separates_file_metadata_and_modal_assumption(
    reader, source, options, verbose):
  result = CliRunner().invoke(cli, [source(), "info", *options])
  assert result.exit_code == 0, result.output
  assert "DG: serendipity p1 (modal)" in result.output
  for detail in ("Metadata sources", "File header", "cells: array",
                 "File metadata (verbatim keys)", "basisType: 'serendipity'",
                 "polyOrder: 1", "Inferred from filename"):
    assert (detail in result.output) == verbose
  assert "Inferred/defaulted" in result.output
  assert "value_form: 'modal' (basis specified; value_form absent)" in result.output
  assert "frame: 7" not in result.output  # header's frame=0 wins


def test_missing_metadata_reports_each_fallback(reader, source):
  with pytest.warns(UserWarning, match="not resolvable"):
    data = pg.load(source(no_metadata=True))
  output = data.info()
  assert "File metadata" not in output
  assert "File metadata (verbatim keys): <none>" in data.info(all=True)
  for setting in ("basis_type: 'serendipity'", "poly_order: 0",
                  "value_form: 'nodal'"):
    assert setting + " (not specified; spatial data fallback)" in output
  assert data.ctx["frame"] == 7
  assert "Frame: 7" in output
  assert "frame: 7" not in output


def test_overrides_preserve_original_metadata(reader, source):
  data = pg.load(source(metadata={"value_form": "modal"}),
                 basis_type="tensor",
                 poly_order=0,
                 value_form="nodal")
  output = data.info(all=True)
  raw = data.ctx["_load_metadata"]["file_metadata"]
  assert raw["basisType"] == "serendipity"
  assert raw["polyOrder"] == 1
  assert raw["value_form"] == "modal"
  assert "Explicit load overrides" in output
  assert "basis_type: 'tensor'" in output
  assert "value_form: 'nodal'" in output
  assert "Inferred/defaulted" not in output


def test_explicit_representation_is_not_reported_as_default(reader, source):
  data = pg.load(source(metadata={"value_form": "nodal"}))
  output = data.info()
  assert "DG: serendipity p1 (nodal)" in output
  assert "value_form:" not in output
  assert "Metadata sources" not in output
  assert "Inferred/defaulted" not in output
  assert "Explicit load overrides" not in output


def test_context_is_separate_from_file_and_defaults(reader, source):
  data = pg.load(source(), ctx={"value_form": "nodal", "custom": 3})
  output = data.info(all=True)
  assert "Explicit initial context" in output
  assert "Inferred/defaulted" not in output
  assert "value_form" not in data.ctx["_load_metadata"]["file_metadata"]


@pytest.mark.parametrize("verbose", [False, True])
def test_info_option_reaches_each_dataset_through_public_api(source, verbose):
  data = pg.load(source())
  expected = data.info(all=verbose, no_header=True)
  assert pg.info(data, data, all=verbose,
                 no_header=True) == [expected, expected]
  group = pg.GDataGroup([data, data])
  assert group.info(all=verbose, no_header=True) == [expected, expected]
  assert "default#0" not in expected
  assert ("File metadata" in expected) == verbose


def test_quad_summary_does_not_repeat_explicit_metadata(reader, source):
  data = pg.load(source(metadata={"value_form": "quad", "num_quad": 2}))
  output = data.info()
  assert "DG: serendipity p1 (quad, num_quad=2)" in output
  for detail in ("Metadata sources", "Inferred/defaulted", "basisType",
                 "polyOrder", "value_form:", "num_quad:"):
    assert detail not in output


def test_partial_load_preserves_original_domain(source):
  data = pg.load(source(), z0="0:1", component="0")
  assert data.values.shape == (1, 1)
  header = data.ctx["_load_metadata"]["file_header"]
  assert header["cells"].tolist() == [2]
  assert header["upper"].tolist() == [1.0]
  assert header["num_comps"] == 2


def test_reload_discards_previous_source_snapshot(reader, source):
  data = pg.load(source(), value_form="nodal")
  data.load(source(metadata={"value_form": "modal"}))
  assert data.ctx["_load_metadata"]["overrides"] == {}
  assert data.ctx["_load_metadata"]["context"] == {}
  assert "Inferred/defaulted" not in data.info()


@pytest.mark.skipif(not gpython.available(),
                    reason="compiled Gkeyll unavailable")
def test_conversion_keeps_load_snapshot_but_save_does_not_write_it(
    source, tmp_path):
  data = pg.load(source()).represent(to="nodal")
  output = data.info()
  assert "DG: serendipity p1 (nodal)" in output
  assert "value_form: 'modal' (basis specified; value_form absent)" in output
  assert "at load (summary above is current state)" in output
  path = data.save(str(tmp_path / "saved.gkyl"))
  raw = pg.load(path).ctx["_load_metadata"]["file_metadata"]
  assert "_load_metadata" not in raw


def test_dynvector_has_no_spatial_defaults():
  path = Path(__file__).parent / "test_data/generated/energy_dynvec.gkyl"
  data = pg.load(str(path))
  assert "Inferred/defaulted" not in data.info()


def test_warning_only_reports_defaults_that_were_applied(reader, source):
  with pytest.warns(UserWarning) as caught:
    data = pg.load(source(no_metadata=True), basis_type="tensor")
  message = str(caught[0].message)
  assert "defaulting to poly_order=0" in message
  assert "basis_type='serendipity'" not in message
  defaults = data.ctx["_load_metadata"]["defaults"]
  assert set(defaults) == {"poly_order", "value_form"}
  assert defaults["value_form"][0] == "modal"
