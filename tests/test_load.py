"""Tests for the single-file and glob forms of ``pg.load``."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import postgkyl as pg


class _StubData(pg.GData):
  """A disk-free loaded dataset used to isolate filename dispatch."""

  def __init__(self, file_name="", **kwargs):
    super().__init__()
    self._file_name = str(file_name)
    self.load_kwargs = kwargs


@pytest.fixture
def stub_load(monkeypatch):
  load_module = importlib.import_module("postgkyl.gdata.load")
  monkeypatch.setattr(load_module, "GData", _StubData)
  return load_module.load


def test_literal_filename_returns_one_dataset(stub_load):
  out = stub_load("frame_0.gkyl", tag="moments")
  assert isinstance(out, _StubData)
  assert not isinstance(out, pg.GDataGroup)
  assert out.file_name == "frame_0.gkyl"
  assert out.load_kwargs["tag"] == "moments"


def test_pathlike_literal_remains_supported(stub_load, tmp_path):
  out = stub_load(tmp_path / "frame_0.gkyl")
  assert isinstance(out, _StubData)
  assert out.file_name == str(tmp_path / "frame_0.gkyl")


def test_glob_returns_group_in_natural_frame_order(stub_load, tmp_path):
  for frame in (10, 2, 1):
    (tmp_path / f"frame_{frame}.gkyl").touch()

  out = stub_load(str(tmp_path / "frame_*.gkyl"),
                  label="series",
                  basis_type="serendipity",
                  poly_order=1)

  assert isinstance(out, pg.GDataGroup)
  assert [d.file_name for d in out] == [
      str(tmp_path / "frame_1.gkyl"),
      str(tmp_path / "frame_2.gkyl"),
      str(tmp_path / "frame_10.gkyl"),
  ]
  assert all(d.load_kwargs["label"] == "series" for d in out)
  assert all(d.load_kwargs["basis_type"] == "serendipity" for d in out)
  assert all(d.load_kwargs["poly_order"] == 1 for d in out)


def test_single_match_glob_still_returns_group(stub_load, tmp_path):
  (tmp_path / "frame_0.gkyl").touch()
  out = stub_load(str(tmp_path / "frame_*.gkyl"))
  assert isinstance(out, pg.GDataGroup)
  assert len(out) == 1


def test_unmatched_glob_has_a_targeted_error(stub_load, tmp_path):
  pattern = str(tmp_path / "missing_*.gkyl")
  with pytest.raises(FileNotFoundError, match="No files match pattern"):
    stub_load(pattern)


def test_group_load_appends_and_returns_the_same_group(stub_load):
  group = pg.GDataGroup()

  out = group.load("frame_0.gkyl", tag="moments").load("frame_1.gkyl")

  assert out is group
  assert [data.file_name for data in group] == ["frame_0.gkyl", "frame_1.gkyl"]
  assert group[0].load_kwargs["tag"] == "moments"


def test_group_load_appends_every_glob_match_in_natural_order(
    stub_load, tmp_path):
  for frame in (10, 2, 1):
    (tmp_path / f"frame_{frame}.gkyl").touch()

  group = pg.GDataGroup().load(str(tmp_path / "frame_*.gkyl"))

  assert [data.file_name for data in group] == [
      str(tmp_path / "frame_1.gkyl"),
      str(tmp_path / "frame_2.gkyl"),
      str(tmp_path / "frame_10.gkyl"),
  ]


def test_failed_group_load_keeps_existing_members(stub_load, tmp_path):
  group = pg.GDataGroup().load("frame_0.gkyl")

  with pytest.raises(FileNotFoundError, match="No files match pattern"):
    group.load(str(tmp_path / "missing_*.gkyl"))

  assert len(group) == 1
  assert group[0].file_name == "frame_0.gkyl"


def test_instance_load_mutates_and_returns_the_same_dataset(monkeypatch):
  calls = []

  def fake_read(file_name, ctx, **kwargs):
    calls.append((file_name, kwargs))
    ctx.update(cells=np.array([2]),
               basis_type="tensor",
               poly_order=1,
               value_form="modal",
               source="reader")
    return [np.linspace(0.0, 1.0, 3)], np.ones((2, 2))

  state_module = importlib.import_module("postgkyl.gdatastate.gdatastate")
  monkeypatch.setattr(state_module.io, "read", fake_read)

  data = pg.GData(tag="moments", label="ions", ctx={"seed": 7})
  out = data.load("frame_0.gkyl",
                  tag="loaded",
                  label="electrons",
                  basis_type="tensor",
                  poly_order=1,
                  value_form="modal",
                  z0=3)

  assert out is data
  assert data.file_name == "frame_0.gkyl"
  assert data.tag == "loaded"
  assert data.label == "electrons"
  assert data.ctx["seed"] == 7
  assert data.ctx["source"] == "reader"
  assert data.values.shape == (2, 2)
  assert calls == [("frame_0.gkyl", {
      "value_form": "modal",
      "basis_type": "tensor",
      "poly_order": 1,
      "z0": 3
  })]


def test_instance_reload_does_not_retain_old_file_metadata(monkeypatch):

  def fake_read(file_name, ctx, **kwargs):
    ctx.update(cells=np.array([1]),
               basis_type="serendipity",
               poly_order=0,
               value_form="nodal")
    if file_name == "first.gkyl":
      ctx["first_file_only"] = True
    return [np.array([0.0, 1.0])], np.ones((1, 1))

  state_module = importlib.import_module("postgkyl.gdatastate.gdatastate")
  monkeypatch.setattr(state_module.io, "read", fake_read)

  data = pg.GData().load("first.gkyl")
  data.load("second.gkyl")

  assert data.file_name == "second.gkyl"
  assert "first_file_only" not in data.ctx


def test_failed_instance_load_leaves_existing_dataset_unchanged(monkeypatch):
  data = pg.GData(ctx={"source": "memory"})
  data.push([np.array([0.0, 1.0])], np.array([[4.0]]))
  old_grid, old_values, old_ctx = data.grid, data.values, data.ctx

  state_module = importlib.import_module("postgkyl.gdatastate.gdatastate")

  def fail_read(*args, **kwargs):
    raise FileNotFoundError("missing")

  monkeypatch.setattr(state_module.io, "read", fail_read)
  with pytest.raises(FileNotFoundError, match="missing"):
    data.load("missing.gkyl")

  assert data.grid is old_grid
  assert data.values is old_values
  assert data.ctx is old_ctx
  assert data.file_name == ""


def test_instance_load_rejects_empty_names_and_globs():
  data = pg.GData()
  with pytest.raises(ValueError, match="non-empty filename"):
    data.load("")
  with pytest.raises(ValueError, match=r"pg\.load\(pattern\)"):
    data.load("frame_*.gkyl")
