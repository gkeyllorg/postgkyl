"""Tests for the single-file and glob forms of ``pg.load``."""

from __future__ import annotations

import importlib

import pytest

import postgkyl as pg


class _StubData(pg.GData):
  """A disk-free loaded dataset used to isolate filename dispatch."""

  def __init__(self, file_name="", **kwargs):
    super().__init__()
    self._file_name = str(file_name)
    self.load_kwargs = kwargs
  # end
# end


@pytest.fixture
def stub_load(monkeypatch):
  load_module = importlib.import_module("postgkyl.gdata.load")
  monkeypatch.setattr(load_module, "GData", _StubData)
  return load_module.load
# end


def test_literal_filename_returns_one_dataset(stub_load):
  out = stub_load("frame_0.gkyl", tag="moments")
  assert isinstance(out, _StubData)
  assert not isinstance(out, pg.GDataGroup)
  assert out.file_name == "frame_0.gkyl"
  assert out.load_kwargs["tag"] == "moments"
# end


def test_pathlike_literal_remains_supported(stub_load, tmp_path):
  out = stub_load(tmp_path / "frame_0.gkyl")
  assert isinstance(out, _StubData)
  assert out.file_name == str(tmp_path / "frame_0.gkyl")
# end


def test_glob_returns_group_in_natural_frame_order(stub_load, tmp_path):
  for frame in (10, 2, 1):
    (tmp_path / f"frame_{frame}.gkyl").touch()
  # end

  out = stub_load(str(tmp_path / "frame_*.gkyl"), label="series",
      basis_type="serendipity", poly_order=1)

  assert isinstance(out, pg.GDataGroup)
  assert [d.file_name for d in out] == [
      str(tmp_path / "frame_1.gkyl"),
      str(tmp_path / "frame_2.gkyl"),
      str(tmp_path / "frame_10.gkyl"),
  ]
  assert all(d.load_kwargs["label"] == "series" for d in out)
  assert all(d.load_kwargs["basis_type"] == "serendipity" for d in out)
  assert all(d.load_kwargs["poly_order"] == 1 for d in out)
# end


def test_single_match_glob_still_returns_group(stub_load, tmp_path):
  (tmp_path / "frame_0.gkyl").touch()
  out = stub_load(str(tmp_path / "frame_*.gkyl"))
  assert isinstance(out, pg.GDataGroup)
  assert len(out) == 1
# end


def test_unmatched_glob_has_a_targeted_error(stub_load, tmp_path):
  pattern = str(tmp_path / "missing_*.gkyl")
  with pytest.raises(FileNotFoundError, match="No files match pattern"):
    stub_load(pattern)
  # end
# end
