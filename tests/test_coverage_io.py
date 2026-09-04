"""Coverage-completing tests for the ``io`` leaf layer.

Golden-path loads in test_postgkyl.py / test_gpython_rio.py only exercise the
happy path of each reader (full, non-partial, version-1, real_type f8 field
reads). This file targets the edges: partial loads (``axes=``/``comp=``),
dynvector multi-chunk continuation, legacy version-0 / float32 files, ghost
cells, the reader-registry failure path, and every ``write()`` format.

Run:  PYTHONPATH=src pytest tests/test_coverage_io.py -v
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)  # dedup harmless across the shared test session

import matplotlib

matplotlib.use("Agg")

import postgkyl as pg  # noqa: E402
from postgkyl import gpython, io  # noqa: E402
from postgkyl.io import mapping, writer  # noqa: E402
from postgkyl.io.gkyl_reader import GkylReader  # noqa: E402
from postgkyl.io.gkyl_c_reader import GkylCReader  # noqa: E402

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

DATA = os.path.join(ROOT, "tests", "test_data")
F1 = os.path.join(
    DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")
F1D_SINGLE_RANGE = os.path.join(DATA, "generated",
                                "1d_ms_p1.gkyl")  # file_type 1
F2D = os.path.join(DATA, "generated", "2d_ms_p1.gkyl")

# ndim=1, 24 cells, 6 comps, split across 4 multi-ranges of 6 cells each
# (1-indexed [1,6] [7,12] [13,18] [19,24]) -- verified by direct header
# inspection; used below to force one whole range to be excluded on a
# partial (``axes=``) read.


# --------------------------------------------------------------------- io/__init__
def test_read_defaults_ctx_to_a_fresh_dict():
  grid, values = io.read(F1)  # ctx omitted entirely
  assert values is not None


def test_read_raises_when_no_reader_is_compatible(tmp_path):
  bogus = tmp_path / "not_a_gkyl_file.dat"
  bogus.write_bytes(b"nope, not a gkyl file")
  with pytest.raises(NameError, match="cannot be read"):
    io.read(str(bogus))


# --------------------------------------------------------------- gkyl_c_reader
@needs_gkeyll
def test_gkyl_c_reader_is_compatible_swallows_backend_errors(monkeypatch):

  def _raise(*a, **k):
    raise RuntimeError("simulated backend failure")

  monkeypatch.setattr(gpython.rio, "file_type", _raise)
  r = GkylCReader(F1, ctx={})
  assert r.is_compatible() is False


@needs_gkeyll
def test_gkyl_c_reader_declines_a_partial_load_request():
  r = GkylCReader(F1, ctx={}, axes=("0", None, None, None, None, None))
  assert r.is_compatible() is False


@needs_gkeyll
def test_gkyl_c_reader_rejects_cell_array_mismatch(monkeypatch):
  from postgkyl.gpython.array import GkylArray

  def _fake_read_field(path):
    return {
        "cells": np.array([10]),
        "lower": np.array([0.0]),
        "upper": np.array([1.0])
    }, GkylArray.alloc(1, 5)  # 5 != 10

  monkeypatch.setattr(gpython.rio, "read_field", _fake_read_field)
  r = GkylCReader(F1, ctx={})
  with pytest.raises(IOError, match="ghost-cell layout"):
    r.load()


# ------------------------------------------------------------------- mapping
def test_adjust_for_ghost_cells_shrinks_and_extends_bounds():
  lower = np.array([0.0])
  upper = np.array([10.0])
  cells = np.array([10])
  lo, up, c = mapping.adjust_for_ghost_cells(lower, upper, cells, (8, ))
  assert c[0] == 8
  dz = 1.0  # (10-0)/10
  assert lo[0] == pytest.approx(-1.0 * dz)
  assert up[0] == pytest.approx(10.0 + 1.0 * dz)


# -------------------------------------------------------------------- writer
def test_write_derives_out_name_from_source_file(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  a = pg.load(F1).interpolate().select(comp=0)
  a._file_name = "source.gkyl"
  out = a.save()  # out_name empty -> derived from _file_name
  assert out == "source_mod.gkyl" or out.endswith("_mod.gkyl")
  assert os.path.exists(out)


def test_write_appends_extension_when_missing(tmp_path):
  a = pg.load(F1).interpolate().select(comp=0)
  out = a.save(str(tmp_path / "no_ext"), extension="gkyl")
  assert out.endswith("no_ext.gkyl")
  assert os.path.exists(out)


def test_write_npy_and_txt_and_rejects_unknown_extension(tmp_path):
  a = pg.load(F1).interpolate().select(comp=0)
  npy_path = writer.save(a, out_name=str(tmp_path / "out.npy"), extension="npy")
  assert os.path.exists(npy_path)
  loaded = np.load(npy_path)
  np.testing.assert_allclose(loaded, np.asarray(a.values).squeeze())

  txt_path = writer.save(a, out_name=str(tmp_path / "out.txt"), extension="txt")
  assert os.path.exists(txt_path)
  with open(txt_path) as fh:
    lines = fh.readlines()
  assert len(lines) == int(np.prod(a.num_cells))

  with pytest.raises(ValueError, match="Unsupported"):
    writer.save(a, out_name=str(tmp_path / "out.bad"), extension="bad")


def test_write_txt_multidim_computes_row_major_strides(tmp_path):
  """``_write_txt``'s stride computation (``basis[d] = prod(cells[d+1:])``)
  only has a loop body for num_dims >= 2 -- a 1-D dataset skips it."""
  b = pg.load(F2D).interpolate().select(comp=0)
  txt_path = writer.save(b,
                         out_name=str(tmp_path / "out2d.txt"),
                         extension="txt")
  with open(txt_path) as fh:
    lines = fh.readlines()
  assert len(lines) == int(np.prod(b.num_cells))


# --------------------------------------------------------- writer / metadata
def test_write_gkyl_roundtrips_metadata_through_meta_blob(tmp_path):
  """DG poly order/basis type, physical params, and time/frame stamps read
  off ``F1`` must survive a write() -> reload() round trip, not just the
  raw field values."""
  a = pg.load(F1)
  out = a.save(str(tmp_path / "roundtrip.gkyl"), extension="gkyl")

  reloaded = GkylReader(out, ctx={})
  reloaded.preload()

  for key in ("poly_order", "basis_type", "time", "frame", "changeset",
              "builddate", "geometry_type", "Description"):
    assert key in reloaded.ctx, f"{key!r} missing after round trip"
    assert reloaded.ctx[key] == a.ctx[key]


def test_write_gkyl_roundtrips_custom_ctx_keys(tmp_path):
  """Any ctx key that isn't structural/session-only (not just the keys
  postgkyl special-cases) must be preserved verbatim."""
  a = pg.load(F1).interpolate().select(comp=0)
  a.ctx["charge"] = -1.0
  a.ctx["mass"] = 1837.0
  out = a.save(str(tmp_path / "custom_meta.gkyl"), extension="gkyl")

  reloaded = GkylReader(out, ctx={})
  reloaded.preload()
  assert reloaded.ctx["charge"] == -1.0
  assert reloaded.ctx["mass"] == 1837.0


def test_write_gkyl_with_no_extra_ctx_writes_zero_meta_size(tmp_path):
  """A dataset whose ctx carries only structural/session keys must produce
  the same zero-length meta blob the writer always emitted -- no spurious
  meta bytes for a dataset with nothing extra to say."""
  out_name = str(tmp_path / "no_meta.gkyl")
  writer._write_gkyl(out_name,
                     num_dims=1,
                     num_comps=1,
                     num_cells=[4],
                     lo=[0.0],
                     up=[4.0],
                     values=np.arange(4, dtype=np.float64),
                     ctx={
                         "cells": np.array([4]),
                         "lower": np.array([0.0]),
                         "upper": np.array([4.0]),
                         "grid_type": "uniform"
                     })

  meta_size = np.fromfile(out_name, dtype=np.dtype("i8"), count=1, offset=21)[0]
  assert meta_size == 0

  reloaded = GkylReader(out_name, ctx={})
  reloaded.preload()
  np.testing.assert_allclose(reloaded.cells, [4])


def test_build_meta_excludes_internal_keys_and_renames_dg_fields():
  ctx = {
      "cells": np.array([4]),
      "lower": np.array([0.0]),
      "upper": np.array([4.0]),
      "num_comps": 1,
      "num_dims": 1,
      "grid_type": "uniform",
      "value_form": "modal",
      "num_quad": 3,
      "interpolated": True,
      "var_names": ["f"],
      "poly_order": 2,
      "basis_type": "serendipity",
      "time": 0.5,
      "frame": 3,
  }
  meta = writer._build_meta(ctx)
  assert meta == {
      "polyOrder": 2,
      "basisType": "serendipity",
      "time": 0.5,
      "frame": 3
  }


def test_to_msgpack_safe_converts_numpy_scalars_and_arrays():
  assert writer._to_msgpack_safe(np.float64(1.5)) == 1.5
  assert isinstance(writer._to_msgpack_safe(np.float64(1.5)), float)
  assert writer._to_msgpack_safe(np.int64(3)) == 3
  assert isinstance(writer._to_msgpack_safe(np.int64(3)), int)
  assert writer._to_msgpack_safe(np.array([1.0, 2.0])) == [1.0, 2.0]
  assert writer._to_msgpack_safe("serendipity") == "serendipity"


# --------------------------------------------------------------- gkyl_reader
def test_is_compatible_false_for_wrong_magic_and_missing_file(tmp_path):
  bogus = tmp_path / "bad.gkyl"
  bogus.write_bytes(b"definitely-not-gkyl-magic-bytes")
  assert GkylReader(str(bogus), ctx={}).is_compatible() is False
  assert GkylReader("/no/such/file.gkyl", ctx={}).is_compatible() is False


def test_defaults_ctx_to_a_fresh_dict_when_omitted():
  r = GkylReader(F1, ctx=None)
  assert r.ctx == {"grid_type": "uniform"}
  r.preload()
  r.load()


def test_partial_load_negative_stop_component():
  r = GkylReader(F1, ctx={}, comp="0:-1")  # drop the last component
  r.preload()
  grid, data = r.load()
  assert data.shape[-1] == 5


def test_partial_load_on_a_single_range_file_defaults_lo_up_idx():
  """``_get_data``'s ``lo_idx is None``/``up_idx is None`` defaults are only
  reached for a file_type-1 (single-range) partial load -- type-3 multi-range
  reads always pass explicit lo/up idx from the stored range headers."""
  full = GkylReader(F1D_SINGLE_RANGE, ctx={})
  full.preload()
  _, full_data = full.load()

  r = GkylReader(F1D_SINGLE_RANGE,
                 ctx={},
                 axes=("0:2", None, None, None, None, None))
  r.preload()
  grid, data = r.load()
  np.testing.assert_allclose(data, full_data[:2])


@needs_gkeyll
def test_partial_load_excludes_a_whole_multirange_and_slices_axis(tmp_path):
  """A real multi-range fixture (4 ranges of 6 cells); selecting exactly
  range 0 forces the other 3 ranges' data blocks to be empty, exercising
  the partial-load domain math, ``_get_block`` and the 'skip empty range'
  continuation in ``_read_t3_v1_data``."""
  full = GkylReader(F1, ctx={})
  full.preload()
  _, full_data = full.load()

  r = GkylReader(F1, ctx={}, axes=("0:6", None, None, None, None, None))
  r.preload()
  grid, data = r.load()
  assert data.shape == (6, 6)
  assert grid[0].shape == (7, )
  np.testing.assert_allclose(data, full_data[:6])


def test_partial_load_digit_axis_and_digit_component(tmp_path):
  r = GkylReader(F1, ctx={}, axes=("2", None, None, None, None, None), comp="1")
  r.preload()
  grid, data = r.load()
  assert data.shape == (1, 1)  # one cell, one component


def test_partial_load_negative_stop_and_colon_component():
  r = GkylReader(F1,
                 ctx={},
                 axes=("0:-2", None, None, None, None, None),
                 comp="0:3")
  r.preload()
  grid, data = r.load()
  assert data.shape[0] == 24 - 2
  assert data.shape[-1] == 3


@needs_gkeyll
def test_dynvec_single_chunk_round_trip_via_pure_python_reader(tmp_path):
  from postgkyl.gpython import rio
  path = str(tmp_path / "series.gkyl")
  time = np.array([0.0, 0.5, 1.0])
  values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  rio.write_dynvec(path, time, values)

  r = GkylReader(path, ctx={})
  r.preload()
  grid, data = r.load()
  np.testing.assert_allclose(grid[0], time)
  np.testing.assert_allclose(data, values)


@needs_gkeyll
def test_dynvec_multi_chunk_continuation(tmp_path):
  """Two dynvec writes concatenated back-to-back simulate the append
  pattern Gkeyll uses for a running time series -- the reader must loop
  back into ``_read_header`` for the second chunk without error."""
  from postgkyl.gpython import rio
  p1, p2 = str(tmp_path / "c1.gkyl"), str(tmp_path / "c2.gkyl")
  rio.write_dynvec(p1, np.array([0.0, 0.1]), np.array([[1.0, 2.0], [3.0, 4.0]]))
  rio.write_dynvec(p2, np.array([0.2, 0.3, 0.4]),
                   np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]))
  combo = tmp_path / "combo.gkyl"
  combo.write_bytes(open(p1, "rb").read() + open(p2, "rb").read())

  r = GkylReader(str(combo), ctx={})
  r.preload()
  grid, data = r.load()
  np.testing.assert_allclose(grid[0], [0.0, 0.1, 0.2, 0.3, 0.4])
  assert data.shape == (5, 2)


@needs_gkeyll
def test_dynvec_continuation_rejects_a_non_dynvec_second_chunk(tmp_path):
  from postgkyl.gpython import rio
  from postgkyl.gpython.array import GkylArray
  p1 = str(tmp_path / "c1.gkyl")
  rio.write_dynvec(p1, np.array([0.0, 0.1]), np.array([[1.0, 2.0], [3.0, 4.0]]))
  pf = str(tmp_path / "field.gkyl")
  rio.write_field(pf, {
      "lower": np.array([0.0]),
      "upper": np.array([1.0]),
      "cells": np.array([3])
  }, GkylArray.from_numpy(np.ones((3, 2))))
  bad = tmp_path / "bad_combo.gkyl"
  bad.write_bytes(open(p1, "rb").read() + open(pf, "rb").read())

  r = GkylReader(str(bad), ctx={})
  r.preload()
  with pytest.raises(TypeError, match="Inconsitent data"):
    r.load()


def _write_legacy_v0_field(path, cells, lower, upper, data, real_type=2):
  """Build a *version-0* raw field file: no gkyl0/version/type/meta header,
  just real_type + the type-1 domain fields + data -- the format predating
  the version-1 wrapper (see the module docstring in gkyl_reader.py)."""
  dti = np.dtype("i8")
  dtf = np.dtype("f4") if real_type == 1 else np.dtype("f8")
  doffset = 4 if real_type == 1 else 8
  ndim = len(cells)
  num_comps = data.shape[-1]
  with open(path, "wb") as fh:
    np.array([real_type], dtype=dti).tofile(fh)
    np.array([ndim], dtype=dti).tofile(fh)
    np.array(cells, dtype=dti).tofile(fh)
    np.array(lower, dtype=dtf).tofile(fh)
    np.array(upper, dtype=dtf).tofile(fh)
    np.array([num_comps * doffset], dtype=dti).tofile(fh)
    np.array([int(np.prod(cells))], dtype=dti).tofile(fh)
    np.array(data, dtype=dtf).tofile(fh)


def test_legacy_version0_file_is_read_via_default_version_and_type(tmp_path):
  path = str(tmp_path / "v0.gkyl")
  data = np.arange(8, dtype=np.float64).reshape(4, 2)
  _write_legacy_v0_field(path, [4], [0.0], [4.0], data)

  r = GkylReader(path, ctx={})
  assert r.is_compatible() is False  # no "gkyl0" magic in this legacy format
  r.preload()
  grid, out = r.load()
  assert r.version == 0
  np.testing.assert_allclose(grid[0], np.linspace(0.0, 4.0, 5))
  np.testing.assert_allclose(out, data)


def _write_v1_field(path, cells, lower, upper, data, real_type=2, meta=b""):
  dti = np.dtype("i8")
  dtf = np.dtype("f4") if real_type == 1 else np.dtype("f8")
  doffset = 4 if real_type == 1 else 8
  ndim = len(cells)
  num_comps = data.shape[-1]
  with open(path, "wb") as fh:
    np.array([103, 107, 121, 108, 48], dtype=np.dtype("b")).tofile(fh)
    np.array([1], dtype=dti).tofile(fh)
    np.array([1], dtype=dti).tofile(fh)
    np.array([len(meta)], dtype=dti).tofile(fh)
    fh.write(meta)
    np.array([real_type], dtype=dti).tofile(fh)
    np.array([ndim], dtype=dti).tofile(fh)
    np.array(cells, dtype=dti).tofile(fh)
    np.array(lower, dtype=dtf).tofile(fh)
    np.array(upper, dtype=dtf).tofile(fh)
    np.array([num_comps * doffset], dtype=dti).tofile(fh)
    np.array([int(np.prod(cells))], dtype=dti).tofile(fh)
    np.array(data, dtype=dtf).tofile(fh)


def test_single_precision_real_type_is_read_as_float32(tmp_path):
  path = str(tmp_path / "f4.gkyl")
  data = np.arange(6, dtype=np.float32).reshape(3, 2)
  _write_v1_field(path, [3], [0.0], [3.0], data, real_type=1)

  r = GkylReader(path, ctx={})
  assert r.is_compatible() is True
  r.preload()
  grid, out = r.load()
  assert r.dtf == np.dtype("f4")
  np.testing.assert_allclose(out, data)


def test_load_raises_for_an_unsupported_file_type(tmp_path):
  path = str(tmp_path / "v1.gkyl")
  _write_v1_field(path, [3], [0.0], [3.0], np.zeros((3, 1)))
  r = GkylReader(path, ctx={})
  r.preload()
  r.file_type = 99  # not 1, 2, or 3; version is 1 so the version==0 branch
  # doesn't rescue it either
  with pytest.raises(TypeError, match="not presently supported"):
    r.load()
