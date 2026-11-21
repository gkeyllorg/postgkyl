"""Normalized poloidal-flux coordinate mapping."""

import importlib
import numpy as np
import pytest

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.operations import map_interp
from postgkyl import gpython

map_module = importlib.import_module("postgkyl.operations.map_interp")


@pytest.mark.parametrize("source,override,expected", [
    ("scan-run-ion_M0_0.gkyl", None, "scan-run-psi.gkyl"),
    ("scan-run_b3-ion_M0_0.gkyl", None, "scan-run-psi.gkyl"),
    ("scan-run-ion_M0_0.gkyl", "override.gkyl", "override.gkyl"),
])
def test_psi_source_discovery(tmp_path, monkeypatch, source, override,
                              expected):
  seen = []

  def fake_read(path, ctx):
    seen.append(path)
    ctx.update(geometry_type=1, tokamak_geo_type=20)
    return [], None

  monkeypatch.setattr(map_module, "read", fake_read)
  map_module._psi_source(str(tmp_path / source),
                         str(tmp_path / override) if override else None)
  assert seen == [str(tmp_path / expected)]


@pytest.mark.parametrize("coord", ["rho_pol", "R_omp", "R_omp_R_sep"])
def test_map_interp_does_not_use_legacy_psi_file(tmp_path, coord):
  (tmp_path / "eq-psi.gkyl").touch()
  data = GDataState(ctx={"interpolated": True, "geqdsk_file": "eq.geqdsk"})
  data._file_name = str(tmp_path / "run_b3-ion_M0_0.gkyl")
  data.push([np.array([0., 1.])], np.ones((1, 1)))

  with pytest.raises(FileNotFoundError, match="run-psi\\.gkyl"):
    map_interp(data, axes=0, dest=coord)


@pytest.mark.parametrize("explicit", [False, True])
def test_psi_source_does_not_fallback_after_read_error(tmp_path, monkeypatch,
                                                       explicit):
  preferred = tmp_path / "run-psi.gkyl"
  preferred.touch()
  (tmp_path / "eq-psi.gkyl").touch()
  seen = []

  def fail_read(path, ctx):
    seen.append(path)
    raise OSError("unreadable psi")

  monkeypatch.setattr(map_module, "read", fail_read)
  with pytest.raises(OSError, match="unreadable psi"):
    map_module._psi_source(str(tmp_path / "run-ion_M0_0.gkyl"),
                           str(preferred) if explicit else None)
  assert seen == [str(preferred)]


def test_psi_source_needs_a_simulation_name():
  with pytest.raises(ValueError, match="source filename or psi_file"):
    map_module._psi_source("", None)


def test_rho_pol_maps_sample_positions_and_preserves_values(monkeypatch):
  values = np.arange(12.0).reshape(3, 4, 1)
  data = GDataState(ctx={"interpolated": True, "geqdsk_file": "eq.geqdsk"})
  data.push([np.array([2., 3., 6., 11.]), np.arange(5.)], values)
  monkeypatch.setattr(map_module, "_flux_reference", lambda *args: (1., 17.))

  result = map_interp(data, axes=0, dest="rho_pol")

  np.testing.assert_allclose(result.grid[0],
                             np.sqrt((np.array([2.5, 4.5, 8.5]) - 1.) / 16.))
  np.testing.assert_array_equal(result.grid[1], data.grid[1])
  np.testing.assert_array_equal(result.values, values)
  np.testing.assert_array_equal(data.grid[0], [2., 3., 6., 11.])
  assert result.ctx["coordinate_labels"][0] == r"$\rho_{\mathrm{pol}}$"


def test_flux_reference_uses_simulation_name_and_lcfs(monkeypatch):
  seen = []

  def fake_read(path, ctx):
    seen.append(path)
    ctx.update(geometry_type=1,
               tokamak_geo_type=20,
               simag=5.,
               psi_lcfs=1.,
               sibry=1.,
               psi_sep=0.)
    return [], None

  monkeypatch.setattr(map_module, "read", fake_read)
  assert map_module._flux_reference("run/sim-field_0.gkyl", None) == (5., 1.)
  assert seen == ["run/sim-psi.gkyl"]
  assert map_module._flux_reference("", "elsewhere/psi.gkyl") == (5., 1.)
  assert seen[-1] == "elsewhere/psi.gkyl"


def test_flux_reference_uses_separatrix_for_diverted_tokamak(monkeypatch):

  def fake_read(path, ctx):
    ctx.update(geometry_type=1,
               tokamak_geo_type=2,
               simag=0.,
               psi_sep=4.,
               sibry=3.)
    return [], None

  monkeypatch.setattr(map_module, "read", fake_read)
  assert map_module._flux_reference("", "psi.gkyl") == (0., 4.)


def test_flux_reference_requires_explicit_tokamak_subtype(monkeypatch):

  def fake_read(path, ctx):
    ctx.update(geometry_type=1, simag=0., psi_sep=4., sibry=3.)
    return [], None

  monkeypatch.setattr(map_module, "read", fake_read)
  with pytest.raises(ValueError, match="tokamak_geo_type"):
    map_module._flux_reference("", "old-psi.gkyl")


def test_flux_reference_rejects_invalid_axis_and_boundary(monkeypatch):
  metadata = {"geometry_type": 1, "tokamak_geo_type": 20}

  def fake_read(path, ctx):
    ctx.update(metadata)
    return [], None

  monkeypatch.setattr(map_module, "read", fake_read)
  with pytest.raises(ValueError, match="numeric simag and psi_lcfs"):
    map_module._flux_reference("", "psi.gkyl")
  metadata.update(simag=1., psi_lcfs=1.)
  with pytest.raises(ValueError, match="finite and distinct"):
    map_module._flux_reference("", "psi.gkyl")
  metadata["psi_lcfs"] = np.inf
  with pytest.raises(ValueError, match="finite and distinct"):
    map_module._flux_reference("", "psi.gkyl")


def test_outboard_radius_rejects_bad_psi_metadata(monkeypatch):
  ctx = dict(geometry_type=1,
             tokamak_geo_type=20,
             rmaxis=1.,
             zmaxis=0.,
             poly_order=1,
             basis_type="tensor",
             lower=[0., -1.],
             upper=[2., 1.],
             cells=[2, 2],
             psi_lcfs=0.5)
  grid = [np.array([0., 1., 2.]), np.array([-1., 0., 1.])]
  coeffs = np.zeros((2, 2, 4))
  monkeypatch.setattr(map_module, "_psi_source", lambda *args:
                      ("psi.gkyl", grid, coeffs, ctx))
  ctx["psi_lcfs"] = np.nan
  with pytest.raises(ValueError, match="finite psi_lcfs"):
    map_module._outboard_radius(np.array([0.2]),
                                "",
                                None,
                                relative_to_boundary=True)
  ctx["psi_lcfs"] = 0.5
  ctx["rmaxis"] = 3.
  with pytest.raises(ValueError, match="axis inside the grid"):
    map_module._outboard_radius(np.array([0.2]), "", None)
  ctx["rmaxis"] = 1.
  coeffs = np.zeros((1, 2, 4))
  with pytest.raises(ValueError, match="coefficients do not match"):
    map_module._outboard_radius(np.array([0.2]), "", None)


def test_map_interp_rejects_invalid_input_layout():
  grid = [np.array([0., 1.]), np.array([0., 1.])]
  values = np.ones((1, 1, 1))
  data = GDataState().push(grid, values)
  with pytest.raises(ValueError, match="integer axes"):
    map_interp(data, axes="bad", dest="rho_pol")
  with pytest.raises(ValueError, match="interpolated NumPy"):
    map_interp(data, axes=0, dest="rho_pol")
  data.ctx["interpolated"] = True
  with pytest.raises(ValueError, match="direction 1 in the data grid"):
    map_interp(GDataState(ctx={
        "interpolated": True
    }).push([grid[0]], values[:, 0]),
               axes=1,
               dest="rho_s")
  data.grid[1] = np.ones((2, 2))
  with pytest.raises(ValueError, match="separable coordinate axis"):
    map_interp(data, axes=1, dest="rho_s")
  data.grid[1] = np.array([0., 1.])
  with pytest.raises(ValueError, match="tokamak geometry metadata"):
    map_interp(data, axes=1, dest="rho_s")


def test_invalid_direction_coord_and_domain(monkeypatch):
  data = GDataState(ctx={"interpolated": True})
  data.push([np.array([0., 1.])], np.ones((1, 1)))
  monkeypatch.setattr(map_module, "_flux_reference", lambda *args: (2., 3.))
  with pytest.raises(ValueError, match="supports axes=0"):
    map_interp(data, axes=1, dest="rho_pol")
  with pytest.raises(ValueError, match="undefined"):
    map_interp(data, axes=0, dest="rho_pol")
  with pytest.raises(ValueError, match="supports axes=0"):
    map_interp(data, axes=0, source="alpha", dest="rho_pol")
  with pytest.raises(ValueError, match="one source per axis"):
    map_interp(data, axes=0, source="psi_pol,alpha", dest="rho_pol")


def test_rho_s_maps_binormal_samples_from_run_metadata():
  values = np.arange(12.).reshape(3, 4, 1)
  data = GDataState(ctx={
      "interpolated": True,
      "geometry_type": 1,
      "Cy": 0.2,
      "rho_s": 0.002
  })
  data.push([np.array([1., 2., 3., 4.]),
             np.array([-0.4, -0.2, 0., 0.2, 0.4])], values)

  mapped = map_interp(data, axes=1, dest="rho_s")
  np.testing.assert_allclose(mapped.grid[1], [-30., -10., 10., 30.])
  np.testing.assert_array_equal(mapped.grid[0], data.grid[0])
  np.testing.assert_array_equal(mapped.values, values)
  assert mapped.ctx["coordinate_labels"][1] == r"$y/\rho_s$"
  np.testing.assert_array_equal(data.grid[1], [-0.4, -0.2, 0., 0.2, 0.4])

  point_data = GDataState(ctx={
      "interpolated": True,
      "geometry_type": 1,
      "Cy": 0.2,
      "rho_s": 0.002
  })
  point_data.push(
      [np.array([1., 2.]), np.array([-0.1, 0.1])], np.ones((2, 2, 1)))
  np.testing.assert_allclose(
      map_interp(point_data, axes=1, dest="rho_s").grid[1], [-10., 10.])


@pytest.mark.parametrize("optional", [
    {
        "rho_s": "rho_s0",
        "Cy": "0.4"
    },
    {
        "rho_s": "rho_s0,Cy=0.4"
    },
    {
        "Cy": "0.4,rho_s=rho_s0"
    },
])
def test_rho_s_normalization_overrides(optional):
  data = GDataState(ctx={
      "interpolated": True,
      "geometry_type": 1,
      "Cy": 0.2,
      "rho_s0": 0.004
  })
  data.push([np.array([0., 1.]), np.array([-0.1, 0.1])], np.ones((1, 2, 1)))
  mapped = map_interp(data, axes=1, dest="rho_s", optional=optional)
  np.testing.assert_allclose(mapped.grid[1], [-10., 10.])
  assert "rho_s" not in mapped.ctx
  assert mapped.ctx["Cy"] == 0.2


def test_rho_s_accepts_numeric_normalization_and_cy_metadata_alias():
  data = GDataState(ctx={"interpolated": True, "geometry_type": 1, "Cy0": 0.22})
  data.push([np.array([0., 1.]), np.array([-1.1, 1.1])], np.ones((1, 2, 1)))
  mapped = map_interp(data,
                      axes=1,
                      dest="rho_s",
                      optional={
                          "rho_s": "1.1",
                          "Cy": "Cy0"
                      })
  np.testing.assert_allclose(mapped.grid[1], [-0.22, 0.22])


@pytest.mark.parametrize("optional,match", [
    ({
        "wrong": "1"
    }, "expects Cy or rho_s"),
    ({
        "rho_s": "rho_s0,Cy"
    }, "expects Cy or rho_s"),
    ({
        "rho_s": "rho_s0,Cy=0.2,Cy=0.3"
    }, "repeats Cy"),
    ({
        "rho_s": "missing"
    }, "numeric rho_s"),
    ({
        "rho_s": "0"
    }, "positive rho_s"),
])
def test_rho_s_rejects_invalid_overrides(optional, match):
  data = GDataState(ctx={"interpolated": True, "geometry_type": 1, "Cy": 0.2})
  data.push([np.array([0., 1.]), np.array([0., 1.])], np.ones((1, 1, 1)))
  with pytest.raises(ValueError, match=match):
    map_interp(data, axes=1, dest="rho_s", optional=optional)


def test_map_interp_maps_two_directions_in_one_result(monkeypatch):
  values = np.arange(12.).reshape(3, 4, 1)
  data = GDataState(ctx={
      "interpolated": True,
      "geometry_type": 1,
      "Cy": 0.2,
      "rho_s": 0.002
  })
  data.push([np.array([1., 2., 3., 4.]),
             np.array([-0.4, -0.2, 0., 0.2, 0.4])], values)
  monkeypatch.setattr(map_module, "_outboard_radius",
                      lambda samples, *args, **kwargs: (samples + 10., None))

  mapped = map_interp(data, axes="0,1", dest="R_omp,rho_s")
  explicit = map_interp(data,
                        axes="0,1",
                        source="psi_pol,alpha",
                        dest="R_omp,rho_s")
  np.testing.assert_allclose(explicit.grid[0], mapped.grid[0])
  np.testing.assert_allclose(explicit.grid[1], mapped.grid[1])
  np.testing.assert_allclose(mapped.grid[0], [11.5, 12.5, 13.5])
  np.testing.assert_allclose(mapped.grid[1], [-30., -10., 10., 30.])
  np.testing.assert_array_equal(mapped.values, values)
  assert mapped.ctx["coordinate_labels"] == {
      0: r"$R_{\mathrm{OMP}}$ (m)",
      1: r"$y/\rho_s$"
  }
  np.testing.assert_array_equal(data.grid[0], [1., 2., 3., 4.])
  np.testing.assert_allclose(
      map_interp(data, axes=[1, 0], dest=["rho_s", "R_omp"]).grid[0],
      mapped.grid[0])

  with pytest.raises(ValueError, match="one destination per axis"):
    map_interp(data, axes="0,1", dest="R_omp")
  with pytest.raises(ValueError, match="distinct"):
    map_interp(data, axes="0,0", dest="R_omp,rho_pol")


@pytest.mark.parametrize("metadata", [{
    "Cy": 0.2
}, {
    "rho_s": 0.002
}, {
    "Cy": 0.,
    "rho_s": 0.002
}, {
    "Cy": 0.2,
    "rho_s": -0.002
}])
def test_rho_s_requires_valid_metadata(metadata):
  data = GDataState(ctx={"interpolated": True, "geometry_type": 1, **metadata})
  data.push([np.array([0., 1.]), np.array([0., 1.])], np.ones((1, 1, 1)))
  with pytest.raises(ValueError,
                     match="numeric Cy|numeric rho_s|positive rho_s"):
    map_interp(data, axes=1, dest="rho_s")


@pytest.mark.skipif(not gpython.available(), reason="Gkeyll basis unavailable")
def test_r_omp_inverts_modal_psi_on_outboard_midplane(monkeypatch):
  """For psi=(R-1.2)^2+(Z-0.25)^2, R_omp=1.2+sqrt(psi)."""
  cells = np.array([4, 2])
  lower = np.array([0., -1.])
  upper = np.array([4., 1.])
  nodes = gpython.basis.node_coords("tensor", 2, 2)
  n2m = gpython.basis.nodal_to_modal_matrix("tensor", 2, 2)
  coeffs = np.empty((4, 2, len(nodes)))
  for i in range(4):
    for j in range(2):
      r = i + 0.5 + nodes[:, 0] / 2
      z = -0.5 + j + nodes[:, 1] / 2
      coeffs[i, j] = n2m @ ((r - 1.2)**2 + (z - 0.25)**2)
  ctx = dict(geometry_type=1,
             tokamak_geo_type=20,
             psi_lcfs=0.25,
             rmaxis=1.2,
             zmaxis=0.25,
             lower=lower,
             upper=upper,
             cells=cells,
             basis_type="tensor",
             poly_order=2,
             value_form="modal")
  grid = [np.linspace(0., 4., 5), np.linspace(-1., 1., 3)]
  monkeypatch.setattr(map_module, "_psi_source", lambda *args:
                      ("eq-psi.gkyl", grid, coeffs, ctx))

  data = GDataState(ctx={"interpolated": True})
  values = np.arange(3.).reshape(3, 1)
  data.push([np.array([0., 0.32, 0.96, 2.]), np.array([0.])], values)
  result = map_interp(data, axes=0, dest="R_omp")
  np.testing.assert_allclose(result.grid[0],
                             1.2 + np.sqrt([0.16, 0.64, 1.48]),
                             rtol=1e-12,
                             atol=1e-12)
  np.testing.assert_array_equal(result.values, values)
  assert result.ctx["coordinate_labels"][0] == r"$R_{\mathrm{OMP}}$ (m)"

  limited = map_interp(data, axes=0, dest="R_omp_R_sep")
  np.testing.assert_allclose(limited.grid[0],
                             np.sqrt([0.16, 0.64, 1.48]) - 0.5,
                             rtol=1e-12,
                             atol=1e-12)
  np.testing.assert_array_equal(limited.values, values)
  assert limited.ctx["coordinate_labels"][0] == \
      r"$R_{\mathrm{omp}} - R_{\mathrm{LCFS}}$ (m)"

  ctx["tokamak_geo_type"] = 2
  ctx["psi_sep"] = 0.36
  diverted = map_interp(data, axes=0, dest="R_omp_R_sep")
  np.testing.assert_allclose(diverted.grid[0],
                             np.sqrt([0.16, 0.64, 1.48]) - 0.6,
                             rtol=1e-12,
                             atol=1e-12)
  assert diverted.ctx["coordinate_labels"][0] == \
      r"$R_{\mathrm{omp}} - R_{\mathrm{sep}}$ (m)"
  del ctx["psi_sep"]
  with pytest.raises(ValueError, match="psi_sep"):
    map_interp(data, axes=0, dest="R_omp_R_sep")

  with pytest.raises(ValueError, match="No outboard R_omp root"):
    map_module._outboard_radius(np.array([9.]), "", "eq-psi.gkyl")

  # A nonmonotone outboard trace crosses psi=0.75 at R=1.5 and R=2.5.
  for i in range(4):
    for j in range(2):
      r = i + 0.5 + nodes[:, 0] / 2
      coeffs[i, j] = n2m @ (1. - (r - 2.)**2)
  np.testing.assert_allclose(map_module._outboard_radius(
      np.array([0.75]), "", "eq-psi.gkyl")[0], [1.5],
                             rtol=1e-12,
                             atol=1e-12)

  with pytest.raises(ValueError, match="finite computational psi"):
    map_module._outboard_radius(np.array([np.nan]), "", "eq-psi.gkyl")
  coeffs[:] = np.nan
  with pytest.raises(ValueError, match="non-finite psi"):
    map_module._outboard_radius(np.array([0.75]), "", "eq-psi.gkyl")

  del ctx["zmaxis"]
  with pytest.raises(ValueError, match="zmaxis"):
    map_module._outboard_radius(np.array([0.75]), "", "eq-psi.gkyl")
