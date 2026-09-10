"""Explicit R-Z mapping and Gkeyll geometry compositions."""

from __future__ import annotations

from dataclasses import replace
import os

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl import operations as mapping
from postgkyl.diagnostics.gk.geometry import resolve_geometry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
F1D = os.path.join(
    DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")
F2D = os.path.join(DATA, "gk_ltx_iwl_2x2v_p1-elc_M2par_10.gkyl")
F2D_GEO = os.path.join(DATA, "gk_ltx_iwl_2x2v_p1-geo_int_mapc2p.gkyl")
F3D = os.path.join(DATA, "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl")

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


@needs_gkeyll
def test_2d_mapping_reference_grid_and_values():
  mapped = pg.gk.rz(pg.load(F2D), nz_interp=2)
  assert [axis.shape for axis in mapped.grid] == [(33, 33), (33, 33)]
  assert mapped.values.shape == (32, 32, 1)
  np.testing.assert_allclose(mapped.values.flat[:5], [
      3.12774618e30, 3.15719364e30, 3.23307916e30, 3.18034806e30, 3.15422838e30
  ],
                             rtol=2e-9)


@needs_gkeyll
def test_3d_mapping_reference_and_fft_phase():
  data = pg.load(F3D)
  geometry = resolve_geometry(data.file_name)
  projection = mapping.resolve_rz_projection(data, geometry, nz_interp=2)
  at_zero = data.map_to_rz(projection=projection, phi_tor=0.0)
  at_quarter = mapping.map_to_rz(data, projection=projection, phi_tor=np.pi / 2)
  assert [axis.shape for axis in at_zero.grid] == [(97, 65), (97, 65)]
  assert at_zero.values.shape == (96, 64, 1)
  np.testing.assert_allclose(at_zero.values.flat[:5], [
      9.97245134e18, 9.71438453e18, 9.57553863e18, 9.50610482e18, 9.81316530e18
  ],
                             rtol=2e-9)
  assert not np.allclose(at_zero.values, at_quarter.values)


@needs_gkeyll
def test_missing_toroidal_geometry_and_incompatible_projection_fail_clearly():
  data = pg.load(F3D)
  coords = [np.array([0.0, 1.0])] * 3
  values = np.ones((2, 2, 2))
  no_phi = mapping.Geometry(coords=coords,
                            major_r=values,
                            vert_z=values,
                            phi=None,
                            corner=None)
  with pytest.raises(ValueError, match="no toroidal-angle component"):
    mapping.resolve_rz_projection(data, no_phi)

  geometry = resolve_geometry(data.file_name)
  projection = mapping.resolve_rz_projection(data, geometry, nz_interp=2)
  shifted = data.clone()
  shifted.grid[0] = shifted.grid[0] + 0.01
  with pytest.raises(ValueError, match="computational grid does not match"):
    mapping.map_to_rz(shifted, projection=projection)


@needs_gkeyll
def test_3d_projection_rejects_thin_and_zero_span_geometry():
  data = pg.load(F3D)
  geometry = resolve_geometry(data.file_name)

  thin = data.clone()
  thin.ctx["poly_order"] = 0
  thin._grid[1] = np.array([0.0, 1.0])
  with pytest.raises(ValueError, match="at least two interpolated y and z"):
    mapping.resolve_rz_projection(thin, geometry)

  zero_span = replace(geometry, phi=np.zeros_like(geometry.phi))
  with pytest.raises(ValueError, match="zero or non-finite"):
    mapping.resolve_rz_projection(data, zero_span)


@needs_gkeyll
def test_3d_projection_uses_corner_geometry_when_available():
  data = pg.load(F3D)
  geometry = resolve_geometry(data.file_name)
  corner_coords = [np.array(axis, copy=True) for axis in geometry.coords]
  dz = geometry.coords[2][-1] - geometry.coords[2][0]
  corner_coords[2] = np.array(
      [geometry.coords[2][0] - dz, geometry.coords[2][-1] + dz])
  corner_r = np.stack([geometry.major_r[..., 0], geometry.major_r[..., -1]],
                      axis=-1)
  corner_z = np.stack([geometry.vert_z[..., 0], geometry.vert_z[..., -1]],
                      axis=-1)
  with_corner = replace(geometry, corner=(corner_coords, corner_r, corner_z))
  projection = mapping.resolve_rz_projection(data, with_corner, nz_interp=2)
  assert projection.r.shape == projection.z.shape == (97, 65)


@needs_gkeyll
def test_reusable_rz_projection_validates_every_shape_contract():
  data_2d = pg.load(F2D)
  geometry_2d = resolve_geometry(data_2d.file_name)
  projection_2d = mapping.resolve_rz_projection(data_2d, geometry_2d)

  invalid_2d = [
      (replace(projection_2d, num_dims=4), "invalid dimensionality"),
      (replace(projection_2d, num_dims=3), "dimensionality does not match"),
      (replace(projection_2d,
               z=projection_2d.z[:, :-1]), "matching 2-D arrays"),
      (replace(projection_2d, r=projection_2d.r[:-1],
               z=projection_2d.z[:-1]), "expected grid shape"),
  ]
  for projection, message in invalid_2d:
    with pytest.raises(ValueError, match=message):
      mapping.map_to_rz(data_2d, projection=projection)

  data_3d = pg.load(F3D)
  geometry_3d = resolve_geometry(data_3d.file_name)
  projection_3d = mapping.resolve_rz_projection(data_3d,
                                                geometry_3d,
                                                nz_interp=2)
  invalid_3d = [
      (replace(projection_3d, zc=None), "metadata is incomplete"),
      (replace(projection_3d, box=0.0), "span must be finite and nonzero"),
      (replace(projection_3d, wind=projection_3d.wind[:-1]),
       "projection and data grid shapes differ"),
  ]
  for projection, message in invalid_3d:
    with pytest.raises(ValueError, match=message):
      mapping.map_to_rz(data_3d, projection=projection)


@needs_gkeyll
def test_state_propagation_projection_reuse_and_public_surfaces():

  class DerivedData(pg.GData):
    pass

  source = DerivedData(F2D, tag="source", label="original")
  original = source.values.copy()
  geometry = resolve_geometry(source.file_name)
  projection = mapping.resolve_rz_projection(source, geometry, nz_interp=2)
  first = mapping.map_to_rz(source,
                            projection=projection,
                            tag="rz",
                            label="mapped")
  second = mapping.map_to_rz(source.clone(), projection=projection)
  assert isinstance(first, DerivedData)
  assert first is not source
  assert first.file_name == source.file_name
  assert (first.tag, first.label, first.ctx["interpolated"]) == ("rz", "mapped",
                                                                 True)
  np.testing.assert_array_equal(source.values, original)
  np.testing.assert_allclose(first.values, second.values)

  fluent = source.map_to_rz(projection=projection)
  functional = pg.gk.rz(source, mapc2p=F2D_GEO, nz_interp=2)
  np.testing.assert_allclose(fluent.values, functional.values)
  assert pg.map_to_rz is mapping.map_to_rz is pg.GData.map_to_rz
  assert not hasattr(pg, "gk_rz")
  assert not hasattr(pg.GData, "gk_rz")

  inplace = source.clone()
  result = pg.gk.rz(inplace, mapc2p=F2D_GEO, nz_interp=2, inplace=True)
  assert result is inplace and result.ctx["interpolated"] is True


@needs_gkeyll
def test_comp_selects_an_explicit_physical_field():
  source = pg.load(F2D)
  multi = pg.GData(ctx={
      key: value
      for key, value in source.ctx.items() if key != "num_comps"
  })
  multi.push([axis.copy() for axis in source.grid],
             np.concatenate([source.values, 2.0 * source.values], axis=-1))
  multi._file_name = source.file_name
  first = pg.gk.rz(multi, comp=0, nz_interp=2)
  second = pg.gk.rz(multi, comp=1, nz_interp=2)
  np.testing.assert_allclose(second.values, 2.0 * first.values)


@needs_gkeyll
@pytest.mark.parametrize("num_dims", [2, 3])
def test_mapping_accepts_filename_free_fields_and_explicit_geometry(num_dims):
  axes = [np.linspace(0.0, 1.0, 5) for _ in range(num_dims)]
  data = pg.GData(ctx={
      "basis_type": "serendipity",
      "poly_order": 0,
      "value_form": "modal"
  })
  # The normalized p0 basis is 2**(-ndim/2); these coefficients represent 7.
  data.push(axes, np.full((4, ) * num_dims + (1, ), 7.0 * 2**(num_dims / 2)))
  mesh = np.meshgrid(*axes, indexing="ij")
  geometry = pg.Geometry(coords=axes,
                         major_r=2.0 + mesh[0],
                         vert_z=mesh[-1],
                         phi=2 * np.pi * mesh[1] if num_dims == 3 else None,
                         corner=None)
  projection = pg.resolve_rz_projection(data, geometry, nz_interp=2)
  mapped = data.map_to_rz(projection=projection, phi_tor=0.37)
  assert not data.file_name and not mapped.file_name
  assert mapped.backend == "numpy" and mapped.is_interpolated
  assert mapped.values.shape == (4, 8 if num_dims == 3 else 4, 1)
  np.testing.assert_allclose(mapped.values, 7.0)
  np.testing.assert_allclose(mapped.grid[0][:, 0], 2.0 + axes[0])
  np.testing.assert_allclose(mapped.grid[1][0],
                             np.linspace(0, 1, mapped.values.shape[1] + 1))
  if num_dims == 3:
    surface = pg.resolve_flux_surface_grid(data, geometry, nphi=5, nz_interp=2)
    result = data.extract_flux_surface(fs_grid=surface)
    assert result.values.shape == (5, 8, 1)
    np.testing.assert_allclose(result.values, 7.0)
