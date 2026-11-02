"""Explicit R-Z mapping and Gkeyll geometry compositions."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl import operations as mapping
from postgkyl.operations.geometry import resolve_geometry

from generate_test_data import TWISTED_SPANS, twisted_fields

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
F1D = os.path.join(
    DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")
F2D = os.path.join(DATA, "gk_ltx_iwl_2x2v_p1-elc_M2par_10.gkyl")
F2D_GEO = os.path.join(DATA, "gk_ltx_iwl_2x2v_p1-geo_int_mapc2p.gkyl")
F3D = os.path.join(DATA, "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl")
GENERATED = Path(DATA) / "generated"

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


@needs_gkeyll
def test_2d_mapping_reference_grid_and_values():
  mapped = pg.map_to_torus_rz(pg.load(F2D), nz_interp=2)
  assert [axis.shape for axis in mapped.grid] == [(33, 33), (33, 33)]
  assert mapped.values.shape == (32, 32, 1)
  np.testing.assert_allclose(mapped.values.flat[:5], [
      3.12774618e30, 3.15719364e30, 3.23307916e30, 3.18034806e30, 3.15422838e30
  ],
                             rtol=2e-9)


@needs_gkeyll
def test_3d_mapping_shapes_and_angle_dependence():
  data = pg.load(F3D)
  geometry = resolve_geometry(data.file_name)
  projection = mapping.resolve_rz_projection(data, geometry, nz_interp=2)
  at_zero = data.map_to_torus_rz(projection=projection, phi_tor=0.0)
  at_quarter = mapping.map_to_torus_rz(data,
                                       projection=projection,
                                       phi_tor=np.pi / 2)
  assert [axis.shape for axis in at_zero.grid] == [(97, 65), (97, 65)]
  assert at_zero.values.shape == (96, 64, 1)
  assert not np.allclose(at_zero.values, at_quarter.values)


@needs_gkeyll
def test_3d_mapping_reads_each_interpolation_point_at_its_own_angle():
  # With nz_interp=1 the slice samples the interpolated x and z centers. The
  # geometry puts binormal center y_k at phi_lo + (y_k - y_lo) / Ly * span,
  # so that angle must return the interpolated value at y_k exactly.
  data = pg.load(F3D)
  projection = mapping.resolve_rz_projection(data,
                                             resolve_geometry(data.file_name),
                                             nz_interp=1)
  values = data.interpolate().values[..., 0]
  y_lo, y_hi = data.grid[1][0], data.grid[1][-1]
  y_edges = np.linspace(y_lo, y_hi, values.shape[1] + 1)
  y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
  for ix, iz, k in ((0, 0, 3), (20, 10, 1), (60, 5, 30), (95, 31, 62)):
    angle = (projection.phi_lo[ix, iz] + (y_centers[k] - y_lo) /
             (y_hi - y_lo) * projection.span[ix, iz])
    mapped = data.map_to_torus_rz(projection=projection, phi_tor=angle)
    np.testing.assert_allclose(mapped.values[ix, iz, 0],
                               values[ix, k, iz],
                               rtol=1e-12)


@needs_gkeyll
@pytest.mark.parametrize("prefix", TWISTED_SPANS)
def test_3d_mapping_locates_every_angle_along_the_binormal_coordinate(prefix):
  # Exact p1 polynomials on a twisted geometry whose binormal domain fills
  # 1/4.19 of the torus, for either orientation of y. Geometry is known only
  # at interior Gauss nodes, like Gkeyll's, so every edge value extrapolates.
  data = pg.load(str(GENERATED / f"{prefix}-field_0.gkyl"))
  geometry = pg.resolve_geometry(data.file_name)
  projection = pg.resolve_rz_projection(data, geometry, nz_interp=4)
  zf = projection.zf
  # Samples within half a cell of the parallel ends, past the geometry nodes.
  assert np.count_nonzero(zf < geometry.coords[2][0]) == 2
  assert np.count_nonzero(zf > geometry.coords[2][-1]) == 2
  # p1 samples two subcell centers per radial cell.
  xs = (-1.0 + 0.5 * (np.arange(6) + 0.5))[:, None]

  for angle in (0.0, 0.9, -2.3):
    expected = twisted_fields(xs, zf, angle, TWISTED_SPANS[prefix])
    for comp in (0, 1):
      mapped = data.map_to_torus_rz(projection=projection,
                                    phi_tor=angle,
                                    comp=comp)
      np.testing.assert_allclose(mapped.values[..., 0],
                                 expected[comp],
                                 rtol=1e-13,
                                 atol=0)
      np.testing.assert_allclose(data.map_to_torus_rz(projection=projection,
                                                      phi_tor=angle + 2 * np.pi,
                                                      comp=comp).values,
                                 mapped.values,
                                 rtol=1e-13,
                                 atol=0)


@needs_gkeyll
def test_3d_mapping_pairs_a_geometry_written_on_another_grid_cell_by_cell():
  # This field spans y in +-0.0970 but its geometry file was written on
  # y in +-0.1076 by another run. The geometry's grid defines the mapping, so
  # geometry point k pairs with field point k: the angle the file gives
  # there, atan2(Y, X), reads the field's interpolated value at point k.
  # atan2 of the p1-projected X and Y is linear in y only to ~1e-4, which
  # bounds the agreement; the unpaired grids differ by order one.
  data = pg.load(F3D)
  geometry = resolve_geometry(data.file_name)
  np.testing.assert_allclose(geometry.domain[1], (-0.10763, 0.10763), atol=1e-5)
  xyz = pg.load(F3D.replace("elc_M0_5", "geo_int_mapc2p")).interpolate().values
  values = data.interpolate().values[..., 0]
  projection = mapping.resolve_rz_projection(data, geometry, nz_interp=1)
  for ix, k, iz in ((0, 3, 0), (20, 40, 10), (60, 17, 5), (95, 62, 31)):
    angle = np.arctan2(xyz[ix, k, iz, 1], xyz[ix, k, iz, 0])
    mapped = data.map_to_torus_rz(projection=projection, phi_tor=angle)
    np.testing.assert_allclose(mapped.values[ix, iz, 0],
                               values[ix, k, iz],
                               rtol=1e-5)


@needs_gkeyll
def test_a_geometry_maps_the_same_on_any_computational_range():
  # Re-expressing the twisted geometry on other ranges (its own grid) must
  # place the data identically: the projection pairs the two grids.
  data = pg.load(str(GENERATED / "twisted_pos_3d_p1-field_0.gkyl"))
  geometry = pg.resolve_geometry(data.file_name)
  ranges = ((0.0, 1.0), (-1.0, 1.0), (10.0, 30.0))
  moved = replace(
      geometry,
      coords=[
          new_lo + (axis - lo) / (hi - lo) * (new_hi - new_lo)
          for axis, (lo, hi), (
              new_lo, new_hi) in zip(geometry.coords, geometry.domain, ranges)
      ],
      domain=ranges)
  expected = pg.resolve_rz_projection(data, geometry, nz_interp=2)
  actual = pg.resolve_rz_projection(data, moved, nz_interp=2)
  for name in ("r", "z", "phi_lo", "span"):
    np.testing.assert_allclose(getattr(actual, name),
                               getattr(expected, name),
                               rtol=1e-12,
                               atol=1e-12)
  with pytest.raises(ValueError, match="increasing"):
    pg.resolve_rz_projection(data, replace(geometry, domain=ranges[:2]))


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
    mapping.map_to_torus_rz(shifted, projection=projection)


@needs_gkeyll
def test_3d_projection_rejects_zero_span_geometry():
  data = pg.load(F3D)
  geometry = resolve_geometry(data.file_name)
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
      mapping.map_to_torus_rz(data_2d, projection=projection)

  data_3d = pg.load(F3D)
  geometry_3d = resolve_geometry(data_3d.file_name)
  projection_3d = mapping.resolve_rz_projection(data_3d,
                                                geometry_3d,
                                                nz_interp=2)
  invalid_3d = [
      (replace(projection_3d, phi_lo=None), "metadata is incomplete"),
      (replace(projection_3d, span=np.zeros_like(projection_3d.span)),
       "span must be finite and nonzero"),
      (replace(projection_3d, phi_lo=projection_3d.phi_lo[:-1]),
       "projection and data grid shapes differ"),
  ]
  for projection, message in invalid_3d:
    with pytest.raises(ValueError, match=message):
      mapping.map_to_torus_rz(data_3d, projection=projection)


@needs_gkeyll
def test_state_propagation_projection_reuse_and_public_surfaces():

  class DerivedData(pg.GData):
    pass

  source = DerivedData(F2D, tag="source", label="original")
  original = source.values.copy()
  geometry = resolve_geometry(source.file_name)
  projection = mapping.resolve_rz_projection(source, geometry, nz_interp=2)
  first = mapping.map_to_torus_rz(source,
                                  projection=projection,
                                  tag="rz",
                                  label="mapped")
  second = mapping.map_to_torus_rz(source.clone(), projection=projection)
  assert isinstance(first, DerivedData)
  assert first is not source
  assert first.file_name == source.file_name
  assert (first.tag, first.label, first.ctx["interpolated"]) == ("rz", "mapped",
                                                                 True)
  np.testing.assert_array_equal(source.values, original)
  np.testing.assert_allclose(first.values, second.values)

  fluent = source.map_to_torus_rz(projection=projection)
  functional = pg.map_to_torus_rz(source, mapc2p=F2D_GEO, nz_interp=2)
  np.testing.assert_allclose(fluent.values, functional.values)
  assert pg.map_to_torus_rz is mapping.map_to_torus_rz is pg.GData.map_to_torus_rz
  assert not hasattr(pg, "gk_rz")
  assert not hasattr(pg.GData, "gk_rz")

  inplace = source.clone()
  result = pg.map_to_torus_rz(inplace,
                              mapc2p=F2D_GEO,
                              nz_interp=2,
                              inplace=True)
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
  first = pg.map_to_torus_rz(multi, comp=0, nz_interp=2)
  second = pg.map_to_torus_rz(multi, comp=1, nz_interp=2)
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
  mapped = data.map_to_torus_rz(projection=projection, phi_tor=0.37)
  assert not data.file_name and not mapped.file_name
  assert mapped.backend == "numpy" and mapped.is_interpolated
  assert mapped.values.shape == (4, 8 if num_dims == 3 else 4, 1)
  np.testing.assert_allclose(mapped.values, 7.0)
  np.testing.assert_allclose(mapped.grid[0][:, 0], 2.0 + axes[0])
  np.testing.assert_allclose(mapped.grid[1][0],
                             np.linspace(0, 1, mapped.values.shape[1] + 1))
  if num_dims == 3:
    surface = pg.resolve_phitheta_projection(data,
                                             geometry,
                                             nphi=5,
                                             nz_interp=2)
    result = data.map_to_torus_phitheta(projection=surface)
    assert result.values.shape == (5, 8, 1)
    np.testing.assert_allclose(result.values, 7.0)
