import os

import click
import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

from postgkyl.data import GData, GInterpModal
from postgkyl.utils import verb_print
import postgkyl.utils.gk_utils as gku


def _file_prefix(file_name):
  if not file_name:
    return None
  return os.path.splitext(file_name)[0].rsplit("-", 1)[0]


def _mapc2p_geometry(path):
  """
  Interpolate a modal mapc2p (or geo R,Z,phi) file to physical R, Z, phi.
  """
  gdat = GData(path)
  if gku.is_gdata_geo_mapc2p(gdat):
    # Cartesian X, Y, Z: R = sqrt(X^2 + Y^2), phi = atan2(Y, X).
    grid, X = _interp(gdat, 0)
    _, Y = _interp(gdat, 1)
    _, Z = _interp(gdat, 2)
    return _centers(grid), np.sqrt(X**2 + Y**2), Z, np.arctan2(Y, X)

  # Components are directly R, Z, phi.
  grid, R = _interp(gdat, 0)
  _, Z = _interp(gdat, 1)
  phi = _interp(gdat, 2)[1] if R.ndim == 3 else None
  return _centers(grid), R, Z, phi


def _gauss_nodes(edges):
  """Coordinates of the p1 nodal points (cell center +/- h/(2*sqrt(3))) of a
  1D edge grid, i.e. where the values of a nodal geometry file live."""
  c = 0.5 * (edges[:-1] + edges[1:])
  off = np.diff(edges) / (2.0 * np.sqrt(3.0))
  return np.ravel(np.column_stack([c - off, c + off]))


def _nodes_geometry(path):
  """
  Read a nodal geometry file (pointwise node coordinates) to physical R, Z, phi.
  """
  gdat = GData(path)
  vals = gdat.get_values()
  # The stored grid is 2x-refined (one edge per value plus one); the values
  # themselves sit at the two p1 nodes of each cell, whose edges are every 
  # other stored grid point.
  coords = []
  for dim, g in enumerate(gdat.get_grid()):
    g = np.squeeze(g)
    if len(g) != vals.shape[dim] + 1 or vals.shape[dim] % 2:
      raise ValueError("Unrecognized nodal geometry layout in " + path)
    coords.append(_gauss_nodes(g[::2]))

  if gku.is_gdata_geo_mapc2p(gdat):
    X, Y, Z = vals[..., 0], vals[..., 1], vals[..., 2]
    return coords, np.sqrt(X**2 + Y**2), Z, np.arctan2(Y, X)

  R, Z = vals[..., 0], vals[..., 1]
  phi = vals[..., 2] if R.ndim == 3 else None
  return coords, R, Z, phi


def _corner_rz(path):
  """R and Z from a pointwise corner geometry file ('-geo_corn_nodes.gkyl').

  The corner file stores one value per cell corner, endpoints included, so
  unlike the interior nodes/mapc2p files it covers the domain boundary (in
  particular z = +/-pi, where the poloidal cross section closes). Returns
  (coords, R, Z) with 'coords' the corner lattice.
  """
  gdat = GData(path)
  vals = gdat.get_values()
  coords = [np.linspace(np.squeeze(g)[0], np.squeeze(g)[-1], n)
            for g, n in zip(gdat.get_grid(), vals.shape[:-1])]

  if gku.is_gdata_geo_mapc2p(gdat):
    X, Y, Z = vals[..., 0], vals[..., 1], vals[..., 2]
    return coords, np.sqrt(X**2 + Y**2), Z

  return coords, vals[..., 0], vals[..., 1]


def _interp(gdat, comp=0):
  """Interpolate component 'comp' of the DG GData object 'gdat'.

  Returns the computational grid (list of 1D node arrays) and the
  interpolated values at fine cell centers.
  """
  poly_order = gdat.ctx["poly_order"]
  basis_type = gdat.ctx["basis_type"]
  if basis_type == "serendipity":
    basis_type = "ms"

  grid, vals = GInterpModal(gdat, poly_order, basis_type).interpolate(comp)
  return [np.squeeze(g) for g in grid], np.squeeze(vals)


def _centers(nodes):
  """Cell centers from a list of 1D node arrays."""
  return [0.5 * (n[:-1] + n[1:]) for n in nodes]


def _sample(values, src_coords, dst_coords):
  """Linearly interpolate `values` onto the grid spanned by `dst_coords`."""
  mesh = np.meshgrid(*dst_coords, indexing="ij")
  return RegularGridInterpolator(
    tuple(src_coords), values, bounds_error=False, fill_value=None
  )(tuple(mesh))


def _fft_poloidal_project(vals, zc, box, wind, phi0_zf, zf, phi_tor):
  """Project a 3D field-aligned dataset onto the poloidal plane at phi = phi_tor.
  """
  Nx, Ny, Nz = vals.shape
  fk = np.fft.rfft(vals, axis=1, norm="forward")  # (Nx, K, Nz)
  K = fk.shape[1]
   # The following sign must match the sign convention in the shift.
  shift_sign = -1.0

  # Twist-and-shift reconnection: add a ghost z-cell at each domain edge (+/-pi)
  # whose value is the opposite end phase-shifted by exp(i k n0 wind).
  dz = zc[1] - zc[0]
  z_ex = np.concatenate(([zc[0] - dz / 2], zc, [zc[-1] + dz / 2]))
  fk_ex = np.zeros((Nx, K, Nz + 2), dtype=complex)
  fk_ex[:, :, 1:-1] = fk
  psh = (2.0 * np.pi / box) * wind  # per-mode phase = n0 * wind(x)
  for k in range(K):
    ph = np.exp(shift_sign * 1j * k * psh)
    fk_ex[:, k, -1] = 0.5 * (fk[:, k, -1] + ph * fk[:, k, 0])
    fk_ex[:, k, 0] = 0.5 * (fk[:, k, 0] + np.conj(ph) * fk[:, k, -1])

  # Up-sample along z (interpolate real and imaginary parts separately).
  fk_zf = (PchipInterpolator(z_ex, fk_ex.real, axis=2)(zf)
           + 1j * PchipInterpolator(z_ex, fk_ex.imag, axis=2)(zf))

  # Phase-sum: reconstruct the field where the physical toroidal angle == phi_tor.
  frac = (phi_tor - phi0_zf) / box
  out = np.zeros((Nx, len(zf)))
  for k in range(K):
    # rfft: modes 0 < k < Nyquist represent both +/-k; do not double k=0 or Nyquist.
    weight = 1.0 if (k == 0 or (Ny % 2 == 0 and k == K - 1)) else 2.0
    out += weight * np.real(fk_zf[:, k, :] * np.exp(shift_sign * 1j * 2.0 * np.pi * k * frac))
  return out


@click.command()
@click.option("--mapc2p", "-m", default=None, type=click.STRING,
  help="Use a modal mapc2p file as the geometry source instead of the default nodes file; "
       "pass '' to look up '<prefix>-geo_int_mapc2p.gkyl' from the first processed dataset's prefix.")
@click.option("--nodes", "-n", default=None, type=click.STRING,
  help="Path to a nodal geometry file, overriding the default '<prefix>-geo_int_nodes.gkyl' lookup.")
@click.option("--z-axis", "-z", default=0.0, type=click.FLOAT,
  help="Vertical position of the magnetic axis (m), added to the geometry Z."
       "mapc2p files store Z relative to the axis; pass Z_axis from the simulation input "
       "file to plot in machine coordinates. Default 0.")
@click.option("--use", "-u", default=None,
  help="Specify tag of datasets to process from the stack.")
@click.option("--tag", "-t", default="rz", type=click.STRING,
  help="Tag for output datasets.")
@click.option("--label", "-l", default=None, type=click.STRING,
  help="Custom label for the result.")
@click.option("--phi-tor", "-p", default=0.0, type=click.FLOAT,
  help="Toroidal angle (radians) of the poloidal plane to project 3D data onto. Default 0.")
@click.option("--nz-interp", default=8, type=click.INT,
  help="Parallel (z) up-sampling factor used to smooth the projected 3D surfaces. Default 8.")
@click.pass_context
def gk_rz(ctx, **kwargs):
  """
  \b
  Gyrokinetics: Interpolate DG dataset(s) and map them to the R-Z plane.
  Assumes DG data (not yet interpolated) has been loaded onto the stack by a
  preceding command.

  The geometry is automatically found from the prefix of the first processed
  dataset: the pointwise '<prefix>-geo_int_nodes.gkyl' is preferred (exact node
  coordinates, robust at coarse z resolution), falling back to the modal
  '<prefix>-geo_int_mapc2p.gkyl'. Use '-n path' to point at a specific nodes
  file, '-m path' at a specific mapc2p file, or "-m ''" to force the default
  mapc2p lookup.

  For 3D (field-aligned) data the field is reconstructed on the poloidal plane at
  toroidal angle --phi-tor (default 0) by interpolating along the binormal
  direction, up-sampled in z (--nz-interp) for smooth surfaces.
  """
  data = ctx.obj["data"]

  # Locate the geometry files from the prefix of the first processed dataset.
  first_data = next(data.iterator(kwargs["use"]), None)
  if first_data is None:
    return

  prefix = _file_prefix(getattr(first_data, "_file_name", None))

  # Geometry source: the pointwise nodes file by default (exact node values;
  # the modal mapc2p representation loses amplitude where the toroidal winding
  # is under-resolved), the modal mapc2p file on request or as fallback.
  mapc2p_opt = kwargs["mapc2p"]
  nodes_opt = kwargs["nodes"]
  if mapc2p_opt is not None and nodes_opt is not None:
    raise click.ClickException("Pass either --mapc2p or --nodes, not both.")

  if nodes_opt is not None:
    geo_path, geo_reader = nodes_opt, _nodes_geometry
  elif mapc2p_opt is not None:
    # An empty value requests the default '<prefix>-geo_int_mapc2p.gkyl'.
    geo_path = mapc2p_opt if mapc2p_opt else (
      prefix + "-geo_int_mapc2p.gkyl" if prefix is not None else None)
    geo_reader = _mapc2p_geometry
  elif prefix is not None:
    geo_path, geo_reader = prefix + "-geo_int_nodes.gkyl", _nodes_geometry
    if not os.path.exists(geo_path):
      geo_path, geo_reader = prefix + "-geo_int_mapc2p.gkyl", _mapc2p_geometry
  else:
    geo_path, geo_reader = None, None

  if geo_path is None or not os.path.exists(geo_path):
    raise click.ClickException(
      "Could not find a geometry file; pass it with -N/--nodes or -n/--mapc2p.")

  if first_data.get_num_dims() == 2:
    # Direct map onto R-Z using mapc2p.
    verb_print(ctx, "Mapping stack data to R-Z using " + geo_path)
    geo_coords, majorR, vertZ, _ = geo_reader(geo_path)
    vertZ = vertZ + kwargs["z_axis"]
    loaded_count = 0
    for dat in data.iterator(kwargs["use"]):
      field_grid, vals = _interp(dat)
      # Evaluate R, Z at the field cell corners (its node arrays) so pcolormesh
      # gets explicit cell edges, not non-monotonic curvilinear cell centers.
      R = _sample(majorR, geo_coords, field_grid)
      Z = _sample(vertZ, geo_coords, field_grid)
      out = GData(tag=kwargs["tag"], label=kwargs["label"], ctx=dat.ctx)
      # Used to allow 'select' after gk-rz.
      out.ctx["value_coords"] = list(field_grid)
      out.push([R, Z], vals[..., np.newaxis])
      data.add(out)
      dat.deactivate()
      loaded_count += 1
  
    if loaded_count > 1:
      data.set_unique_labels()
  
    verb_print(ctx, "Finishing R-Z mapping.")
    return

  # 3D: project onto the poloidal plane at phi_tor.
  phi_tor = kwargs["phi_tor"]
  nz_interp = max(1, kwargs["nz_interp"])

  # Field cell-center coordinates (from the field's own DG grid).
  fine_grid, _ = _interp(first_data)
  xc, yc, zc = _centers(fine_grid)
  Nx, Ny, Nz = xc.size, yc.size, zc.size

  verb_print(ctx, "3D data: projecting onto phi = %g rad using geometry %s"
                  % (phi_tor, geo_path))
  gx_gy_gz, majorR, vertZ, phi = geo_reader(geo_path)
  if phi is None:
    raise click.ClickException(
      "The geometry file has no toroidal-angle component; cannot project 3D data.")
  vertZ = vertZ + kwargs["z_axis"]
  gx, gy, gz = gx_gy_gz

  # Physical toroidal angle on the field grid, made continuous for interpolation.
  phi = np.unwrap(np.unwrap(np.unwrap(phi, axis=2), axis=1), axis=0)
  phiF = _sample(phi, [gx, gy, gz], [xc, yc, zc])
  # The binormal domain spans one 1/n0 toroidal sector (wedge).
  box = np.mean(np.diff(phiF[Nx // 2, :, Nz // 2])) * Ny
  n0 = max(1, int(round(abs(2.0 * np.pi / box))))
  box = np.sign(box) * 2.0 * np.pi / n0
  wind = phiF[:, 0, -1] - phiF[:, 0, 0]

  # Up-sampled z: edges (zf_edges) for the plotting grid, centers (zf) for the
  # field reconstruction.
  xn = fine_grid[0]
  zf_edges = np.linspace(fine_grid[2][0], fine_grid[2][-1], nz_interp * Nz + 1)
  zf = 0.5 * (zf_edges[:-1] + zf_edges[1:])
  # Toroidal angle at the binormal origin along (x, zf), used by the phase-sum.
  phi0_zf = np.array([np.interp(zf, zc, phiF[ix, 0, :]) for ix in range(Nx)])

  # R, Z (y-independent) on the (x, z) plane.
  R2d, Z2d, gz_rz = majorR[:, 0, :], vertZ[:, 0, :], gz
  corn_path = prefix + "-geo_corn_nodes.gkyl" if prefix is not None else None
  if corn_path is not None and os.path.exists(corn_path):
    verb_print(ctx, "Closing the theta = +/-pi ends with " + corn_path)
    ccoords, cornR, cornZ = _corner_rz(corn_path)
    cx, cz = ccoords[0], ccoords[2]
    cornR, cornZ = cornR[:, 0, :], cornZ[:, 0, :] + kwargs["z_axis"]
    R2d = np.concatenate([
      np.interp(gx, cx, cornR[:, 0])[:, None], R2d,
      np.interp(gx, cx, cornR[:, -1])[:, None]], axis=1)
    Z2d = np.concatenate([
      np.interp(gx, cx, cornZ[:, 0])[:, None], Z2d,
      np.interp(gx, cx, cornZ[:, -1])[:, None]], axis=1)
    gz_rz = np.concatenate([[cz[0]], gz, [cz[-1]]])

  # Interpolate the geometry.
  Rrz = _sample(R2d, [gx, gz_rz], [xn, zf_edges])
  Zrz = _sample(Z2d, [gx, gz_rz], [xn, zf_edges])

  loaded_count = 0
  for dat in data.iterator(kwargs["use"]):
    _, vals = _interp(dat)
    proj = _fft_poloidal_project(vals, zc, box, wind, phi0_zf, zf, phi_tor)
    out = GData(tag=kwargs["tag"], label=kwargs["label"], ctx=dat.ctx)
    # Rrz, Zrz are curvilinear; expose the logical (x, z) coordinates used to
    # build them so 'select' can search by value.
    out.ctx["value_coords"] = [xn, zf_edges]
    out.push([Rrz, Zrz], proj[..., np.newaxis])
    data.add(out)
    dat.deactivate()
    loaded_count += 1

  if loaded_count > 1:
    data.set_unique_labels()

  verb_print(ctx, "Finishing R-Z mapping.")
