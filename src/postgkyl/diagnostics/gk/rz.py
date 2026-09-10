"""Gkeyll geometry discovery composed with explicit rz mapping."""
from __future__ import annotations

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.operations.map import resolve_rz_projection, map_to_rz
from postgkyl.operations.geometry import RzProjection, validate_mapping_grid
from .geometry import geometry_prefix, per_block_path, resolve_geometry


def rz_projections(datasets,
                   *,
                   mapc2p: str | None = None,
                   nodes_file: str | None = None,
                   z_axis: float = 0.0,
                   nz_interp: int = 8) -> dict[str | None, RzProjection]:
  """Build one reusable R-Z projection per block geometry.

  Frames from the same block share a projection; distinct blocks resolve
  their own geometry. A ``'*'`` in an explicit geometry path is replaced by
  the dataset's block index.
  """
  projections: dict[str | None, RzProjection] = {}
  for data in datasets:
    key = geometry_prefix(data.file_name)
    if key in projections:
      validate_mapping_grid(data, projections[key].computational_grid)
      continue
    block = data.ctx.get("block")
    geometry = resolve_geometry(data.file_name,
                                mapc2p=per_block_path(mapc2p, block),
                                nodes_file=per_block_path(nodes_file, block))
    projections[key] = resolve_rz_projection(data,
                                             geometry,
                                             z_axis=z_axis,
                                             nz_interp=nz_interp)
  return projections


def projection_for(projections: dict[str | None, RzProjection],
                   data: "GDataState") -> RzProjection:
  """Return the projection belonging to ``data``'s block."""
  return projections[geometry_prefix(data.file_name)]


def rz(
    data: "GDataState",
    *,
    mapc2p: str | None = None,
    nodes_file: str | None = None,
    z_axis: float = 0.0,
    phi_tor: float = 0.0,
    nz_interp: int = 8,
    comp: int = 0,
    inplace: bool = False,
    tag: str | None = None,
    label: str | None = None,
) -> "GDataState":
  """Interpolate one DG component and project it onto a physical R-Z grid.

  ``data`` must be un-interpolated modal DG data with two computational
  dimensions, or three for a field-aligned reconstruction. Geometry is
  inferred from ``data.file_name``: the pointwise
  ``'<prefix>-geo_int_nodes.gkyl'`` file is preferred, falling back to
  ``'<prefix>-geo_int_mapc2p.gkyl'``. ``nodes_file`` or ``mapc2p`` may
  override that choice, but they are mutually exclusive; ``mapc2p=''``
  forces the inferred modal filename.

  Args:
    data: Un-interpolated 2-D or 3-D modal DG field.
    mapc2p: Optional explicit modal geometry path, or ``''`` for inferred.
    nodes_file: Optional explicit nodal geometry path.
    z_axis: Magnetic-axis vertical position in meters, added to geometry Z.
    phi_tor: Toroidal angle in radians for a 3-D poloidal reconstruction.
    nz_interp: Positive integer z-direction up-sampling factor for 3-D data.
    comp: Zero-based physical field component to map (default first).
    inplace: Replace ``data`` rather than returning a new concrete instance.
    tag: Optional result tag; ``None`` preserves the source tag.
    label: Optional result label; ``None`` preserves the source label.

  Returns:
    The caller's concrete data class, marked ``interpolated=True``. If the
    interpolated input counts are ``(Nx, Nz)``, 2-D grid arrays have shape
    ``(Nx+1, Nz+1)`` and values ``(Nx, Nz, 1)``. For interpolated 3-D counts
    ``(Nx, Ny, Nz)``, grid arrays have shape
    ``(Nx+1, nz_interp*Nz+1)`` and values
    ``(Nx, nz_interp*Nz, 1)``.

  Raises:
    ValueError: For mutually exclusive or missing geometry, input other than
      un-interpolated 2-D/3-D modal data, a missing 3-D toroidal angle,
      invalid ``comp`` or ``nz_interp``, malformed geometry, or a projection
      incompatible with its data grid (when using :func:`map_to_rz`).
  """
  geometry = resolve_geometry(data.file_name,
                              mapc2p=mapc2p,
                              nodes_file=nodes_file)
  projection = resolve_rz_projection(data,
                                     geometry,
                                     z_axis=z_axis,
                                     nz_interp=nz_interp)
  return map_to_rz(data,
                   projection=projection,
                   phi_tor=phi_tor,
                   comp=comp,
                   inplace=inplace,
                   tag=tag,
                   label=label)


__all__ = ["rz", "rz_projections", "projection_for"]
