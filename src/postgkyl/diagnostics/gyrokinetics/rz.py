"""Compatibility aliases for the gyrokinetic R-Z operation.

Canonical imports live in :mod:`postgkyl.operations.gyrokinetics`. This
module remains for the current major version and contains no copied
algorithm or defaults.
"""

from postgkyl.operations.gyrokinetics.rz import (
    Geometry,
    RzProjection,
    geometry_prefix,
    gk_rz,
    map_to_rz,
    per_block_path,
    resolve_geometry,
    resolve_rz_projection,
)


def rz_projections(datasets, *, mapc2p: str | None = None,
    nodes_file: str | None = None, z_axis: float = 0.0,
    nz_interp: int = 8) -> dict:
  """Compatibility batch wrapper using this module's patchable aliases."""
  projections = {}
  for data in datasets:
    key = geometry_prefix(data.file_name)
    if key in projections:
      continue
    # end
    block = data.ctx.get("block")
    geometry = resolve_geometry(data.file_name,
        mapc2p=per_block_path(mapc2p, block),
        nodes_file=per_block_path(nodes_file, block))
    projections[key] = resolve_rz_projection(data, geometry, z_axis=z_axis,
        nz_interp=nz_interp)
  # end
  return projections
# end


def projection_for(projections: dict, data):
  """Return the compatibility projection belonging to ``data``'s block."""
  return projections[geometry_prefix(data.file_name)]
# end

__all__ = [
    "Geometry", "RzProjection", "geometry_prefix", "gk_rz", "map_to_rz",
    "per_block_path", "projection_for", "resolve_geometry",
    "resolve_rz_projection", "rz_projections",
]
