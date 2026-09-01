"""Compatibility aliases for the gyrokinetic R-Z operation.

Canonical imports live in :mod:`postgkyl.operations.gyrokinetics`.  This
module remains for the current major version, is scheduled for removal in the
next major version, and intentionally contains no algorithm or copied defaults.
"""

from postgkyl.operations.gyrokinetics.rz import (
    Geometry,
    RzProjection,
    gk_rz,
    map_to_rz,
    resolve_geometry,
    resolve_rz_projection,
)

__all__ = ["Geometry", "RzProjection", "gk_rz", "map_to_rz",
    "resolve_geometry", "resolve_rz_projection"]
