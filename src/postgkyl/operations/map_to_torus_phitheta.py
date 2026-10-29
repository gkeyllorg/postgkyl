"""Map modal fields onto a flux surface's (phi, theta) plane, with reusable projections."""
from __future__ import annotations

from typing import Annotated
from postgkyl.cli_spec import CliHidden

from dataclasses import dataclass
from functools import partial
import numpy as np

from postgkyl.gdatastate.gdatastate import GDataState
from .geometry import Geometry, resolve_geometry
from ._mapping import (
    DEFAULT_NZ_INTERP,
    _radial_index,
    _validate_component,
    _validate_modal_data,
    _validate_positive_int,
    mappings_by_geometry,
    mapping_for,
)
from .map_to_torus_rz import (RzProjection, resolve_rz_projection,
                              _toroidal_samples, _validate_projection)

_DEFAULT_NPHI = 128


@dataclass(frozen=True)
class PhiThetaProjection:
  """Precomputed (phi, theta) sampling of one flux surface on one grid.

  The surface is radial interpolation sample ``x_idx``. Fields are sampled
  through the ``rz`` projection at toroidal angles ``phi_tor`` and at its
  parallel positions ``rz.zf``.
  """

  rz: RzProjection
  x_idx: int
  phi_tor: np.ndarray

  @property
  def computational_grid(self) -> tuple[np.ndarray, ...]:
    return self.rz.computational_grid


def resolve_phitheta_projection(
    first: "GDataState",
    geo: Geometry,
    *,
    x_idx: int = 0,
    nphi: int = _DEFAULT_NPHI,
    nz_interp: int = DEFAULT_NZ_INTERP) -> PhiThetaProjection:
  """Build a (phi, theta) projection for ``first``'s grid and ``geo``.

  Only the computational grid and DG metadata are read from ``first``.
  Toroidal samples cover one turn, ``2*pi * k / nphi``.
  """
  _validate_modal_data(first, "map_to_torus_phitheta", (3, ))
  nphi = _validate_positive_int(nphi, "nphi")
  rz = resolve_rz_projection(first, geo, nz_interp=nz_interp)
  return PhiThetaProjection(rz=rz,
                            x_idx=_radial_index(
                                x_idx, rz.computational_grid[0].size - 1),
                            phi_tor=np.linspace(0.0,
                                                2.0 * np.pi,
                                                nphi,
                                                endpoint=False))


def map_to_torus_phitheta(
    data: "GDataState",
    *,
    projection: Annotated[
        PhiThetaProjection | None,
        CliHidden("reuse a projection through the Python API")] = None,
    mapc2p: str | None = None,
    nodes_file: str | None = None,
    x_idx: int = 0,
    nphi: int = _DEFAULT_NPHI,
    nz_interp: int = DEFAULT_NZ_INTERP,
    comp: int = 0,
    inplace: bool = False,
    tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Map a modal field component onto one flux surface's (phi, theta) plane.

  Theta is the parallel coordinate z, which makes one poloidal turn. Each
  sample evaluates the DG field as map_to_torus_rz does, so the column at angle
  phi is row x_idx of the R-Z slice at phi. Unless the binormal domain is a
  whole fraction of the torus, its copies meet along field lines with a
  jump.

  Args:
    data: Un-interpolated 3-D field-aligned modal DG field.
    projection: Reusable Python projection built with
      resolve_phitheta_projection; its computational grid must match data.
      Omit to load geometry.
    mapc2p: Explicit modal geometry path, or '' for the inferred modal file.
    nodes_file: Explicit interior-node geometry path; excludes mapc2p.
      When both paths are omitted, infer geometry from data.file_name,
      preferring geo_int_nodes over geo_int_mapc2p. '*' selects the block.
    x_idx: Flux-surface index on the interpolated radial grid; negative
      indices count back from the outer edge.
    nphi: Number of toroidal samples over one turn.
    nz_interp: Positive parallel-direction up-sampling factor.
    comp: Zero-based physical field component to map.
    inplace: Mutate and return data instead of creating a dataset.
    tag: Optional result tag.
    label: Optional result label.

  Returns:
    The caller's concrete data class with NumPy point values on a
    ``(phi, theta)`` grid of sample points. Projection sampling options
    cannot accompany a reusable projection; set them when building it.
  """
  _validate_modal_data(data, "map_to_torus_phitheta", (3, ))
  _validate_component(data, comp)
  if projection is None:
    geometry = resolve_geometry(data.file_name,
                                mapc2p=mapc2p,
                                nodes_file=nodes_file)
    projection = resolve_phitheta_projection(data,
                                             geometry,
                                             x_idx=x_idx,
                                             nphi=nphi,
                                             nz_interp=nz_interp)
  elif (mapc2p is not None or nodes_file is not None or x_idx != 0
        or nphi != _DEFAULT_NPHI or nz_interp != DEFAULT_NZ_INTERP):
    raise ValueError(
        "projection cannot be combined with geometry or sampling options; "
        "set them in resolve_phitheta_projection instead.")
  rz = projection.rz
  _validate_projection(data, rz)
  x_idx = _radial_index(projection.x_idx, rz.phi_lo.shape[0])
  surface = _toroidal_samples(data, comp, rz, x_idx,
                              projection.phi_tor[:, np.newaxis])
  return data._result([projection.phi_tor, rz.zf],
                      surface[..., np.newaxis],
                      inplace=inplace,
                      tag=tag,
                      label=label,
                      interpolated=True)


def phitheta_projections(
    datasets,
    *,
    mapc2p: str | None = None,
    nodes_file: str | None = None,
    x_idx: int = 0,
    nphi: int = _DEFAULT_NPHI,
    nz_interp: int = DEFAULT_NZ_INTERP) -> dict[str | None, PhiThetaProjection]:
  """Build one reusable (phi, theta) projection per block across fields or frames."""
  return mappings_by_geometry(datasets,
                              partial(resolve_phitheta_projection,
                                      x_idx=x_idx,
                                      nphi=nphi,
                                      nz_interp=nz_interp),
                              mapc2p=mapc2p,
                              nodes_file=nodes_file)


projection_for = mapping_for
