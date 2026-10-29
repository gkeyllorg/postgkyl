"""Native gyrokinetic velocity moments of GData distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl.dg import physics
from postgkyl.gdatastate.layout import require_native_modal as _require_native_modal
from postgkyl.operations._compatibility import require_collocated_layout as _require_collocated_layout, uniform_cartesian_grid as _uniform_cartesian_grid
from postgkyl.numerics import grids_compatible as _grids_compatible

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState


def mom_gyrokinetic(distribution: GDataState,
                    bmag: GDataState,
                    *,
                    mass: float,
                    moment: str = "M0",
                    charge: float = 0.0,
                    phi: GDataState | None = None,
                    inplace: bool = False,
                    tag: str | None = None,
                    label: str | None = None) -> GDataState:
  """Integrate a modal gyrokinetic distribution over velocity using Gkeyll.

  bmag defines the configuration grid; basis order and the velocity-space
  split come from the datasets. Supports p1 gkhybrid and p2 serendipity in
  1x1v, 1x2v, 2x2v, plus p1 3x2v. Uniform physical velocity coordinates
  are required; mapped velocity coordinates are not interpreted implicitly.

  moment is M0, M1, M2, M2par, M2perp, M3par, M3perp, ThreeMoments,
  FourMoments, or HamiltonianMoments. Perpendicular moments require 2v;
  HamiltonianMoments requires the electrostatic potential phi and charge.
  Combined outputs pack complete modal field blocks in the named order.

  Retains the native stored-distribution normalization: for 2v, the measure
  is (2*pi/mass) dv_parallel dmu. A configuration Jacobian present in the
  distribution remains in the output. This function does not divide out
  geometry or multiply by B. Returns modal configuration-space GData.

  Args:
    distribution: Scalar modal distribution on a uniform phase-space grid.
    bmag: Scalar modal magnetic-field magnitude on the configuration grid.
    mass: Positive species mass.
    moment: Native moment name, for example M0 or FourMoments.
    charge: Species charge, used by HamiltonianMoments.
    phi: Scalar electrostatic potential for HamiltonianMoments.
    inplace: Replace the distribution with its configuration-space moment.
    tag: Result tag.
    label: Result label.
  """
  phase = _require_native_modal(distribution)
  conf = _require_native_modal(bmag)
  cdim, vdim = conf.ndim, phase.ndim - conf.ndim
  expected = "gkhybrid" if conf.poly_order == 1 else "serendipity"
  if conf.num_fields != 1 or phase.num_fields != 1:
    raise ValueError(
        "gyrokinetic moments need one distribution and scalar bmag")
  if conf.basis_type != "serendipity" or phase.basis_type != expected or phase.poly_order != conf.poly_order:
    raise ValueError(
        "gyrokinetic moments require matching configuration and phase bases")
  if phase.basis_type == "gkhybrid" and (phase.cdim, phase.vdim) != (cdim,
                                                                     vdim):
    raise ValueError("phase-space basis split disagrees with bmag")
  if not _grids_compatible(distribution.grid[:cdim], bmag.grid):
    raise ValueError("bmag must match the distribution's configuration grid")
  if distribution.ctx.get("mapped_axes") or distribution.ctx.get("mapc2p_vel"):
    raise ValueError(
        "mapped coordinates require an explicit velocity-map binding")
  if phi is not None:
    _require_native_modal(phi)
    _require_collocated_layout(bmag, phi)
    if phi.native.ncomp != bmag.native.ncomp:
      raise ValueError("phi must be a scalar configuration-space field")
  grid = _uniform_cartesian_grid(distribution)
  out = physics.mom_gyrokinetic(cdim,
                                vdim,
                                conf.poly_order,
                                distribution.native,
                                bmag.native,
                                grid["lower"],
                                grid["upper"],
                                grid["cells"],
                                mass=mass,
                                moment=moment,
                                charge=charge,
                                phi=None if phi is None else phi.native)
  return distribution._result(bmag.grid,
                              out,
                              inplace=inplace,
                              tag=tag,
                              label=label,
                              cells=bmag.num_cells,
                              basis_type="serendipity",
                              poly_order=conf.poly_order,
                              value_form="modal",
                              num_cdim=cdim,
                              num_vdim=0,
                              cdim=cdim,
                              vdim=0,
                              num_quad=None,
                              quad_rule=None)
