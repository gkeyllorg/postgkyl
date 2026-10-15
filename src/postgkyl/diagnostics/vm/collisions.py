"""GData collision diagnostics using Gkeyll's Vlasov implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl.dg import physics, rep
from postgkyl.gdatastate.layout import dg_layout as _dg_layout, require_native_modal as _require_native_modal
from postgkyl.operations._compatibility import require_collocated_layout as _require_collocated_layout

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState


def coulomb_log(density_self: GDataState,
                density_other: GDataState,
                temperature_self: GDataState,
                temperature_other: GDataState,
                bmag: GDataState,
                *,
                mass_self: float,
                mass_other: float,
                charge_self: float,
                charge_other: float,
                epsilon_0: float,
                hbar: float,
                num_quad: int | None = None,
                inplace: bool = False,
                tag: str | None = None,
                label: str | None = None) -> GDataState:
  """Compute Gkeyll's species-ordered Coulomb logarithm on collocated fields.

  Inputs are scalar densities, temperatures in energy units (T/m = v_t^2),
  and magnetic-field magnitude. Supply physical constants in consistent
  units. This native expression is ordered: interchanging species can change
  it; Gkeyll's Spitzer updater averages both orders internally.

  Modal inputs are evaluated at quadrature points and projected back into
  the same modal basis. Nodal, quadrature, and ordinary point inputs retain
  their representation and backend. No input is changed unless inplace=True.
  num_quad optionally selects Gauss points per direction for modal inputs.

  Args:
    density_self: Scalar test-species density.
    density_other: Scalar field-species density.
    temperature_self: Scalar test-species temperature in energy units.
    temperature_other: Scalar field-species temperature in energy units.
    bmag: Scalar magnetic-field magnitude.
    mass_self: Test-species mass.
    mass_other: Field-species mass.
    charge_self: Test-species charge, including sign.
    charge_other: Field-species charge, including sign.
    epsilon_0: Vacuum permittivity in the chosen unit system.
    hbar: Reduced Planck constant; zero selects the classical cutoff.
    num_quad: Gauss points per axis for modal projection.
    inplace: Replace the first dataset's values.
    tag: Result tag.
    label: Result label.
  """
  fields = (density_self, density_other, temperature_self, temperature_other,
            bmag)
  layout = _dg_layout(density_self)
  for field in fields:
    _require_collocated_layout(density_self, field)
    current = _dg_layout(field)
    if (current.num_fields if current else field.values.shape[-1]) != 1:
      raise ValueError("Coulomb logarithm requires scalar fields")
  modal = layout is not None and layout.value_form == "modal"
  if modal:
    layout = _require_native_modal(density_self)
    values = [
        rep.modal_to_quad(*layout.basis_args, field.native, num_quad,
                          **layout.basis_kwargs).view() for field in fields
    ]
  else:
    if num_quad is not None:
      raise ValueError("num_quad applies only to modal inputs")
    values = [field.values for field in fields]
  out = physics.coulomb_log(*values,
                            mass_self=mass_self,
                            mass_other=mass_other,
                            charge_self=charge_self,
                            charge_other=charge_other,
                            epsilon_0=epsilon_0,
                            hbar=hbar)
  if modal:
    out = rep.quad_to_modal(*layout.basis_args, rep.wrap(out), num_quad,
                            **layout.basis_kwargs)
  elif density_self.backend == "gkyl":
    out = rep.wrap(out)
  return density_self._result(density_self.grid,
                              out,
                              inplace=inplace,
                              tag=tag,
                              label=label)


def spitzer_coll_freq(moms_self: GDataState,
                      moms_other: GDataState,
                      bmag: GDataState,
                      *,
                      mass_self: float,
                      mass_other: float,
                      charge_self: float,
                      charge_other: float,
                      epsilon_0: float,
                      hbar: float,
                      nufrac: float = 1.0,
                      vt_sq_min_self: float = 0.0,
                      vt_sq_min_other: float = 0.0,
                      num_quad: int | None = None,
                      inplace: bool = False,
                      tag: str | None = None,
                      label: str | None = None) -> GDataState:
  """Compute modal Spitzer collision frequency with Gkeyll's native updater.

  moms_self and moms_other pack Maxwellian moments [n, u..., T/m], not raw
  velocity moments. bmag is a scalar on the same configuration grid. The
  Coulomb log uses cell averages; the density/thermal-speed factor is
  evaluated and projected using num_quad Gauss points per direction
  (default poly_order+2). Velocity floors are squared thermal speeds.
  epsilon_0, hbar, masses, charges, and fields must use consistent units.
  Returns one scalar modal field on the configuration grid.

  Args:
    moms_self: Test-species Maxwellian moments, packed as n, u..., T/m.
    moms_other: Field-species Maxwellian moments in the same layout.
    bmag: Scalar modal magnetic-field magnitude.
    mass_self: Test-species mass.
    mass_other: Field-species mass.
    charge_self: Test-species charge, including sign.
    charge_other: Field-species charge, including sign.
    epsilon_0: Vacuum permittivity in the chosen unit system.
    hbar: Reduced Planck constant; zero selects the classical cutoff.
    nufrac: Nonnegative multiplier of the collision frequency.
    vt_sq_min_self: Test-species squared thermal-speed floor.
    vt_sq_min_other: Field-species squared thermal-speed floor.
    num_quad: Gauss points per direction; defaults to poly_order+2.
    inplace: Replace the first dataset's values.
    tag: Result tag.
    label: Result label.
  """
  layout = _require_native_modal(moms_self)
  for field in (moms_other, bmag):
    _require_native_modal(field)
    _require_collocated_layout(moms_self, field)
  out = physics.spitzer_coll_freq(*layout.basis_args,
                                  moms_self.native,
                                  moms_other.native,
                                  bmag.native,
                                  mass_self=mass_self,
                                  mass_other=mass_other,
                                  charge_self=charge_self,
                                  charge_other=charge_other,
                                  epsilon_0=epsilon_0,
                                  hbar=hbar,
                                  nufrac=nufrac,
                                  vt_sq_min_self=vt_sq_min_self,
                                  vt_sq_min_other=vt_sq_min_other,
                                  num_quad=num_quad)
  return moms_self._result(moms_self.grid,
                           out,
                           inplace=inplace,
                           tag=tag,
                           label=label)
