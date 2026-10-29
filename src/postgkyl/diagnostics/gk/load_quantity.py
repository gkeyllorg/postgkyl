"""Loader for pre-named gyrokinetic quantities.

Resolves a quantity name through the :mod:`postgkyl.diagnostics.gk.
registry`, loads the required source files, computes the quantity, and
returns ready datasets. Ported from
``src_bak/postgkyl/loaders/gk_quantity.py``.
"""

from __future__ import annotations

from typing import Annotated, Literal, TYPE_CHECKING

from postgkyl.cli_spec import ChoiceProvider, KeyValue
from .registry import gk_quant_registry
from .vector_projection import vector_direction, load_vector_geometry
from postgkyl.operations.vector_projection import BASIS_REPRESENTATION, REPRESENTATIONS

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState


def available_quantities() -> list[str]:
  """Return the sorted list of registered quantity names."""
  return gk_quant_registry.list()


def _quantity_catalog() -> list[tuple[str, str]]:
  """Return sorted names and descriptions, including default representations."""
  rows = []
  for name in available_quantities():
    quantity = gk_quant_registry.get(name)
    description = quantity.description
    if quantity.component_basis:
      description += f" (default: {quantity.component_basis})"
    rows.append((name, description))
  return rows


def load_quantity(
    quantity: Annotated[str,
                        ChoiceProvider(available_quantities, metavar="TEXT")],
    species: str | None,
    name: str,
    frame: str | None = None,
    *,
    path: str = "./",
    tag: str = "default",
    label: str | None = None,
    direction: str | None = None,
    rep: Literal["covar", "contra", "cyl", "cart", "norm"] | None = None,
    mass: float | None = None,
    charge: float | None = None,
    gamma_e: float | None = None,
    gamma_i: float | None = None,
    ti_over_te: float | None = None,
    te_ref: float | None = None,
    bmag_ref: float | None = None,
    den_ref: list[float] | None = None,
    temp_ref: list[float] | None = None,
    nu_frac: float | None = None,
    conv: float | None = None,
    fluct: Literal["none", "y", "yz"] | None = None,
    options: Annotated[dict[str, str] | None, KeyValue()] = None,
) -> list:
  """Load and compute a pre-named gyrokinetic quantity.

  Derived vector names have no representation suffix: use ``B_tot``, for
  example, and select components with ``rep`` and ``direction``. The catalog
  lists each quantity once and states its default representation.

  Modal source files retain their DG representation through the calculation:
  products, inverses, square roots, and derivatives use native DG operations.
  Call ``interpolate()`` on the returned datasets when point samples are needed.

  Args:
    quantity: Quantity to load, see -a, --avail_quant for available quantities.
    species: Species name, or a comma-separated list of them; ``None`` for
      species-independent quantities.
    name: Simulation name prefix (e.g. ``'gk_sheath_2x2v_p1'``).
    frame: Frame number, comma-separated list, or ``'start:stop[:step]'``
      range; ``':'``/``None`` selects all available frames.
    path: Directory containing the simulation files.
    tag: Tag for the output dataset(s); suffixed with the species when more
      than one species is requested.
    label: Label override; defaults to the quantity's registered label.
    direction: Numeric component index 0,1,2 (Python also accepts integers).
      For covar/contra these are computational components; for cyl the order
      is R,phi,Z; for cart it is X,Y,Z; norm selects increasing-coordinate
      unit normals. In 2x the full indices remain 0=radial, 1=binormal,
      2=parallel.
    rep: Vector representation: covar, contra, cyl, cart, or norm. Omission
      uses the quantity's native representation, shown by --avail_quant.
      Only for vector quantities.
    mass: Species mass used by quantities that require it.
    charge: Species charge used by quantities that require it.
    gamma_e: Electron adiabatic index for sound-speed quantities.
    gamma_i: Ion adiabatic index for sound-speed quantities.
    ti_over_te: Ion-to-electron temperature ratio of adiabatic electrons,
      used by multi-species quantities when no electron species is listed
      (default 1).
    te_ref: Constant electron temperature (J) replacing the electron
      temperature profile in gyro-Bohm normalizations.
    bmag_ref: Reference magnetic field (T): enters the collision frequency's
      Coulomb logarithm, and replaces the field profile in gyro-Bohm
      normalizations.
    den_ref: Reference density (m^-3) of the collision frequency's Coulomb
      logarithm; one value, or one per species.
    temp_ref: Reference temperature (J) of the collision frequency's Coulomb
      logarithm; one value, or one per species.
    nu_frac: Collision frequency multiplier used by the simulation
      (default 1).
    conv: Coefficient ``c`` of the convective energy flux ``c*T*Gamma``
      removed from the energy flux to form the heat flux (default 3/2).
    fluct: For cross-field fluxes, keep only the turbulent part: the
      correlation of the fluctuations about the Jacobian-weighted ``y`` or
      ``(y, z)`` average in 3x. In 2x, both ``y`` and ``yz`` average along
      stored dimension 1 (z). ``none`` (default) keeps the total flux.
    options: Additional provider options as repeated key/value entries.

  Returns:
    A list of computed ``GDataState`` datasets.

  Raises:
    ValueError: if ``quantity`` is not registered, or it is an
      ``is_multi_species`` quantity requested without a species list.
  """
  extra = dict(options or {})
  for key, value in (("dir", direction), ("rep", rep), ("mass", mass),
                     ("charge", charge), ("gamma_e", gamma_e),
                     ("gamma_i", gamma_i), ("ti_over_te", ti_over_te),
                     ("te_ref", te_ref), ("bmag_ref", bmag_ref),
                     ("den_ref", den_ref), ("temp_ref", temp_ref),
                     ("nu_frac", nu_frac), ("conv", conv), ("fluct", fluct)):
    if value is not None:
      # One value of a per-species option applies to every species.
      extra[key] = value[0] if isinstance(value,
                                          list) and len(value) == 1 else value

  if not gk_quant_registry.has(quantity):
    valid = gk_quant_registry.list()
    raise ValueError(f"Unknown quantity '{quantity}'. Available quantities: "
                     f"{', '.join(valid)}.")

  gkquant = gk_quant_registry.get(quantity)
  requested_rep = extra.get("rep")
  if requested_rep is not None:
    if requested_rep not in REPRESENTATIONS:
      raise ValueError("Choose rep from " + ", ".join(REPRESENTATIONS))
    if not gkquant.is_vector:
      raise ValueError(
          f"Quantity {quantity!r} is not a vector; rep is unsupported.")
  if gkquant.is_vector and not (gkquant.packed_vector and extra.get("dir")
                                is None and requested_rep is None):
    direction = vector_direction(extra.get("dir"))
    extra["dir"] = direction
    extra["rep"] = requested_rep or BASIS_REPRESENTATION[
        gkquant.component_basis]
    extra["vector_geometry"] = load_vector_geometry(path, name, direction,
                                                    gkquant.component_basis,
                                                    extra["rep"])

  path = path.rstrip("/") + "/"
  species_list = [s.strip() for s in species.split(",")] if species else [None]

  frame_inp = str(frame) if frame is not None else None

  if gkquant.is_multi_species:
    # Combine every species into a single dataset (e.g. the sound speed),
    # so it is fetched once for the whole species list instead of once
    # per species.
    if species_list == [None]:
      raise ValueError(
          f"Quantity '{quantity}' combines several species, so it needs a "
          "species list, e.g. --species elc,ion.")

    src_combo_idx, frames = gkquant.get_avail_source_multi(
        path, name, species_list, frame_inp)

    datasets: list["GDataState"] = []
    for fr in frames:
      out = gkquant.fetch_multi(path, name, species_list, fr, src_combo_idx,
                                **extra)

      out_label = (label if label is not None else gkquant.get_label(
          species=species_list[0], direction=extra.get("dir")))
      if len(frames) > 1:
        out_label += f" f{fr}"
      out.set_label(out_label)
      out.set_tag(tag)

      datasets.append(out)
    return datasets

  datasets: list["GDataState"] = []
  for species_idx, sp in enumerate(species_list):
    src_combo_idx, frames = gkquant.get_avail_source(path, name, sp, frame_inp)

    # Tells the fetch functions which entry of a per-species option (e.g.
    # one den_ref per species) applies to the species being computed.
    species_extra = dict(extra, species_idx=species_idx)

    for fr in frames:
      out = gkquant.fetch(path, name, sp, fr, src_combo_idx, **species_extra)

      default_label = gkquant.get_label(
          species=sp,
          direction=extra.get("dir"),
          units=out.ctx.get("projection_units"),
          rep=out.ctx.get("vector_representation"))
      if label is not None:
        out_label = label + (f" {sp}" if len(species_list) > 1 else "")
      else:
        out_label = default_label
      if len(frames) > 1:
        out_label += f" f{fr}"
      out.set_label(out_label)

      out_tag = tag + (f"_{sp}" if len(species_list) > 1 else "")
      out.set_tag(out_tag)

      datasets.append(out)

  return datasets
