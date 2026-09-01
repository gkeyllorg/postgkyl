"""Loader for pre-named gyrokinetic quantities.

Resolves a quantity name through the :mod:`postgkyl.diagnostics.gyrokinetics.
registry`, loads the required source files, computes the quantity, and
returns ready datasets. Ported from
``src_bak/postgkyl/loaders/gk_quantity.py``.
"""

from __future__ import annotations

from typing import Annotated, TYPE_CHECKING

from postgkyl.command_spec import ChoiceProvider, KeyValue
from .registry import gk_quant_registry

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState
# end


def available_quantities() -> list[str]:
  """Return the sorted list of registered quantity names."""
  return gk_quant_registry.list()
# end


def load_gk_quantity(
    quantity: Annotated[str, ChoiceProvider(available_quantities)],
    species: str | None, name: str, frame: str | None = None, *,
    path: str = "./", tag: str = "default", label: str | None = None,
    direction: int | None = None, mass: float | None = None,
    charge: float | None = None, gamma_e: float | None = None,
    gamma_i: float | None = None, kind: str | None = None,
    read_options: Annotated[dict[str, str] | None, KeyValue()] = None,
    ) -> list:
  """Load and compute a pre-named gyrokinetic quantity.

  Args:
    quantity: Registered quantity name (see :func:`available_quantities`).
    species: Species name, or a comma-separated list of them; ``None`` for
      species-independent quantities.
    name: Simulation name prefix (e.g. ``'gk_sheath_2x2v_p1'``).
    frame: Frame number, comma-separated list, or ``'start:stop[:step]'``
      range; ``':'``/``None`` selects all available frames.
    path: Directory containing the simulation files.
    tag: Tag for the output dataset(s); suffixed with the species when more
      than one species is requested.
    label: Label override; defaults to the quantity's registered label.
    direction: Vector direction for quantities that expose components.
    mass: Species mass used by quantities that require it.
    charge: Species charge used by quantities that require it.
    gamma_e: Electron adiabatic index for sound-speed quantities.
    gamma_i: Ion adiabatic index for sound-speed quantities.
    kind: Named variant accepted by a quantity provider.
    read_options: Additional provider options as repeated key/value entries.

  Returns:
    A list of computed ``GDataState`` datasets.

  Raises:
    ValueError: if ``quantity`` is not registered, or it is an
      ``is_multi_species`` quantity requested without a species list.
  """
  extra = dict(read_options or {})
  for key, value in (("dir", direction), ("mass", mass), ("charge", charge),
      ("gamma_e", gamma_e), ("gamma_i", gamma_i), ("kind", kind)):
    if value is not None:
      extra[key] = value
    # end
  # end

  if not gk_quant_registry.has(quantity):
    valid = gk_quant_registry.list()
    raise ValueError(
        f"Unknown quantity '{quantity}'. Available quantities: "
        f"{', '.join(valid)}.")
  # end

  gkquant = gk_quant_registry.get(quantity)
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
    # end

    src_combo_idx, frames = gkquant.get_avail_source_multi(
        path, name, species_list, frame_inp)

    datasets: list["GDataState"] = []
    for fr in frames:
      out = gkquant.fetch_multi(path, name, species_list, fr, src_combo_idx,
          **extra)

      out_label = label if label is not None else gkquant.get_label()
      if len(frames) > 1:
        out_label += f" f{fr}"
      # end
      out.set_label(out_label)
      out.set_tag(tag)

      datasets.append(out)
    # end
    return datasets
  # end

  datasets: list["GDataState"] = []
  for species_idx, sp in enumerate(species_list):
    src_combo_idx, frames = gkquant.get_avail_source(path, name, sp, frame_inp)

    # Tells the fetch functions which entry of a per-species '--extra' array
    # (e.g. 'mass=1,2,3') applies to the species being computed.
    species_extra = dict(extra, species_idx=species_idx)

    for fr in frames:
      out = gkquant.fetch(path, name, sp, fr, src_combo_idx, **species_extra)

      default_label = gkquant.get_label(species=sp, direction=extra.get("dir"))
      if label is not None:
        out_label = label + (f" {sp}" if len(species_list) > 1 else "")
      # end
      else:
        out_label = default_label
      # end
      if len(frames) > 1:
        out_label += f" f{fr}"
      # end
      out.set_label(out_label)

      out_tag = tag + (f"_{sp}" if len(species_list) > 1 else "")
      out.set_tag(out_tag)

      datasets.append(out)
    # end
  # end

  return datasets
# end
