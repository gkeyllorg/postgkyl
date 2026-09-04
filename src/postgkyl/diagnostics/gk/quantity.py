"""``GkQuantity`` -- a registered gyrokinetic quantity, and its registry.

Ported from ``src_bak/postgkyl/gk/gk_quantities/gkquantity.py``. A quantity
names one or more *source combinations* (files and/or other, already-
registered ``GkQuantity`` objects) together with the ``fetch_func`` that
turns a resolved combination into the quantity's data. Source-combination
frame discovery calls :mod:`postgkyl.diagnostics.discovery` -- the one home
for "what outputs does this directory hold" -- instead of globbing on its
own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from postgkyl.gdata import GData

from .. import discovery

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState
# end


@dataclass(frozen=True)
class GkQuantity:
  """A gyrokinetic quantity: one or more source combinations + fetch logic.

  Attributes:
    name: Name of the quantity (the registry key).
    source: List of source combinations to try, in preference order; each
      combination is a list of either file-naming-convention source strings
      (e.g. ``"M0"``) or nested ``GkQuantity`` (computed on demand).
    fetch_func: The fetch function for each entry in ``source`` (same
      index), taking the resolved list of source ``GDataState`` and
      returning the quantity's ``GDataState``.
    label: LaTeX-format label for plotting (``%s`` for species name or
      direction).
    is_time_dep: Whether the quantity is time-dependent (written in frames).
    is_species_dep: Whether the quantity is species-dependent.
    is_vector: Whether the quantity is a vector (multiple components,
      selected via the ``dir`` extra).
    is_tensor: Whether the quantity is a tensor.
    is_integrated: Whether the quantity is a grid integral.
    is_geo: Whether the quantity is a (frame-independent) geometry
      quantity, named ``<name>-<src>.gkyl`` with no frame number.
    is_multi_species: Whether the quantity combines several species into a
      single dataset (e.g. the sound speed, which mixes the electrons and
      every ion). Such a quantity is fetched once for the whole species
      list rather than once per species (:meth:`get_avail_source_multi`/
      :meth:`fetch_multi`), and its fetch function receives one list of
      sources per species instead of a flat list.
  """

  name: str
  source: list
  fetch_func: list[Callable]
  label: str
  is_time_dep: bool = False
  is_species_dep: bool = False
  is_vector: bool = False
  is_tensor: bool = False
  is_integrated: bool = False
  is_geo: bool = False
  is_multi_species: bool = False

  # ------------------------------------------------------------ internal
  def _src_stem(self, path: str, name: str, species: str, src: str) -> str:
    """Stem of a string source's file name, up to (not including) the frame
    number (geo files have no frame, so no trailing separator)."""
    if self.is_geo:
      return os.path.join(path, f"{name}-{src}")
    # end
    if self.is_species_dep:
      src_ = f"{src}_" if src else ""
      return os.path.join(path, f"{name}-{species}_{src_}")
    # end
    return os.path.join(path, f"{name}-{src}_")
  # end

  def _src_file_name(self, path: str, name: str, species: str, src: str,
      frame: int | None) -> str:
    """Full file name for a string source at the given frame."""
    stem = self._src_stem(path, name, species, src)
    if self.is_geo:
      return f"{stem}.gkyl"
    # end
    return f"{stem}{frame}.gkyl"
  # end

  def _avail_frames_src(self, path: str, name: str, species: str, src: str,
      frames: list[int] | None = None) -> set[int]:
    """Available frames for a string source's ``<stem><frame>.gkyl`` family."""
    stem = self._src_stem(path, name, species, src)
    return discovery.available_frames(stem, frames=frames)
  # end

  def _avail_combo_frames(self, path: str, name: str, species: str,
      frames: list[int] | None = None) -> tuple[int, set[int]]:
    """Find the first source combination whose files all exist and share the
    same set of available frames.

    Returns:
      ``(combo_idx, frames_avail)``; a combination made up only of geo files
      is flagged with ``frames_avail == {-1}``.
    """
    frames_avail: set[int] = set()
    combo_idx = 0
    for cidx, combo in enumerate(self.source):
      for src in combo:
        if isinstance(src, str) and self.is_geo:
          if not os.path.isfile(os.path.join(path, f"{name}-{src}.gkyl")):
            frames_avail = set()
            break
          # end
          continue
        # end

        if isinstance(src, str):
          frames_avail_q = self._avail_frames_src(path, name, species, src, frames)
        # end
        else:
          _, frames_avail_q = src._avail_combo_frames(path, name, species, frames)
        # end

        if frames_avail_q == {-1}:
          combo_idx = cidx
          continue
        # end

        if frames_avail_q:
          if not frames_avail:
            frames_avail = set(frames_avail_q)
          # end
          elif frames_avail_q != frames_avail:
            frames_avail = set()
            break
          # end
          combo_idx = cidx
        # end
        else:
          break
      # end
        # end
      else:
        if not frames_avail:
          frames_avail = {-1}
          combo_idx = cidx
        # end
      # end

      if frames_avail:
        break
      # end
    # end
    return combo_idx, frames_avail
  # end

  # -------------------------------------------------------------- public
  def get_label(self, species: str | None = None,
      direction: str | None = None) -> str:
    """Get the display label, substituting ``%s`` with species or direction."""
    if self.is_vector:
      return self.label % str(direction) if direction is not None else self.label % "i"
    # end
    if self.is_species_dep:
      return self.label % str(species[0]) if species is not None else self.label % "s"
    # end
    return self.label
  # end

  def get_avail_source(self, path: str, name: str, species: str,
      frame_inp: str | None) -> tuple[int, list]:
    """Identify the source combination and frame list needed for this
    quantity.

    Args:
      path: Directory containing the simulation files.
      name: Simulation name prefix.
      species: Species name.
      frame_inp: A single frame, a comma-separated list, or a
        ``'start:stop[:step]'`` range (``None``/``':'`` means every
        available frame).

    Returns:
      ``(combo_idx, frames)``.

    Raises:
      FileNotFoundError: if no source combination's files are found.
    """
    frame_list: list[int] = []
    if frame_inp is not None:
      frame_inp = frame_inp.strip()
      if "," in frame_inp:
        frame_list = [int(f.strip()) for f in frame_inp.split(",")]
      # end
      elif ":" not in frame_inp:
        frame_list = [int(frame_inp)]
      # end
    # end

    combo_idx, frames_avail = self._avail_combo_frames(path, name, species, frame_list)

    if not frames_avail:
      raise FileNotFoundError(
          f"No files found for the requested quantity (path={path!r}, "
          f"name={name!r}).")
    # end

    if frames_avail == {-1}:
      return combo_idx, [None]
    # end

    if len(frame_list) == 0:
      frames_avail_sorted = sorted(frames_avail)
      parts = frame_inp.split(":") if frame_inp else [""]
      lower = int(parts[0]) if parts[0] else frames_avail_sorted[0]
      upper = (int(parts[1]) if len(parts) > 1 and parts[1]
               else frames_avail_sorted[-1] + 1)
      step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
      frame_list = [f for f in frames_avail_sorted
                    if lower <= f < upper and (f - lower) % step == 0]
    # end

    return combo_idx, frame_list
  # end

  def get_src_gdata(self, src: "str | GkQuantity", path: str, name: str,
      species: str, frame: int | None, **extra) -> "GDataState":
    """The populated dataset for one source: a loaded file, or a nested
    quantity computed from its own sources."""
    if isinstance(src, str):
      return GData(self._src_file_name(path, name, species, src, frame))
    # end
    combo_idx, _ = src.get_avail_source(path, name, species,
        str(frame) if frame is not None else None)
    combo = src.source[combo_idx]
    fetch_func = src.fetch_func[combo_idx]
    gdatas = [src.get_src_gdata(s, path, name, species, frame, **extra)
              for s in combo]
    return fetch_func(gdatas, **extra)
  # end

  def fetch(self, path: str, name: str, species: str, frame: int | None,
      combo_idx: int, **extra) -> "GDataState":
    """Fetch the source files for ``combo_idx`` and compute the quantity."""
    combo = self.source[combo_idx]
    fetch_func = self.fetch_func[combo_idx]
    gdatas = [self.get_src_gdata(src, path, name, species, frame, **extra)
              for src in combo]
    extra = dict(extra, path=path, name=name, species=species, frame=frame)
    return fetch_func(gdatas, **extra)
  # end

  def get_avail_source_multi(self, path: str, name: str,
      species_list: list[str], frame_inp: str | None) -> tuple[int, list]:
    """Multi-species counterpart of :meth:`get_avail_source`: resolve the
    source combination and frames for every species in ``species_list``,
    keeping only the frames available for all of them (the quantity folds
    every species into one dataset, so a frame missing for any one of them
    can't be computed at all).

    Raises:
      FileNotFoundError: if no frame is available for every species.
    """
    combo_idx = 0
    frames_common: set[int] | None = None
    for species in species_list:
      combo_idx, frames = self.get_avail_source(path, name, species, frame_inp)
      frames_common = (set(frames) if frames_common is None
                        else frames_common & set(frames))
    # end
    if not frames_common:
      raise FileNotFoundError(
          f"No frames are available for all of the requested species "
          f"{species_list} (path={path!r}, name={name!r}).")
    # end
    return combo_idx, sorted(frames_common)
  # end

  def fetch_multi(self, path: str, name: str, species_list: list[str],
      frame: int | None, combo_idx: int, **extra) -> "GDataState":
    """Multi-species counterpart of :meth:`fetch`, for
    ``is_multi_species`` quantities.

    The fetch function is handed one list of sources per species, in the
    order of ``species_list``: ``gdatas[i][j]`` is the ``j``-th source of
    the ``i``-th species. Each species' sources are resolved with
    ``extra['species_idx']`` set to that species' position, so a
    per-species ``--extra`` array (e.g. ``mass=1,2,3``) picks the right
    entry inside the sources too, not just at the top level. Species names
    are passed along as ``extra['species']``.
    """
    combo = self.source[combo_idx]
    fetch_func = self.fetch_func[combo_idx]
    gdatas = [[self.get_src_gdata(src, path, name, species, frame,
                                   **dict(extra, species_idx=species_idx))
               for src in combo]
              for species_idx, species in enumerate(species_list)]
    extra = dict(extra, path=path, name=name, species=list(species_list),
        frame=frame)
    return fetch_func(gdatas, **extra)
  # end
# end


class GkQuantityRegistry:
  """Registry of pre-named gyrokinetic quantities."""

  def __init__(self):
    self._registry: dict[str, GkQuantity] = {}
  # end

  def register(self, quantity: GkQuantity) -> None:
    """Register a new gyrokinetic quantity."""
    self._registry[quantity.name] = quantity
  # end

  def get(self, name: str) -> GkQuantity | None:
    """Get a registered quantity by name, or ``None`` if unregistered."""
    return self._registry.get(name)
  # end

  def list(self) -> list[str]:
    """Sorted list of all registered quantity names."""
    return sorted(self._registry)
  # end

  def has(self, name: str) -> bool:
    """Whether ``name`` is registered."""
    return name in self._registry
  # end
# end
