"""Gyrokinetic particle- and energy-balance diagnostic.

Ported from ``src_bak/postgkyl/apps/gk_particle_balance.py`` and
``gk_energy_balance.py``. Reads the integrated time-trace files a
gyrokinetic simulation writes (integrated moments of ``df/dt``, of the
source(s) and of the boundary fluxes; for the energy, also the field/apar
energy rate of change), sums them over species and (for multiblock runs)
blocks, and plots the residual of the particle or energy balance::

    N_err = S + S_BGK - bflux - df/dt
    E_err = S + S_BGK - bflux - (df/dt - dfield/dt [- dapar/dt])

with ``S`` the source and ``S_BGK`` the BGK source. Without those files, it
falls back to the balance files Gkeyll writes when a species or the app
requests the balance (``<name>-[<species>_]balance_<quantity>.gkyl``). The
app's energy balance also contains the background-charge term of adiabatic
species::

    E_err = S + S_BGK - bflux - (df/dt - dfield/dt + dbg/dt)

Typer options become explicit keyword-only parameters; the old CLI's dataset
stack (``ctx.obj.data``) and ``verb_print`` echo are dropped -- the computed
traces come back as a :class:`BalanceTraces` alongside the Figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from . import utils

_DIRS = ("x", "y", "z")
_EDGES = ("lower", "upper")
_LINE_STYLES = ("-", "--", ":", "-.")
_XY_LABEL_FONT_SIZE = 17
_TITLE_FONT_SIZE = 17
_TICK_FONT_SIZE = 14
_LEGEND_FONT_SIZE = 14


@dataclass(frozen=True)
class _Quantity:
  """What distinguishes the balance of one conserved quantity.

  Attributes:
    moments: Integrated moments holding the quantity, in order of preference.
    comp: Component of the quantity in those moments.
    has_field: Whether the balance includes the field-energy terms. Gkeyll's
      per-species balance files lack them, so only its app balance file holds
      such a balance.
    symbol: Mathtext symbol of the quantity, without ``$``.
    noun: The quantity in the plot titles.
  """

  moments: tuple[str, ...]
  comp: int
  has_field: bool
  symbol: str
  noun: str


_QUANTITIES = {
    # The density is the first component of these moments.
    "particle":
    _Quantity(("M0", "M0M1M2", "M0M1M2parM2perp", "HamiltonianMoments"), 0,
              False, r"\mathcal{N}", "particle"),
    # Hamiltonian moments store (M0, M1, H).
    "energy":
    _Quantity(("HamiltonianMoments", ), 2, True, r"\mathcal{E}", "energy"),
}


@dataclass(frozen=True)
class BalanceTraces:
  """Computed balance time traces (all 1-D, aligned to ``time``).

  Attributes:
    time: Time stamps of the ``fdot`` trace after t=0, where the rates are
      defined.
    fdot: Rate of change of the balanced moment of the distribution function,
      summed over species and blocks.
    src: Rate of change from the sources, or ``None`` if no source file was
      found for any species/block (balance file: if it is zero throughout).
    bflux_tot: Rate of change from boundary particle fluxes, or ``None`` if
      none were found (balance file: if it is zero throughout).
    mom_err: The balance residual (``None`` when ``relative_error``).
    mom_err_norm: The *relative* balance residual (only set when
      ``relative_error=True``; ``None`` otherwise).
    src_bgk: Rate of change from the BGK sources, or ``None`` if none was
      found (balance file: if it is zero throughout, or the file predates
      the ``src_bgk_mom`` column and includes it in ``src``).
    field_dot: Rate of change of the field energy (energy balance only).
    apar_dot: Rate of change of the vector-potential energy (electromagnetic
      energy balance only), or ``None``.
    bg_dot: Rate of change of the background-charge energy ``int rho_bg phi``
      of adiabatic species, or ``None``. Only Gkeyll's energy balance file
      has it, and it is ``None`` there if it is zero throughout.
  """

  time: np.ndarray
  fdot: np.ndarray
  src: np.ndarray | None
  bflux_tot: np.ndarray | None
  mom_err: np.ndarray | None
  mom_err_norm: np.ndarray | None = None
  src_bgk: np.ndarray | None = None
  field_dot: np.ndarray | None = None
  apar_dot: np.ndarray | None = None
  bg_dot: np.ndarray | None = None


def _accumulate(target: np.ndarray | None, addend) -> np.ndarray:
  """Sum ``addend`` into ``target`` (over species/blocks), copying on first
  use so the caller's array is never mutated in place."""
  addend = np.asarray(addend)
  return addend.copy() if target is None else target + addend


def balance_error(fdot: np.ndarray,
                  src: np.ndarray,
                  bflux_tot: np.ndarray,
                  field_dot: np.ndarray | None = None,
                  apar_dot: np.ndarray | None = None) -> np.ndarray:
  """The balance residual: ``S - bflux - (df/dt [- dfield/dt] [- dapar/dt])``.

  Pure array arithmetic -- the one formula every balance trace (particle or
  energy, single- or multi-block, single- or multi-species) reduces to once
  the per-species/per-block sums are in hand. The field terms only enter the
  energy balance.
  """
  rate = fdot
  if field_dot is not None:
    rate = rate - field_dot
  if apar_dot is not None:
    rate = rate - apar_dot
  return src - bflux_tot - rate


def _block_prefix(file_prefix: str, block_idx: int) -> str:
  return file_prefix.replace("*", str(block_idx))


def _moment_file(stem: str, quantity: _Quantity) -> str:
  return utils.integrated_moment_file(stem, quantity.moments)


def _resolve(path: str,
             override: str | None,
             default: str,
             block_idx: int,
             species: str | None = None) -> str:
  """Resolve a file-family member's path: ``override`` (with ``*``
  substituted for the block index, then the species) if given, else the
  naming-convention ``default``."""
  if override is None:
    return default
  resolved = (path + override).replace("*", str(block_idx), 1)
  if species is not None:
    resolved = resolved.replace("*", species)
  return resolved


def _read_trace(file_name: str) -> np.ndarray:
  """Values of a required time trace; a single sample is squeezed to 0-D."""
  found, _, values, _ = utils.read_time_trace_if_present(file_name)
  if not found:
    raise FileNotFoundError(f"Required file not found: {file_name}")
  return np.atleast_1d(values)


def _read_moment(file_name: str, comp: int) -> tuple[np.ndarray, np.ndarray]:
  """``(time, values)`` of a required integrated-moment component."""
  found, time, values = utils.read_moment_if_present(file_name, comp)
  if not found:
    raise FileNotFoundError(f"Required file not found: {file_name}")
  return time, values


def _moment_file_traces(
    path: str,
    file_prefix: str,
    species: list[str],
    quantity: _Quantity,
    multib: str,
    relative_error: bool,
    fdot_file: str | None,
    source_file: str | None,
    bflux_files: dict[str, str],
    f_file: str | None,
    dt_file: str | None,
    field_dot_file: str | None,
    apar_dot_file: str | None,
    field_file: str | None,
    apar_file: str | None,
) -> BalanceTraces:
  """Balance traces from the integrated-moment (and, for the energy,
  field-energy) files, summed over species and blocks."""
  comp = quantity.comp
  probe = fdot_file or _moment_file(file_prefix + species[0] + "_fdot_",
                                    quantity)
  blocks = utils.get_block_indices(multib, probe)

  field_dot = apar_dot = fdot = src = src_bgk = bflux_tot = None
  has_src = has_src_bgk = has_bflux = False
  time_fdot = None

  for block_idx in blocks:
    block_prefix = _block_prefix(file_prefix, block_idx)

    if quantity.has_field:
      field_dot = _accumulate(
          field_dot,
          _read_trace(
              _resolve(path, field_dot_file,
                       block_prefix + "field_energy_dot.gkyl", block_idx)))
      found, _, v, _ = utils.read_time_trace_if_present(
          _resolve(path, apar_dot_file, block_prefix + "apar_energy_dot.gkyl",
                   block_idx))
      if found:
        apar_dot = _accumulate(apar_dot, np.atleast_1d(v))

    for sp in species:
      time_fdot, fdot_sp = _read_moment(
          _resolve(path, fdot_file,
                   _moment_file(block_prefix + sp + "_fdot_", quantity),
                   block_idx, sp), comp)

      found_s, _, v = utils.read_moment_if_present(
          _resolve(path, source_file,
                   _moment_file(block_prefix + sp + "_source_", quantity),
                   block_idx, sp), comp)
      has_src = has_src or found_s
      src = _accumulate(src, v if found_s else 0.0 * fdot_sp)

      found_s, _, v = utils.read_moment_if_present(
          _moment_file(block_prefix + sp + "_source_bgk_", quantity), comp)
      has_src_bgk = has_src_bgk or found_s
      src_bgk = _accumulate(src_bgk, v if found_s else 0.0 * fdot_sp)

      bflux_terms = []
      for d in _DIRS:
        for e in _EDGES:
          found_b, _, v = utils.read_moment_if_present(
              _resolve(
                  path, bflux_files.get(d + e),
                  _moment_file(block_prefix + sp + f"_bflux_{d}{e}_", quantity),
                  block_idx, sp), comp)
          if found_b:
            has_bflux = True
            bflux_terms.append(v)
      bflux_tot = _accumulate(
          bflux_tot,
          sum(bflux_terms) if bflux_terms else 0.0 * fdot_sp)

      fdot = _accumulate(fdot, fdot_sp)

  # Rates are only defined once a step has been taken: skip t=0.
  time = time_fdot[1:]
  fdot, src, src_bgk, bflux_tot = fdot[1:], src[1:], src_bgk[1:], bflux_tot[1:]
  field_dot = field_dot[1:] if field_dot is not None else None
  apar_dot = apar_dot[1:] if apar_dot is not None else None
  mom_err = balance_error(fdot, src + src_bgk, bflux_tot, field_dot, apar_dot)

  mom_err_norm = None
  if relative_error:
    dt = _read_trace(
        _resolve(path, dt_file,
                 file_prefix.replace("_b*", "") + "dt.gkyl", 0))
    # The conserved quantity: the moments of f, minus the field energies.
    conserved = None
    for block_idx in blocks:
      block_prefix = _block_prefix(file_prefix, block_idx)
      for sp in species:
        _, v = _read_moment(
            _resolve(path, f_file,
                     _moment_file(block_prefix + sp + "_", quantity), block_idx,
                     sp), comp)
        conserved = _accumulate(conserved, v)
      if quantity.has_field:
        conserved = conserved - _read_trace(
            _resolve(path, field_file, block_prefix + "field_energy.gkyl",
                     block_idx))
        found, _, v, _ = utils.read_time_trace_if_present(
            _resolve(path, apar_file, block_prefix + "apar_energy.gkyl",
                     block_idx))
        if found:
          conserved = conserved - np.atleast_1d(v)
    mom_err_norm = mom_err * dt / conserved[1:]
    mom_err = None

  return BalanceTraces(time=time,
                       fdot=fdot,
                       src=src if has_src else None,
                       bflux_tot=bflux_tot if has_bflux else None,
                       mom_err=mom_err,
                       mom_err_norm=mom_err_norm,
                       src_bgk=src_bgk if has_src_bgk else None,
                       field_dot=field_dot,
                       apar_dot=apar_dot)


def _balance_file_traces(file_names: list[str], quantity: _Quantity,
                         relative_error: bool) -> BalanceTraces:
  """Balance traces from Gkeyll's balance files, one per block (and, for
  per-species files, species)."""
  time, col = utils.read_balance_files(file_names)
  if relative_error and "mom_err_norm" not in col:
    raise ValueError(
        f"The relative {quantity.noun} error of several blocks or species "
        "needs the moment files: each balance file normalizes by its own "
        f"{quantity.noun} content.")
  # Rates are only defined once a step has been taken: skip t=0.
  col = {key: values[1:] for key, values in col.items()}

  def nonzero(column: str) -> np.ndarray | None:
    """The column, or ``None`` if it is absent or zero throughout."""
    values = col.get(column)
    return values if values is not None and np.any(values) else None

  # Older balance files have no src_bgk_mom: their src_mom includes it.
  return BalanceTraces(
      time=time[1:],
      fdot=col["mom_dot"],
      src=nonzero("src_mom"),
      bflux_tot=nonzero("bflux_tot_mom"),
      mom_err=None if relative_error else col["mom_err"],
      mom_err_norm=col["mom_err_norm"] if relative_error else None,
      src_bgk=nonzero("src_bgk_mom"),
      field_dot=col.get("field_dot"),
      bg_dot=nonzero("bg_dot"))


def _plot(traces: BalanceTraces, quantity: _Quantity, relative_error: bool,
          logy: bool, absy: bool, xlabel: str, ylabel: str | None,
          title: str | None, indent_left: float,
          add_width: float) -> plt.Figure:
  """Plot every balance term and the residual, or the relative residual."""
  fig = plt.figure(figsize=(7.5, 4.5))
  ax = fig.add_axes([0.11 + indent_left, 0.15, 0.87 + add_width, 0.78])
  ax.plot([-1.0, 1.0], [0.0, 0.0], color="grey", linestyle=":", linewidth=1)

  absy_func = np.abs if absy else (lambda v: v)
  time = traces.time
  err_symbol = r"E_{\dot{" + quantity.symbol + r"}}"

  if not relative_error:
    legend_handles = []
    legend_strings = []

    def add(values, label, **style):
      h, = ax.plot(time, absy_func(values), **style)
      legend_handles.append(h)
      legend_strings.append(label)

    if traces.src is not None:
      add(traces.src, r"$\mathcal{S}$", linestyle=_LINE_STYLES[2])
    if traces.src_bgk is not None:
      add(traces.src_bgk,
          r"$\mathcal{S}_{\mathrm{BGK}}$",
          linestyle=_LINE_STYLES[2],
          marker="o",
          markevery=8,
          fillstyle="none")
    if traces.bflux_tot is not None:
      add(-traces.bflux_tot,
          r"$-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f$",
          linestyle=_LINE_STYLES[1])
    if traces.field_dot is not None:
      add(-traces.field_dot,
          r"$-\dot{\phi}$",
          linestyle=":",
          marker="+",
          markevery=8)
    if traces.apar_dot is not None:
      add(-traces.apar_dot,
          r"$-\dot{A}_{\parallel}$",
          linestyle=":",
          marker="+",
          markevery=8)
    if traces.bg_dot is not None:
      add(-traces.bg_dot,
          r"$-\frac{\mathrm{d}}{\mathrm{d}t}\int\rho_{bg}\phi$",
          linestyle=":",
          marker="x",
          markevery=8)
    add(-traces.fdot, r"$-\dot{f}$", linestyle=_LINE_STYLES[0])
    add(traces.mom_err,
        f"${err_symbol}=$" + utils.sum_label(legend_strings),
        linestyle=_LINE_STYLES[3])

    ax.legend(legend_handles,
              legend_strings,
              fontsize=_LEGEND_FONT_SIZE,
              frameon=False)

    ylabel_string = ylabel or ""
    title_string = title or f"{quantity.noun.capitalize()} balance"
  else:
    ax.plot(time, absy_func(traces.mom_err_norm))

    ylabel_string = ylabel or f"${err_symbol}~\\Delta t/{quantity.symbol}$"
    title_string = title or f"Relative error in {quantity.noun} conservation"

  if logy:
    ax.set_yscale("log")
  if absy and ylabel_string:
    ylabel_string = r"|" + ylabel_string + r"|"

  ax.set_xlabel(xlabel, fontsize=_XY_LABEL_FONT_SIZE)
  ax.set_ylabel(ylabel_string, fontsize=_XY_LABEL_FONT_SIZE)
  ax.set_title(title_string, fontsize=_TITLE_FONT_SIZE)
  ax.set_xlim(time[0], time[-1])
  utils.set_tick_font_size(ax, _TICK_FONT_SIZE)
  return fig


def balance(
    name: str,
    species: list[str] | None = None,
    *,
    quantity: Literal["particle", "energy"] = "particle",
    path: str = "./",
    relative_error: bool = False,
    multib: str = "-10",
    fdot_file: str | None = None,
    source_file: str | None = None,
    bflux_files: dict[str, str] | None = None,
    f_file: str | None = None,
    dt_file: str | None = None,
    field_dot_file: str | None = None,
    apar_dot_file: str | None = None,
    field_file: str | None = None,
    apar_file: str | None = None,
    logy: bool = False,
    absy: bool = False,
    xlabel: str = "Time (s)",
    ylabel: str | None = None,
    title: str | None = None,
    indent_left: float = 0.0,
    add_width: float = 0.0,
    show: bool = False,
    saveas: str | None = None,
) -> tuple[plt.Figure, BalanceTraces]:
  """Plot (and compute) the particle or energy balance of a gyrokinetic run.

  Reads, per species, the integrated moments
  ``<name>-<species>_<kind>integrated_<moment>.gkyl`` of the first
  ``<moment>`` holding the balanced quantity (or the older
  ``<kind>integrated_moms.gkyl``): the density is the first component of
  ``M0``, ``M0M1M2``, ``M0M1M2parM2perp`` and ``HamiltonianMoments``, the
  energy the third component of ``HamiltonianMoments``. ``<kind>`` is
  ``fdot_`` (required) and, only if the run had sources or non-periodic
  boundaries, ``source_``, ``source_bgk_`` and ``bflux_<direction><side>_``.
  The energy balance also requires ``<name>-field_energy_dot.gkyl`` and reads
  ``<name>-apar_energy_dot.gkyl`` if present (electromagnetic simulations).
  If ``relative_error`` is requested, the moments of ``f`` (no ``<kind>``)
  and ``<name>-dt.gkyl`` are also required, and for the energy
  ``field_energy.gkyl`` (and ``apar_energy.gkyl`` if present).

  If no path override is given and one of these rate files is missing, the
  terms are instead read from the balance files Gkeyll writes when a species
  or the app requests the balance: ``<name>-<species>_balance_particle.gkyl``
  summed over the species, and ``<name>-balance_energy.gkyl``, which already
  sums over every species (``species`` is not used) and adds the
  background-charge term ``bg_dot`` of adiabatic species to the residual.
  Their relative error is only available for a single file. Older balance
  files have no ``src_bgk_mom`` column: their ``src_mom`` includes the BGK
  source. Without ``species``, only ``<name>-balance_<quantity>.gkyl``,
  summed over species, is read.

  The rates are only defined once a step has been taken, so t=0 is skipped.

  Args:
    name: Simulation name (also the file prefix).
    species: Species names to sum over; if omitted, only the app's balance
      file (summed over species) is read.
    quantity: The conserved quantity whose balance is computed.
    path: Directory holding the simulation output.
    relative_error: Plot the relative error instead of every balance term.
    multib: ``"-10"`` (default) for a single block; ``"-1"`` to discover
      every block; otherwise a comma list or ``'start:stop[:step]'`` slice
      of block indices (see :func:`~postgkyl.diagnostics.gk.utils.
      get_block_indices`).
    fdot_file: Explicit distribution derivative moments path override.
    source_file: Explicit source moments path override (not of the BGK
      source).
    bflux_files: Optional per-boundary path overrides, keyed by
      ``"<direction><side>"`` (e.g. ``"xlower"``); unlisted boundaries use
      the naming convention.
    f_file: Explicit integrated distribution moments path override.
    dt_file: Explicit time-step history path override.
    field_dot_file: Explicit field-energy derivative path override (energy
      balance only).
    apar_dot_file: Explicit parallel-vector-potential energy derivative path
      (energy balance only).
    field_file: Explicit field-energy path override (energy balance only).
    apar_file: Explicit parallel-vector-potential energy path override
      (energy balance only). In path overrides, ``*`` stands for block index
      and, for per-species files, species name.
    logy: Log-scale the y axis.
    absy: Take the absolute value of every trace before plotting.
    xlabel: Horizontal-axis label.
    ylabel: Vertical-axis label; use the derived formula when ``None``.
    title: Figure title; use the balance description when ``None``.
    indent_left: Horizontal axes-position adjustment in figure units.
    add_width: Axes-width adjustment in figure units.
    show: Call ``plt.show()`` before returning.
    saveas: If given, save the figure to this path.

  Returns:
    ``(figure, traces)``.

  Raises:
    FileNotFoundError: if a required file family is missing.
    ValueError: if the relative error of several blocks or species is
      requested from balance files, a path override is given without
      ``species``, a field-energy override is given for the particle
      balance, or no step was taken.
  """
  spec = _QUANTITIES[quantity]
  path = path.rstrip("/") + "/"
  file_prefix = f"{path}{name}-" if multib == "-10" else f"{path}{name}_b*-"

  field_overrides = (field_dot_file, apar_dot_file, field_file, apar_file)
  if not spec.has_field and any(field_overrides):
    raise ValueError("Field-energy path overrides only apply to the energy "
                     "balance.")
  overrides = (fdot_file, source_file, bflux_files, f_file, dt_file,
               *field_overrides)
  if not species and any(overrides):
    raise ValueError("Path overrides select the moment files, which need the "
                     "species.")

  # The per-species balance files lack the field terms of the energy balance.
  if species and not spec.has_field:
    balance_names = [
        f"{file_prefix}{sp}_balance_{quantity}.gkyl" for sp in species
    ]
  else:
    balance_names = [f"{file_prefix}balance_{quantity}.gkyl"]
  # The rate files the balance is computed from when they are all present.
  moment_files = [
      _moment_file(file_prefix + sp + "_fdot_", spec) for sp in species or []
  ]
  if species and spec.has_field:
    moment_files.append(file_prefix + "field_energy_dot.gkyl")
  blocks = [] if any(overrides) else utils.balance_file_blocks(
      multib, balance_names, moment_files)

  if blocks:
    traces = _balance_file_traces([
        file_name.replace("*", str(b)) for b in blocks
        for file_name in balance_names
    ], spec, relative_error)
  elif not species:
    raise FileNotFoundError(f"Required file not found: {balance_names[0]}")
  else:
    traces = _moment_file_traces(path, file_prefix, species, spec, multib,
                                 relative_error, fdot_file, source_file,
                                 bflux_files or {}, f_file, dt_file,
                                 field_dot_file, apar_dot_file, field_file,
                                 apar_file)

  if traces.time.size == 0:
    raise ValueError("No time step was taken: the balance is only defined "
                     "after t=0.")
  fig = _plot(traces, spec, relative_error, logy, absy, xlabel, ylabel, title,
              indent_left, add_width)
  if saveas:
    fig.savefig(saveas)
  if show:
    plt.show()
  return fig, traces
