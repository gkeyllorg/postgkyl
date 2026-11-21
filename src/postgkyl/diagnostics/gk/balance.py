"""Gyrokinetic particle- and energy-balance diagnostic.

Sums over species and blocks the integrated moments a gyrokinetic run writes
(of ``df/dt``, the sources and the boundary fluxes; for the energy, also the
field and apar energy rates of change), and plots the balance residual::

    N_err = S + S_BGK - bflux - df/dt
    E_err = S + S_BGK - bflux - (df/dt - dfield/dt [- dapar/dt])

with ``S`` the source and ``S_BGK`` the BGK source. Without those files, it
reads the balance files Gkeyll writes,
``<name>-[<species>_]balance_<quantity>.gkyl``. The app's energy balance file
also holds the background-charge term of adiabatic species, ``+ dbg/dt`` in
the parentheses.
"""

from __future__ import annotations

import glob
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from . import utils

_BOUNDARIES = tuple(d + e for d in "xyz" for e in ("lower", "upper"))
_LINE_STYLES = ("-", "--", ":", "-.")
# Integrated moments holding each quantity, by preference, and its component:
# the density is the first component of these moments, the energy (H) the
# third of the Hamiltonian moments (M0, M1, H).
_MOMENTS = {
    "particle": (("M0", "M0M1M2", "M0M1M2parM2perp", "HamiltonianMoments"), 0),
    "energy": (("HamiltonianMoments", ), 2),
}
_SYMBOLS = {"particle": r"\mathcal{N}", "energy": r"\mathcal{E}"}
# Plotted terms: (trace, sign, label, line style), in legend order.
_TERMS = (
    ("src", 1, r"$\mathcal{S}$", dict(linestyle=":")),
    ("src_bgk", 1, r"$\mathcal{S}_{\mathrm{BGK}}$",
     dict(linestyle=":", marker="o", markevery=8, fillstyle="none")),
    ("bflux_tot", -1,
     r"$-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f$",
     dict(linestyle="--")),
    ("field_dot", 1, r"$\dot{\phi}$",
     dict(linestyle=":", marker="+", markevery=8)),
    ("apar_dot", 1, r"$\dot{A}_{\parallel}$",
     dict(linestyle=":", marker="+", markevery=8)),
    ("bg_dot", -1, r"$-\frac{\mathrm{d}}{\mathrm{d}t}\int\rho_{bg}\phi$",
     dict(linestyle=":", marker="x", markevery=8)),
    ("fdot", -1, r"$-\dot{f}$", dict(linestyle="-")),
)
_TRACES = ("time", "fdot", "src", "src_bgk", "bflux_tot", "field_dot",
           "apar_dot", "bg_dot", "mom_err", "mom_err_norm",
           "mom_err_norm_parts")


def balance_error(fdot: np.ndarray,
                  src: np.ndarray,
                  bflux_tot: np.ndarray,
                  field_dot: np.ndarray | None = None,
                  apar_dot: np.ndarray | None = None) -> np.ndarray:
  """The balance residual: ``S - bflux - (df/dt [- dfield/dt] [- dapar/dt])``.

  The field terms only enter the energy balance.
  """
  rate = fdot
  for field_rate in (field_dot, apar_dot):
    if field_rate is not None:
      rate = rate - field_rate
  return src - bflux_tot - rate


def _accumulate(target: np.ndarray | None, addend) -> np.ndarray:
  """Sum ``addend`` into ``target``, copying on first use so the caller's
  array is never mutated in place."""
  addend = np.asarray(addend)
  return addend.copy() if target is None else target + addend


def _resolve(path: str,
             override: str | None,
             default: str,
             block_idx: int,
             species: str | None = None) -> str:
  """``override`` (``*`` substituted by the block index, then the species)
  if given, else ``default``."""
  if override is None:
    return default
  resolved = (path + override).replace("*", str(block_idx), 1)
  return resolved if species is None else resolved.replace("*", species)


def _read(
    file_name: str,
    comp: int = 0,
    required: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
  """``(time, values)`` of component ``comp`` of a time trace, or
  ``(None, None)`` if the file is absent and not ``required``."""
  if not os.path.exists(file_name):
    if required:
      raise FileNotFoundError(f"Required file not found: {file_name}")
    return None, None
  time, values = utils.read_time_trace(file_name)
  return time, values[:, comp]


def _moment_file(stem: str, quantity: str) -> str:
  """The first existing ``<stem>integrated_<moment>.gkyl`` holding the
  quantity (or the older ``<stem>integrated_moms.gkyl``), else the first
  candidate."""
  names = [
      f"{stem}integrated_{m}.gkyl" for m in (*_MOMENTS[quantity][0], "moms")
  ]
  return next((n for n in names if glob.glob(n)), names[0])


SourceMoments = Literal["HamiltonianMoments", "M0M1M2", "M0M1M2parM2perp"]


def _source_rate(file_name: str, quantity: str,
                 source_moments: SourceMoments) -> np.ndarray | None:
  """A source's rate of the quantity, or ``None`` if the file is absent.

  An older ``<kind>integrated_moms.gkyl`` file does not say which moments it
  integrates (the app names it so whatever the input chose), so for the
  energy ``source_moments`` says: the Hamiltonian's third component, or
  ``m/2`` times ``M2`` (``M0M1M2``) or ``M2par + M2perp``
  (``M0M1M2parM2perp``), with the species mass from the file's metadata.
  The latter two leave out ``int q phi S``, which vanishes for a
  quasineutral source.
  """
  comp = _MOMENTS[quantity][1]
  legacy = file_name.endswith("integrated_moms.gkyl")
  if quantity != "energy" or source_moments == "HamiltonianMoments" or not legacy:
    return _read(file_name, comp)[1]
  if not os.path.exists(file_name):
    return None
  grid, values, gdata = utils.read_gfile(file_name)
  values = np.asarray(values).reshape(np.atleast_1d(grid[0]).size, -1)
  m2 = values[:, 2] + (values[:, 3]
                       if source_moments == "M0M1M2parM2perp" else 0.0)
  return 0.5 * float(gdata.ctx["mass"]) * m2


def _moment_file_traces(
    path: str,
    file_prefix: str,
    species: list[str],
    quantity: str,
    multib: str,
    relative_error: bool,
    overrides: dict[str, str | None],
    source_moments: SourceMoments = "HamiltonianMoments") -> dict:
  """Traces from the integrated-moment (and, for the energy, field-energy)
  files, summed over species and blocks. ``overrides`` maps a file's kind
  (``fdot_``, ``source_``, ``bflux_<boundary>_``, ``""`` for f, or the
  field-file name such as ``field_energy_dot``) to its path override."""
  comp = _MOMENTS[quantity][1]
  has_field = quantity == "energy"

  def moment(block_prefix, kind, sp, block):
    return _resolve(path, overrides.get(kind),
                    _moment_file(f"{block_prefix}{sp}_{kind}", quantity), block,
                    sp)

  def field(block_prefix, kind, block):
    return _resolve(path, overrides.get(kind), f"{block_prefix}{kind}.gkyl",
                    block)

  fdot_file = overrides.get("fdot_")
  blocks = utils.get_block_indices(
      multib, path + fdot_file if fdot_file else _moment_file(
          f"{file_prefix}{species[0]}_fdot_", quantity))
  sums, found, time = {}, set(), None

  def add(key, values):
    sums[key] = _accumulate(sums.get(key), values)

  for block in blocks:
    block_prefix = file_prefix.replace("*", str(block))
    if has_field:
      add(
          "field_dot",
          _read(field(block_prefix, "field_energy_dot", block),
                required=True)[1])
      _, apar_dot = _read(field(block_prefix, "apar_energy_dot", block))
      if apar_dot is not None:
        found.add("apar_dot")
        add("apar_dot", apar_dot)
    for sp in species:
      time, fdot = _read(moment(block_prefix, "fdot_", sp, block), comp, True)
      add("fdot", fdot)
      for key, kinds in (("src", ("source_", )), ("src_bgk", ("source_bgk_", )),
                         ("bflux_tot", [f"bflux_{b}_" for b in _BOUNDARIES])):
        for kind in kinds:
          file_name = moment(block_prefix, kind, sp, block)
          values = (_source_rate(file_name, quantity, source_moments)
                    if key != "bflux_tot" else _read(file_name, comp)[1])
          if values is not None:
            found.add(key)
          add(key, 0.0 * fdot if values is None else values)

  # Rates are only defined once a step has been taken: skip t=0.
  rates = {key: values[1:] for key, values in sums.items()}
  mom_err = balance_error(rates["fdot"], rates["src"] + rates["src_bgk"],
                          rates["bflux_tot"], rates.get("field_dot"),
                          rates.get("apar_dot"))
  traces = {
      key: rates[key] if key in found else None
      for key in ("src", "src_bgk", "bflux_tot", "apar_dot")
  }
  traces.update(time=time[1:],
                fdot=rates["fdot"],
                field_dot=rates.get("field_dot"))
  if not relative_error:
    return dict.fromkeys(_TRACES) | traces | {"mom_err": mom_err}

  _, dt = _read(_resolve(path, overrides.get("dt"),
                         file_prefix.replace("_b*", "") + "dt.gkyl", 0),
                required=True)
  # The conserved quantity: the moments of f, minus the field energies.
  conserved = None
  for block in blocks:
    block_prefix = file_prefix.replace("*", str(block))
    for sp in species:
      conserved = _accumulate(
          conserved,
          _read(moment(block_prefix, "", sp, block), comp, True)[1])
    if has_field:
      conserved = conserved - _read(field(block_prefix, "field_energy", block),
                                    required=True)[1]
      _, apar = _read(field(block_prefix, "apar_energy", block))
      if apar is not None:
        conserved = conserved - apar
  return dict.fromkeys(_TRACES) | traces | {
      "mom_err_norm": mom_err * dt / conserved[1:]
  }


def _read_balance_files(
    file_names: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
  """``(time, columns)`` of Gkeyll balance files, columns keyed by their
  ``Columns`` metadata and summed over the files. ``mom_err_norm``, which
  each file normalizes by its own content, is only kept for a single file.

  Raises:
    ValueError: if a file has no ``Columns`` metadata.
  """
  time, columns = None, {}
  for file_name in file_names:
    grid, values, gdata = utils.read_gfile(file_name)
    names = gdata.ctx.get("Columns")
    if names is None:
      raise ValueError(f"No 'Columns' metadata in {file_name}")
    # A single sample loses its time axis to the squeeze in read_gfile.
    time, values = np.atleast_1d(grid[0]), np.atleast_2d(values)
    for comp, column in enumerate(c.strip() for c in names.split(",")):
      columns[column] = columns.get(column, 0.0) + values[:, comp]
  if len(file_names) > 1:
    del columns["mom_err_norm"]
  return time, columns


def _balance_file_traces(files: dict[str, str], relative_error: bool) -> dict:
  """Traces from Gkeyll's balance files, keyed by the block and/or species
  each holds, summed over them. With several files, their relative residuals
  stay separate in ``mom_err_norm_parts``."""
  time, col = _read_balance_files(list(files.values()))
  # Rates are only defined once a step has been taken: skip t=0.
  col = {key: values[1:] for key, values in col.items()}
  parts = None
  if relative_error and "mom_err_norm" not in col:
    parts = {
        label: _read_balance_files([file_name])[1]["mom_err_norm"][1:]
        for label, file_name in files.items()
    }

  def nonzero(column):
    values = col.get(column)
    return values if values is not None and np.any(values) else None

  # Older balance files have no src_bgk_mom: their src_mom includes it.
  return dict.fromkeys(_TRACES) | {
      "time": time[1:],
      "fdot": col["mom_dot"],
      "src": nonzero("src_mom"),
      "src_bgk": nonzero("src_bgk_mom"),
      "bflux_tot": nonzero("bflux_tot_mom"),
      "field_dot": col.get("field_dot"),
      "bg_dot": nonzero("bg_dot"),
      "mom_err": None if relative_error else col["mom_err"],
      "mom_err_norm": col.get("mom_err_norm") if relative_error else None,
      "mom_err_norm_parts": parts,
  }


def _balance_file_blocks(multib: str, balance_files: list[str],
                         moment_files: list[str]) -> list[int]:
  """The selected blocks if each has every balance file but one of the
  ``moment_files`` the balance is otherwise computed from is missing; else
  ``[]`` (read the moment files). Paths have ``*`` for the block index."""

  def present(file_name: str, blocks: list[int]) -> bool:
    return bool(blocks) and all(
        os.path.exists(file_name.replace("*", str(b))) for b in blocks)

  if moment_files:
    moment_blocks = utils.get_block_indices(multib, moment_files[0])
    if all(present(f, moment_blocks) for f in moment_files):
      return []
  blocks = utils.get_block_indices(multib, balance_files[0])
  return blocks if all(present(f, blocks) for f in balance_files) else []


def _sum_label(terms: list[str]) -> str:
  """Mathtext label of the sum of terms given by their labels: ``+`` before
  every term but the first that does not start with a minus sign."""
  return "".join(t if i == 0 or t.startswith("$-") else "$+" + t[1:]
                 for i, t in enumerate(terms))


def _plot(traces: dict, quantity: str, relative_error: bool, logy: bool,
          absy: bool, xlabel: str, ylabel: str | None,
          title: str | None) -> plt.Figure:
  """Plot every balance term and the residual, or the relative residual."""
  # The constrained layout keeps the labels inside the figure.
  fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
  ax.plot([-1.0, 1.0], [0.0, 0.0], color="grey", linestyle=":", linewidth=1)
  absy_func = np.abs if absy else (lambda v: v)
  time = traces["time"]
  err_symbol = r"E_{\dot{" + _SYMBOLS[quantity] + r"}}"

  if not relative_error:
    labels = []
    for key, sign, label, style in _TERMS:
      if traces[key] is not None:
        ax.plot(time, absy_func(sign * traces[key]), label=label, **style)
        labels.append(label)
    ax.plot(time,
            absy_func(traces["mom_err"]),
            label=f"${err_symbol}=$" + _sum_label(labels),
            linestyle="-.")
    ax.legend(fontsize=14, frameon=False)
    ylabel = ylabel or ""
    title = title or f"{quantity.capitalize()} balance"
  else:
    # Several balance files are each normalized by their own content.
    parts = traces["mom_err_norm_parts"] or {None: traces["mom_err_norm"]}
    for i, (label, values) in enumerate(parts.items()):
      # Matplotlib cycles 10 colors: change the line style after each cycle.
      ax.plot(time,
              absy_func(values),
              label=label,
              linestyle=_LINE_STYLES[i // 10 % len(_LINE_STYLES)])
    if len(parts) > 1:
      ax.legend(fontsize=10, frameon=False, ncol=1 + (len(parts) - 1) // 4)
    ylabel = ylabel or f"${err_symbol}~\\Delta t/{_SYMBOLS[quantity]}$"
    title = title or f"Relative error in {quantity} conservation"

  if logy:
    ax.set_yscale("log")
  ax.set_xlabel(xlabel, fontsize=17)
  ax.set_ylabel(f"|{ylabel}|" if absy and ylabel else ylabel, fontsize=17)
  ax.set_title(title, fontsize=17)
  ax.set_xlim(time[0], time[-1])
  utils.set_tick_font_size(ax, 14)
  return fig


def balance(
    name: str,
    species: list[str] | None = None,
    *,
    quantity: Literal["particle", "energy"] = "particle",
    path: str = "./",
    relative_error: bool = False,
    source_moments: SourceMoments = "HamiltonianMoments",
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
    show: bool = False,
    saveas: str | None = None,
) -> tuple[plt.Figure, dict]:
  """Plot (and compute) the particle or energy balance of a gyrokinetic run.

  Reads, per species, the integrated moments
  ``<name>-<species>_<kind>integrated_<moment>.gkyl`` of the first
  ``<moment>`` holding the quantity (or the older
  ``<kind>integrated_moms.gkyl``): the density is the first component of
  ``M0``, ``M0M1M2``, ``M0M1M2parM2perp`` and ``HamiltonianMoments``, the
  energy the third of ``HamiltonianMoments``. ``<kind>`` is ``fdot_``
  (required) and, if present, ``source_``, ``source_bgk_`` and
  ``bflux_<direction><side>_``. The energy balance also requires
  ``<name>-field_energy_dot.gkyl`` and reads ``<name>-apar_energy_dot.gkyl``
  if present. ``relative_error`` also requires the moments of ``f`` (no
  ``<kind>``), ``<name>-dt.gkyl`` and, for the energy, ``field_energy.gkyl``
  (and ``apar_energy.gkyl`` if present).

  Without a path override, if one of the rate files is missing, the terms are
  read from Gkeyll's balance files instead:
  ``<name>-<species>_balance_particle.gkyl``, summed over the species, or
  ``<name>-balance_energy.gkyl``, already summed over every species and with
  the background-charge term ``bg_dot`` of adiabatic species. With several
  files (blocks or species), each normalizes its relative error by its own
  content, so these stay separate (``mom_err_norm_parts``). Older balance
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
    source_moments: What an older ``<species>_source[_bgk]_integrated_moms``
      file holds (its name does not say): for the energy balance, its
      Hamiltonian moments, or ``M0M1M2``/``M0M1M2parM2perp`` moments whose
      energy is ``m/2 M2``. Files named for their moments are unaffected.
    multib: ``"-10"`` (default) for a single block; ``"-1"`` to discover
      every block; otherwise a comma list or ``'start:stop[:step]'`` slice of
      block indices.
    fdot_file: Explicit distribution derivative moments path override.
    source_file: Explicit source moments path override (not of the BGK
      source).
    bflux_files: Optional per-boundary path overrides, keyed by
      ``"<direction><side>"`` (e.g. ``"xlower"``).
    f_file: Explicit integrated distribution moments path override.
    dt_file: Explicit time-step history path override.
    field_dot_file: Explicit field-energy derivative path override (energy
      balance only).
    apar_dot_file: Explicit parallel-vector-potential energy derivative path
      (energy balance only).
    field_file: Explicit field-energy path override (energy balance only).
    apar_file: Explicit parallel-vector-potential energy path override
      (energy balance only). In path overrides, ``*`` stands for the block
      index and, for per-species files, the species name.
    logy: Log-scale the y axis.
    absy: Take the absolute value of every trace before plotting.
    xlabel: Horizontal-axis label.
    ylabel: Vertical-axis label; use the derived formula when ``None``.
    title: Figure title; use the balance description when ``None``.
    show: Call ``plt.show()`` before returning.
    saveas: If given, save the figure to this path.

  Returns:
    ``(figure, traces)``. ``traces`` maps names to 1-D arrays aligned to
    ``traces["time"]`` (after t=0): ``fdot`` (rate of change of the quantity
    in f), ``src``, ``src_bgk``, ``bflux_tot``, ``field_dot``, ``apar_dot``,
    ``bg_dot`` (``None`` when absent, or zero throughout in a balance file),
    and the residual ``mom_err``, or with ``relative_error`` its relative
    value ``mom_err_norm`` or, per balance file, ``mom_err_norm_parts`` (a
    dict keyed by block and/or species).

  Raises:
    FileNotFoundError: if a required file is missing.
    ValueError: if a path override is given without ``species``, a
      field-energy override is given for the particle balance, or no step was
      taken.
  """
  path = path.rstrip("/") + "/"
  file_prefix = f"{path}{name}-" if multib == "-10" else f"{path}{name}_b*-"
  has_field = quantity == "energy"
  overrides = {
      "fdot_": fdot_file,
      "source_": source_file,
      "": f_file,
      "dt": dt_file,
      "field_energy_dot": field_dot_file,
      "apar_energy_dot": apar_dot_file,
      "field_energy": field_file,
      "apar_energy": apar_file,
      **{
          f"bflux_{b}_": f
          for b, f in (bflux_files or {}).items()
      },
  }
  if not has_field and any(
      (field_dot_file, apar_dot_file, field_file, apar_file)):
    raise ValueError("Field-energy path overrides only apply to the energy "
                     "balance.")
  if not species and any(overrides.values()):
    raise ValueError("Path overrides select the moment files, which need the "
                     "species.")

  # Balance files by the species they hold (None: summed over species). The
  # per-species ones lack the field terms of the energy balance.
  if species and not has_field:
    balance_names = {
        sp: f"{file_prefix}{sp}_balance_{quantity}.gkyl"
        for sp in species
    }
  else:
    balance_names = {None: f"{file_prefix}balance_{quantity}.gkyl"}
  # The rate files the balance is computed from when they are all present.
  moment_files = [
      _moment_file(f"{file_prefix}{sp}_fdot_", quantity)
      for sp in species or []
  ]
  if species and has_field:
    moment_files.append(file_prefix + "field_energy_dot.gkyl")
  blocks = [] if any(overrides.values()) else _balance_file_blocks(
      multib, list(balance_names.values()), moment_files)

  if blocks:
    files = {}
    for b in blocks:
      for sp, file_name in balance_names.items():
        block = None if multib == "-10" else f"block {b}"
        label = " ".join(part for part in (block, sp) if part is not None)
        files[label] = file_name.replace("*", str(b))
    traces = _balance_file_traces(files, relative_error)
  elif not species:
    raise FileNotFoundError(f"Required file not found: {balance_names[None]}")
  else:
    traces = _moment_file_traces(path, file_prefix, species, quantity, multib,
                                 relative_error, overrides, source_moments)

  if traces["time"].size == 0:
    raise ValueError("No time step was taken: the balance is only defined "
                     "after t=0.")
  fig = _plot(traces, quantity, relative_error, logy, absy, xlabel, ylabel,
              title)
  if saveas:
    fig.savefig(saveas)
  if show:
    plt.show()
  return fig, traces
