"""Equation-blind comparison of two Gkeyll simulations' output directories.

Finds the output quantities two simulation runs have in common, ranks them
by how much they differ (:func:`postgkyl.numerics.max_abs_rel_diff`), and
plots the most-different ones (plus any the caller always wants) baseline
overlaid on candidate for 1-D/dynvector data, or side by side for 2-D
(including a ``select()``-ed slice of 3-D+ data), optionally also plotting
``sim0 - sim1`` alongside them via ``diff``. Shares a home with
:mod:`postgkyl.diagnostics.discovery` -- like it, this performs no equation-
specific physics and belongs to no model family -- rather than under
``gk``/``vm``/``pkpm``/``mom``. By default every figure is collected into one
multi-page PDF, via ``matplotlib.backends.backend_pdf.PdfPages`` (a plain
Matplotlib facility, not a new dependency); ``file_per_fig`` opts into
individual image files instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from postgkyl import io
from postgkyl.gdata.load import load
from postgkyl.gdatastate import materialize_point_values
from postgkyl.numerics import max_abs_rel_diff
from postgkyl.numerics.grid_centering import sample_coordinates
from postgkyl.numerics.idx_parser import idx_parser
from postgkyl.operations.interpolate import interpolate
from postgkyl.operations.select import select
from postgkyl.render.matplotlib import plot

from . import discovery

_DEFAULT_SUBDIR = "postgkyl_compare_sims"
_DEFAULT_PDF_NAME = "postgkyl_compare_sims.pdf"
_MAX_PANELS_PER_FIGURE = 6
_COMPARE_FIGSIZE = (12, 8)

# No quantity-name prefix is excluded by default: geometry (``geo_*``),
# per-boundary (``bc_*``), and flux (``bflux_*``) outputs are ranked and
# plottable like any other quantity. ``--exclude`` lets a caller skip
# specific prefixes on a case-by-case basis.
_DEFAULT_EXCLUDE_PREFIXES: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuantityDiff:
  """One shared quantity's difference ranking.

  Attributes:
    quantity: The bare quantity name (e.g. ``"field_energy"``,
      ``"elc_integrated_moms"`` -- the part of the filename after the sim
      name and block suffix; see :mod:`postgkyl.io.naming`).
    max_abs_diff: Largest absolute element-wise difference across every
      block/frame compared for this quantity.
    max_rel_diff: Largest relative element-wise difference, likewise.
  """

  quantity: str
  max_abs_diff: float
  max_rel_diff: float


@dataclass(frozen=True)
class CompareSimsReport:
  """What :func:`sims` found, ranked, and plotted.

  Attributes:
    compared: Every shared quantity actually diffed, ranked by
      ``max_rel_diff`` descending.
    plotted: The quantities a figure was saved for, in the order plotted.
    only_in_0: Quantities found only under ``path0``.
    only_in_1: Quantities found only under ``path1``.
    skipped: Shared quantities neither ranked nor plotted because loading or
      differencing them raised (e.g. a non-field, non-dynvector file like a
      multiblock ``block_topo``), mapped to the error message.
    out_dir: Directory the figures were written to (resolved, never
      ``None``, even when the caller left ``out_dir`` unset). By default
      (combined PDF), this is the directory the PDF was written to, not the
      PDF path itself.
  """

  compared: tuple[QuantityDiff, ...]
  plotted: tuple[str, ...]
  only_in_0: tuple[str, ...] = field(default_factory=tuple)
  only_in_1: tuple[str, ...] = field(default_factory=tuple)
  skipped: dict[str, str] = field(default_factory=dict)
  out_dir: str = ""


@dataclass(frozen=True)
class _ComparisonPlan:
  """Ranked selection and resolved inputs for one simulation pair."""

  quantities0: dict[str, str]
  quantities1: dict[str, str]
  frames: dict[str, int | None]
  report: CompareSimsReport


class _ReportDict(dict):
  """The dict ``sims()`` returns, identical in every way to a plain
  ``dict[str, CompareSimsReport]`` except for how it prints at the CLI
  edge: the CLI echoes a command's return value verbatim via ``repr()``
  (see ``postgkyl.cli.compiler._present``), and a full ``CompareSimsReport``
  per compared quantity can run to hundreds of lines -- unreadable for a
  real CI sweep with many quantities. ``verbose=False`` (the CLI default)
  prints a one-line-per-sim summary instead; ``--verbose``/``-v`` restores
  the full dict repr. Python callers of :func:`sims` are unaffected either
  way -- this is still a real ``dict``, just one whose __repr__ is overridden."""

  def __init__(self, data: dict[str, CompareSimsReport], *, verbose: bool):
    super().__init__(data)
    self._verbose = verbose

  def __repr__(self) -> str:
    if self._verbose:
      return dict.__repr__(self)
    if not self:
      return "{}"
    lines = [
        f"{name}: {len(report.plotted)} plotted, "
        f"{len(report.skipped)} skipped, out_dir={report.out_dir!r}"
        for name, report in self.items()
    ]
    lines.append("(pass --verbose/-v for the full per-quantity report)")
    return "\n".join(lines)


def _default_labels(path0: str, path1: str) -> list[str]:
  """The shortest distinguishing trailing path segments of ``path0``/
  ``path1`` (mirrors ``render.matplotlib._dataset_labels``) -- a bare
  basename is useless whenever both sims share a name, as a baseline and a
  candidate directory for the same simulation normally do."""
  parts0 = Path(os.path.normpath(path0)).parts
  parts1 = Path(os.path.normpath(path1)).parts
  depth = 1
  while (depth < len(parts0) and depth < len(parts1)
         and parts0[-depth:] == parts1[-depth:]):
    depth += 1
  return [os.path.join(*parts0[-depth:]), os.path.join(*parts1[-depth:])]


def _parse_label_pair(labels: str | None) -> list[str] | None:
  """Split ``"label0,label1"`` into the two labels ``sims()``'s ``-l``
  takes, or ``None`` unchanged so callers fall back to
  :func:`_default_labels`.

  A single comma-separated string for ``path0``/``path1``'s pair of labels,
  rather than a repeatable/JSON-array list option like ``plot``'s
  ``--legend_labels`` -- there are always exactly two labels here (one per
  compared sim), so one ``-l 'baseline,candidate'`` reads more directly than
  ``-l baseline -l candidate`` or a quoted JSON array would.
  """
  if labels is None:
    return None
  parts = labels.split(",")
  if len(parts) != 2 or not all(p.strip() for p in parts):
    raise ValueError(
        "labels must be exactly two comma-separated, non-empty labels, "
        f"e.g. 'baseline,candidate' (got {labels!r})")
  return [p.strip() for p in parts]


def _stems_by_quantity(path: str) -> dict[str, str]:
  """Map each distinct quantity name under ``path`` to one representative
  stem (block index stripped, since blocks of the same field compare as one
  quantity)."""
  out: dict[str, str] = {}
  for stem in discovery.find_output_stems("gkyl", path)["gkyl"]:
    name = io.parse_output_name(f"{stem}.gkyl")
    out.setdefault(name.quantity, stem)
  return out


def _is_excluded(quantity: str, exclude: tuple[str, ...]) -> bool:
  return any(quantity.startswith(prefix) for prefix in exclude)


def _block_paths(path: str, stem: str) -> list[str]:
  """Every per-block file path for ``stem`` (one entry for a single-block
  stem), sorted by block index."""
  name = io.parse_output_name(f"{stem}.gkyl")
  if name.block is None:
    return [os.path.join(path, f"{stem}.gkyl")]
  blocks = sorted(
      b for b in discovery.find_output_stems("gkyl", path)["gkyl"]
      if io.parse_output_name(f"{b}.gkyl").quantity == name.quantity)
  return [os.path.join(path, f"{b}.gkyl") for b in blocks]


def _latest_common_frame(path0: str, stem0: str, path1: str,
                         stem1: str) -> int | None:
  """The latest frame both sims wrote for a framed quantity family, or
  ``None`` when the family carries no frame index at all (e.g. a
  dynvector/time-series quantity, always one file with no trailing
  number)."""
  frames0 = discovery.available_frames(os.path.join(path0, f"{stem0}_"))
  frames1 = discovery.available_frames(os.path.join(path1, f"{stem1}_"))
  common = frames0 & frames1
  if not common and not frames0 and not frames1:
    return None
  if not common:
    raise ValueError(f"'{stem0}' has no frame in common between the two sims "
                     f"(found {sorted(frames0)} vs {sorted(frames1)})")
  return max(common)


def _load_quantity(path: str, stem: str, frame: int | None):
  """Load (and block-combine, if multiblock) one quantity family as a single
  dataset or list of per-block datasets."""
  paths = _block_paths(path, stem)
  suffix = f"_{frame}.gkyl" if frame is not None else ".gkyl"
  files = [p.removesuffix(".gkyl") + suffix for p in paths]
  return [load(f) for f in files]


def _point_values(data):
  """Interpolate/materialize ``data`` into plain point values for both
  plotting and differencing, leaving an already-evaluated dataset (a
  dynvector, or anything with no DG basis) untouched."""
  if data.ctx.get("basis_type") and not data.is_interpolated:
    data = interpolate(data)
  return materialize_point_values(data)


def _dominant_axes(values0: np.ndarray, values1: np.ndarray,
                   grid: list[np.ndarray],
                   free_axes: list[int]) -> dict[int, int]:
  """Which of ``free_axes`` to pin (and at what index) so at most 2 survive
  for plotting, chosen from the data rather than an arbitrary fixed axis.

  Finds the array index (including its component, if any) where ``values0``
  and ``values1`` differ the most by relative difference -- the same metric
  :func:`postgkyl.numerics.max_abs_rel_diff` ranks whole quantities by, just
  applied per-element here to locate *where* a quantity differs most, not
  just *whether* it does. At that index, estimates how fast the difference
  ``values1 - values0`` itself changes along each free axis (a 2nd-order
  finite difference -- centered in the interior, one-sided at a boundary,
  via ``np.gradient(..., edge_order=2)``), normalized by the difference's
  own value there so axes are compared on equal footing regardless of the
  quantity's units or scale. The 2 axes with the largest such normalized
  rate of change are kept; every other free axis is pinned at its own index
  from the point of maximum difference.

  A 2-D (or already ``z0``..``z5``-reduced) quantity never reaches this --
  callers only invoke it when more than 2 free axes remain.

  Args:
    values0: Reference ("before"/baseline) array, component axis last (if
      any); any number of leading spatial axes.
    values1: Array to compare against the reference; same shape as
      ``values0``.
    grid: One 1-D coordinate array per spatial axis (i.e. excluding the
      trailing component axis), in axis order. May be nodal (one more
      coordinate than samples along that axis); resolved to cell-centered
      sample points before differencing.
    free_axes: Spatial axis indices not already pinned by the caller's own
      ``z0``..``z5``; only these are candidates to keep or pin here.

  Returns:
    A dict mapping every axis in ``free_axes`` that should be pinned (all
    but the 2 kept) to its index at the point of maximum difference. Empty
    when ``len(free_axes) <= 2`` -- nothing needs pinning.
  """
  if len(free_axes) <= 2:
    return {}

  abs_diff = np.abs(values1 - values0)
  with np.errstate(divide="ignore", invalid="ignore"):
    rel_diff = abs_diff / np.abs(values0)
  # A real inf (values0 exactly 0, values1 not) is a legitimate "biggest
  # possible difference" and must be able to win the pivot; a nan (both
  # exactly 0, a true no-op) must not -- mirrors max_abs_rel_diff's own
  # inf-keeps/nan-drops policy, applied per-element to locate the pivot
  # rather than reduced to one scalar.
  pivot = np.unravel_index(
      np.argmax(np.where(np.isnan(rel_diff), -np.inf, rel_diff)),
      rel_diff.shape)

  diff = values1 - values0
  value_at_pivot = diff[pivot]
  derivatives: dict[int, float] = {}
  for axis in free_axes:
    length = diff.shape[axis]
    if length < 2:
      derivatives[axis] = 0.0  # nothing to resolve along a singleton axis
      continue
    edge_order = 2 if length >= 3 else 1
    coords = sample_coordinates(grid[axis], length)
    gradient = np.gradient(diff, coords, axis=axis, edge_order=edge_order)
    derivatives[axis] = float(gradient[pivot])

  # Normalizing by a near-zero value would make an otherwise-flat direction
  # look artificially steep, so fall back to comparing raw derivatives there.
  if abs(value_at_pivot) > np.finfo(float).eps:
    scores = {axis: abs(d / value_at_pivot) for axis, d in derivatives.items()}
  else:
    scores = {axis: abs(d) for axis, d in derivatives.items()}

  keep = set(sorted(scores, key=scores.get, reverse=True)[:2])
  return {axis: int(pivot[axis]) for axis in free_axes if axis not in keep}


def _reduced(data0, data1, zs: tuple):
  """Apply the caller's ``z0``..``z5`` selectors to both datasets, then pin
  any remaining axes beyond 2 via :func:`_dominant_axes` so both are
  plottable regardless of their original dimensionality. Return the pinned
  indices alongside the reduced datasets for the plot title."""
  num_dims = min(data0.num_dims, data1.num_dims)
  explicit = {
      f"z{d}": z
      for d, z in enumerate(zs) if z is not None and d < num_dims
  }
  free_axes = [d for d in range(num_dims) if f"z{d}" not in explicit]
  auto = _dominant_axes(data0.values, data1.values, data0.grid, free_axes)
  kwargs = {**explicit, **{f"z{d}": idx for d, idx in auto.items()}}
  pinned = {}
  for d, z in enumerate(zs[:num_dims]):
    if z is None:
      continue
    grid = data0.grid[d]
    if grid.ndim == 1:
      index = idx_parser(z, grid, data0.values.shape[d] == len(grid))
    else:
      index = idx_parser(z) if str(z).lstrip("-").isdigit() else None
    if isinstance(index, int):
      pinned[d] = index % data0.values.shape[d]
  pinned.update(auto)
  if not kwargs:
    return data0, data1, pinned
  return select(data0, **kwargs), select(data1, **kwargs), pinned


def _resolve_frame(frame: int | None, path0: str, stem0: str, path1: str,
                   stem1: str) -> int | None:
  """The caller's pinned ``frame``, or the latest one both sims wrote."""
  if frame is not None:
    return frame
  return _latest_common_frame(path0, stem0, path1, stem1)


def _diff_quantity(path0: str, path1: str, quantity: str, stem0: str,
                   stem1: str, frame: int | None) -> QuantityDiff:
  blocks0 = _load_quantity(path0, stem0, frame)
  blocks1 = _load_quantity(path1, stem1, frame)
  if len(blocks0) != len(blocks1):
    raise ValueError(
        f"'{quantity}' has {len(blocks0)} blocks under {path0!r} but "
        f"{len(blocks1)} under {path1!r}")
  max_abs = max_rel = 0.0
  for data0, data1 in zip(blocks0, blocks1):
    values0 = _point_values(data0).values
    values1 = _point_values(data1).values
    if values0.shape != values1.shape:
      raise ValueError(
          f"'{quantity}' has shape {values0.shape} under {path0!r} but "
          f"{values1.shape} under {path1!r}")
    abs_diff, rel_diff = max_abs_rel_diff(values0, values1)
    max_abs = max(max_abs, abs_diff)
    max_rel = max(max_rel, rel_diff) if rel_diff != float("inf") else max_rel
  return QuantityDiff(quantity, max_abs, max_rel)


def _save_or_collect(fig, pdf: PdfPages | None) -> None:
  """Append ``fig`` as one page of ``pdf`` and close it -- closing lets
  ``plot(..., figure=0)`` safely recreate a fresh figure 0 the next time
  it's called, confirmed by testing (a closed figure number is reused
  cleanly, with no stale axes/legend bleeding into the next quantity)."""
  if pdf is not None:
    pdf.savefig(fig)
    plt.close(fig)


def _plot_quantity(path0: str,
                   path1: str,
                   stem0: str,
                   stem1: str,
                   frame: int | None,
                   zs: tuple,
                   labels: list[str],
                   title: str,
                   out_path: str | None,
                   pdf: PdfPages | None,
                   fig_per_quantity: bool = False,
                   diff: bool = False) -> None:
  blocks0 = _load_quantity(path0, stem0, frame)
  blocks1 = _load_quantity(path1, stem1, frame)
  num_blocks = len(blocks0)

  if num_blocks > 1:
    # Each block is its own coordinate patch of the same decomposed domain
    # (or, for a dynvector, its own independent time series) -- overlaying
    # every block's baseline+candidate on one shared axis gets unreadable
    # past a handful of blocks (one legend entry per block per sim), so
    # every multiblock quantity, dynvector or spatial field alike, gets one
    # plot per block.
    for block_idx, (data0, data1) in enumerate(zip(blocks0, blocks1)):
      block_title = f"{title} (block {block_idx})"
      block_path = None
      if out_path is not None:
        block_path = f"{os.path.splitext(out_path)[0]}_b{block_idx}" \
            f"{os.path.splitext(out_path)[1]}"
      _plot_one_block(data0,
                      data1,
                      zs,
                      labels,
                      block_title,
                      block_path,
                      pdf,
                      fig_per_quantity=fig_per_quantity,
                      diff=diff)
    return

  _plot_one_block(blocks0[0],
                  blocks1[0],
                  zs,
                  labels,
                  title,
                  out_path,
                  pdf,
                  fig_per_quantity=fig_per_quantity,
                  diff=diff)


def _diff_panel_zbounds(data0, data1,
                        diff_data) -> tuple[dict[int, float], dict[int, float]]:
  """``zmin``/``zmax`` dicts for the combined ``plot(data0, data1, diff_data,
  ...)`` call: panels ``0..num_comps-1`` (baseline) and
  ``num_comps..2*num_comps-1`` (candidate) share one color scale per
  component; panels ``2*num_comps..3*num_comps-1`` (the diff) get their own
  independent scale instead -- a difference is usually much smaller in
  magnitude than the original quantity, so sharing the scale would make it
  look flat. Returned as complete dicts covering every panel, since
  ``plot()``'s own auto shared-scale fallback is keyed only by component (not
  by panel) and would otherwise let the diff's range leak into the
  baseline/candidate scale. A degenerate (all-NaN or constant) component is
  simply absent from the dict, deferring to Matplotlib's own default."""
  num_comps = data0.num_comps
  zmin: dict[int, float] = {}
  zmax: dict[int, float] = {}
  for comp in range(num_comps):
    shared = _finite_bounds(data0.values[..., comp], data1.values[..., comp])
    if shared is not None:
      for panel in (comp, num_comps + comp):
        zmin[panel], zmax[panel] = shared
    diff_vals = _finite_bounds(diff_data.values[..., comp])
    if diff_vals is not None:
      panel = 2 * num_comps + comp
      zmin[panel], zmax[panel] = diff_vals
  return zmin, zmax


def _finite_bounds(*arrays: np.ndarray) -> tuple[float, float] | None:
  """The range of finite values, absent when no color scale can be set."""
  values = np.concatenate([array.ravel() for array in arrays])
  values = values[np.isfinite(values)]
  if not values.size or values.min() == values.max():
    return None
  return float(values.min()), float(values.max())


def _page_layout(data, diff: bool) -> tuple[bool, int]:
  """Whether the reduced data plots as lines, and components per page."""
  is_line = sum(length > 1 for length in data.values.shape[:-1]) <= 1
  panels_per_comp = (1 if is_line else 2) + int(diff)
  return is_line, _MAX_PANELS_PER_FIGURE // panels_per_comp


def _plot_one_block(data0,
                    data1,
                    zs: tuple,
                    labels: list[str],
                    title: str,
                    out_path: str | None,
                    pdf: PdfPages | None,
                    fig_per_quantity: bool = False,
                    diff: bool = False) -> None:
  data0, data1, pinned = _reduced(_point_values(data0), _point_values(data1),
                                  zs)
  is_line, comps_per_page = _page_layout(data0, diff)
  num_pages = (1 if fig_per_quantity else
               (data0.num_comps + comps_per_page - 1) // comps_per_page)
  for page in range(num_pages):
    if not fig_per_quantity:
      plt.figure(0).set_size_inches(*_COMPARE_FIGSIZE)
    start = page * comps_per_page
    stop = min(start + comps_per_page, data0.num_comps)
    page_title = title
    if num_pages > 1:
      comp_range = (f"c{start}" if stop == start +
                    1 else f"c{start}-c{stop - 1}")
      page_title += f" (page {page + 1}/{num_pages}, {comp_range})"
    if pinned:
      page_title += " (" + ", ".join(
          f"z{d}={index}" for d, index in sorted(pinned.items())) + ")"
    page_title = textwrap.fill(page_title, width=60, break_on_hyphens=False)
    page_path = out_path
    if out_path is not None and num_pages > 1:
      stem, extension = os.path.splitext(out_path)
      page_path = f"{stem}_p{page + 1}{extension}"
    if num_pages == 1:
      _plot_unlimited_block(data0,
                            data1,
                            labels,
                            page_title,
                            page_path,
                            pdf,
                            diff=diff,
                            is_line=is_line)
    else:
      _plot_split_page(data0,
                       data1,
                       range(start, stop),
                       labels,
                       page_title,
                       page_path,
                       pdf,
                       diff=diff,
                       is_line=is_line)
      if page < num_pages - 1 and pdf is None:
        plt.close(0)


def _plot_unlimited_block(data0, data1, labels: list[str], title: str,
                          out_path: str | None, pdf: PdfPages | None, *,
                          diff: bool, is_line: bool) -> None:
  # Not data0 - data1: that operator enforces a strict (rtol=1e-9) grid-
  # coordinate match, but two independently-run sims' grids commonly agree
  # only to ~1e-6 relative (platform/compiler floating-point noise) even
  # when every other part of this module already treats them as the same
  # quantity (ranking only ever compares .values shape, never grid
  # closeness -- see _diff_quantity). Building the result directly from
  # .values, the same way every other diagnostic in this codebase computes
  # a derived dataset, skips that check; the shapes are already guaranteed
  # equal by _diff_quantity before either dataset reaches this function.
  diff_data = (data0._result(data0.grid, data0.values -
                             data1.values) if diff else None)
  num_comps = data0.num_comps
  # plot()'s explicit legend_labels are used verbatim on every component
  # subplot, with no added _cN suffix (a deliberate, documented contract
  # relied on elsewhere) -- so a multi-component quantity would otherwise
  # show the identical pair of labels on every one of its component panels,
  # with no way to tell which panel is which component. Its *default*
  # label path (triggered by omitting legend_labels) does add that suffix,
  # built from each dataset's own .label -- so for >1 component, set the
  # label directly and let plot() derive "<label>_c<i>" on its own; for a
  # single component there is nothing to disambiguate, so pass legend_labels
  # explicitly instead, keeping today's plain, unsuffixed labels.
  if num_comps > 1:
    data0.label, data1.label = labels
    legend_kwargs = {}
  else:
    legend_kwargs = {"legend_labels": labels}
  if diff_data is not None:
    if num_comps > 1:
      # plot()'s default-label path only ever appends "_c<i>" to the whole
      # dataset's own label, giving every one of the diff's components the
      # identical "<label0> - <label1>_c<i>" (the labels joined, THEN
      # suffixed) instead of joining each component's own suffixed label
      # ("<label0>_c<i> - <label1>_c<i>"). Splitting the diff into one
      # single-component dataset per component via select(), each with its
      # own explicit legend_labels entry, sidesteps that path entirely.
      diff_pieces = [select(diff_data, comp=c) for c in range(num_comps)]
      diff_piece_labels = [
          f"{labels[0]}_c{c} - {labels[1]}_c{c}" for c in range(num_comps)
      ]
    else:
      diff_pieces = [diff_data]
      diff_piece_labels = [f"{labels[0]} - {labels[1]}"]
  # Both datasets carry no multiblock identity of their own (a single sim
  # directory's quantity, not one block of a decomposed one), so without a
  # shared target figure `plot` would treat them as two singleton families
  # and draw each on its own figure instead of overlaying/pairing them;
  # figure=0 forces one shared figure, clear=True keeps it from
  # accumulating axes left over from an earlier quantity plotted in the
  # same process (e.g. an earlier call from the same `sims()` run).
  if is_line:
    # A single combined call with all 3 datasets would NOT keep baseline and
    # candidate overlaid -- each dataset would get its own axis block, since
    # overlay_axes is all-or-nothing for the whole call. Reserving
    # num_axes=2*num_comps with overlay_axes=True instead keeps baseline and
    # candidate sharing block 0 while leaving block 1 free for a second,
    # separate call to append the diff dataset right after -- 1-D line
    # panels carry no colorbar, so (unlike the 2-D case below) a second call
    # appending to the same figure is safe.
    extra_axes_kwargs = ({
        "num_axes": 2 * num_comps,
        "overlay_axes": True
    } if diff_data is not None else {})
    fig = plot(data0,
               data1,
               figure=0,
               clear=True,
               title=title,
               no_show=True,
               saveas=None,
               **extra_axes_kwargs,
               **legend_kwargs)
    if diff_data is not None:
      fig = plot(*diff_pieces,
                 figure=0,
                 start_axes=num_comps,
                 num_axes=num_comps,
                 no_show=True,
                 saveas=None,
                 legend_labels=diff_piece_labels)
  else:
    if diff_data is None:
      fig = plot(data0,
                 data1,
                 num_axes=2 * num_comps,
                 figure=0,
                 clear=True,
                 title=title,
                 no_show=True,
                 saveas=None,
                 **legend_kwargs)
    else:
      # Unlike 1-D, a 2-D (pcolormesh) panel brings along its own colorbar
      # axis, interleaved into the figure's axes list in a way a second,
      # separate plot() call's start_axes arithmetic can't account for -- so
      # this must be one single combined call with all 3 datasets instead.
      # Each dataset is likewise split into its single-component pieces (see
      # above), so every one of the 3*num_comps panels gets its own
      # explicit, correctly-suffixed legend_labels entry.
      base_pieces = [select(data0, comp=c) for c in range(num_comps)]
      cand_pieces = [select(data1, comp=c) for c in range(num_comps)]
      all_pieces = base_pieces + cand_pieces + diff_pieces
      all_labels = ([f"{labels[0]}_c{c}" for c in range(num_comps)] +
                    [f"{labels[1]}_c{c}" for c in range(num_comps)] +
                    diff_piece_labels) if num_comps > 1 else (list(labels) +
                                                              diff_piece_labels)
      zmin, zmax = _diff_panel_zbounds(data0, data1, diff_data)
      fig = plot(*all_pieces,
                 num_axes=3 * num_comps,
                 num_subplot_row=num_comps,
                 zmin=zmin,
                 zmax=zmax,
                 figure=0,
                 clear=True,
                 title=title,
                 no_show=True,
                 saveas=None,
                 legend_labels=all_labels)
  _finish_figure(fig, out_path, pdf)


def _plot_split_page(data0, data1, components: range, labels: list[str],
                     title: str, out_path: str | None, pdf: PdfPages | None, *,
                     diff: bool, is_line: bool) -> None:
  """Draw complete original components on one page with at most six panels."""
  difference = (data0._result(data0.grid, data0.values -
                              data1.values) if diff else None)
  num_components = len(components)
  num_panels = num_components * ((1 if is_line else 2) + int(diff))

  def label(which: int, comp: int) -> str:
    return f"{labels[which]}_c{comp}"

  if is_line:
    # Line panels have no colorbar axes, so each component can be appended
    # to its reserved axis. Baseline and candidate share the top-row axis;
    # the difference for that component occupies the axis beneath it.
    fig = None
    for local, comp in enumerate(components):
      fig = plot(select(data0, comp=comp),
                 select(data1, comp=comp),
                 num_axes=num_panels,
                 start_axes=local,
                 overlay_axes=True,
                 figure=0,
                 clear=fig is None,
                 title=title if fig is None else None,
                 no_show=True,
                 legend_labels=[label(0, comp), label(1, comp)])
    if difference is not None:
      for local, comp in enumerate(components):
        fig = plot(select(difference, comp=comp),
                   num_axes=num_panels,
                   start_axes=num_components + local,
                   figure=0,
                   no_show=True,
                   legend_labels=[f"{label(0, comp)} - {label(1, comp)}"])
  else:
    # A 2-D plot appends a colorbar axis per panel, so all panels must be
    # supplied in one plot() call; a later start_axes would count colorbars.
    panels = []
    panel_labels = []
    zmin: dict[int, float] = {}
    zmax: dict[int, float] = {}
    shared_bounds = {
        comp: _finite_bounds(data0.values[..., comp], data1.values[..., comp])
        for comp in components
    }
    if difference is None:
      # Baselines occupy the top row and candidates the bottom row when
      # three two-panel components fill a page.
      panel_specs = ([(data0, comp, label(0, comp), shared_bounds[comp])
                      for comp in components] +
                     [(data1, comp, label(1, comp), shared_bounds[comp])
                      for comp in components])
    else:
      # Two three-panel components fill two rows, one component per row.
      panel_specs = []
      for comp in components:
        panel_specs.extend([(data0, comp, label(0, comp), shared_bounds[comp]),
                            (data1, comp, label(1, comp), shared_bounds[comp]),
                            (difference, comp,
                             f"{label(0, comp)} - {label(1, comp)}",
                             _finite_bounds(difference.values[..., comp]))])
    for data, comp, panel_label, scale in panel_specs:
      panel = len(panels)
      panels.append(select(data, comp=comp))
      panel_labels.append(panel_label)
      if scale is not None:
        zmin[panel], zmax[panel] = scale
    fig = plot(*panels,
               num_axes=num_panels,
               num_subplot_col=3 if num_panels in (3, 6) else None,
               zmin=zmin,
               zmax=zmax,
               figure=0,
               clear=True,
               title=title,
               no_show=True,
               legend_labels=panel_labels)
  _finish_figure(fig, out_path, pdf)


def _finish_figure(fig, out_path: str | None, pdf: PdfPages | None) -> None:
  if fig._suptitle is not None:
    fig._suptitle.set_fontsize(10)
  if out_path is not None:
    fig.savefig(out_path, dpi=200)
  _save_or_collect(fig, pdf)


def _prepare_one(
    path0: str,
    path1: str,
    *,
    out_dir: str,
    num_plots: int,
    must_include: list[str] | None,
    exclude: list[str] | None,
    frame: int | None,
) -> _ComparisonPlan:
  """Rank shared quantities once, before counting or rendering PDF pages."""
  exclude_prefixes = _DEFAULT_EXCLUDE_PREFIXES + tuple(exclude or ())
  must_include = list(must_include or [])

  quantities0 = _stems_by_quantity(path0)
  quantities1 = _stems_by_quantity(path1)
  shared = sorted(set(quantities0) & set(quantities1))
  shared = [q for q in shared if not _is_excluded(q, exclude_prefixes)]
  only_in_0 = sorted(set(quantities0) - set(quantities1))
  only_in_1 = sorted(set(quantities1) - set(quantities0))

  missing_must_include = [
      quantity for quantity in must_include
      if quantity not in quantities0 or quantity not in quantities1
  ]
  if missing_must_include:
    raise ValueError("must_include quantity(s) not present in both sims: "
                     f"{', '.join(sorted(missing_must_include))}")

  compared: list[QuantityDiff] = []
  skipped: dict[str, str] = {}
  for quantity in shared:
    stem0, stem1 = quantities0[quantity], quantities1[quantity]
    try:
      common_frame = _resolve_frame(frame, path0, stem0, path1, stem1)
      compared.append(
          _diff_quantity(path0, path1, quantity, stem0, stem1, common_frame))
    except Exception as exc:  # noqa: BLE001 -- see docstring: an explicit
      # must_include still raises (below); an auto-ranked quantity merely
      # doesn't participate in ranking. Not every ".gkyl" stem is a field or
      # dynvector `load()` can read (e.g. a multiblock `block_topo` file),
      # and a single such output has nothing to do with the health of the
      # rest of the comparison.
      if quantity in must_include:
        raise
      skipped[quantity] = str(exc)
  compared.sort(key=lambda d: d.max_rel_diff, reverse=True)

  # Every must_include quantity, then the highest-ranked rest, until
  # num_plots is filled -- every ranked quantity is plottable now (any N-D
  # quantity reduces to 2 axes via _dominant_axes), so nothing here needs
  # to skip or raise for dimensionality.
  selected = list(dict.fromkeys(must_include))
  for quantity_diff in compared:
    if len(selected) >= num_plots:
      break
    if quantity_diff.quantity not in selected:
      selected.append(quantity_diff.quantity)

  frames = {
      quantity:
      _resolve_frame(frame, path0, quantities0[quantity], path1,
                     quantities1[quantity])
      for quantity in selected
  }
  report = CompareSimsReport(
      compared=tuple(compared),
      plotted=tuple(selected),
      only_in_0=tuple(only_in_0),
      only_in_1=tuple(only_in_1),
      skipped=skipped,
      out_dir=out_dir,
  )
  return _ComparisonPlan(quantities0, quantities1, frames, report)


def _page_count_for_plan(path0: str, path1: str, plan: _ComparisonPlan,
                         zs: tuple, fig_per_quantity: bool, diff: bool) -> int:
  """Count the pages a prepared comparison will render, including blocks."""
  count = 0
  for quantity in plan.report.plotted:
    blocks0 = _load_quantity(path0, plan.quantities0[quantity],
                             plan.frames[quantity])
    blocks1 = _load_quantity(path1, plan.quantities1[quantity],
                             plan.frames[quantity])
    for data0, data1 in zip(blocks0, blocks1):
      if fig_per_quantity:
        count += 1
      else:
        reduced, _, _ = _reduced(_point_values(data0), _point_values(data1), zs)
        _, comps_per_page = _page_layout(reduced, diff)
        count += (reduced.num_comps + comps_per_page - 1) // comps_per_page
  return count


def _compare_one(
    path0: str,
    path1: str,
    *,
    out_dir: str,
    num_plots: int,
    must_include: list[str] | None,
    exclude: list[str] | None,
    labels: list[str] | None,
    z0: str | None,
    z1: str | None,
    z2: str | None,
    z3: str | None,
    z4: str | None,
    z5: str | None,
    format: str,
    frame: int | None,
    pdf: PdfPages | None = None,
    title_prefix: str = "",
    fig_per_quantity: bool = False,
    diff: bool = False,
    plan: _ComparisonPlan | None = None,
) -> CompareSimsReport:
  """Render one prepared simulation comparison."""
  if plan is None:
    plan = _prepare_one(path0,
                        path1,
                        out_dir=out_dir,
                        num_plots=num_plots,
                        must_include=must_include,
                        exclude=exclude,
                        frame=frame)
  if pdf is None:
    os.makedirs(out_dir, exist_ok=True)
  resolved_labels = labels or _default_labels(path0, path1)
  zs = (z0, z1, z2, z3, z4, z5)
  for quantity in plan.report.plotted:
    out_path = None if pdf is not None else os.path.join(
        out_dir, f"{quantity}.{format}")
    _plot_quantity(path0,
                   path1,
                   plan.quantities0[quantity],
                   plan.quantities1[quantity],
                   plan.frames[quantity],
                   zs,
                   resolved_labels,
                   f"{title_prefix}{quantity}",
                   out_path,
                   pdf,
                   fig_per_quantity=fig_per_quantity,
                   diff=diff)
  return plan.report


def _shared_subdirs(path0: str, path1: str) -> list[str]:
  """Simulation subdirectory names ``path0``/``path1`` share, sorted."""
  names0 = {entry.name for entry in os.scandir(path0) if entry.is_dir()}
  names1 = {entry.name for entry in os.scandir(path1) if entry.is_dir()}
  common = sorted(names0 & names1)
  if not common:
    raise ValueError(
        f"no simulation subdirectory is shared between {path0!r} and "
        f"{path1!r}")
  return common


def _batch(
    path0: str,
    path1: str,
    *,
    out_dir: str,
    num_plots: int,
    must_include: list[str] | None,
    exclude: list[str] | None,
    labels: list[str] | None,
    z0: str | None,
    z1: str | None,
    z2: str | None,
    z3: str | None,
    z4: str | None,
    z5: str | None,
    format: str,
    frame: int | None,
    pdf: PdfPages | None = None,
    fig_per_quantity: bool = False,
    diff: bool = False,
    plans: dict[str, _ComparisonPlan] | None = None,
) -> dict[str, CompareSimsReport]:
  """``_compare_one`` on every simulation ``path0``/``path1`` share. ``pdf``,
  when given, is shared by every simulation's pages instead of each sim
  writing its own ``<out_dir>/<name>/`` subfolder of individual files.
  ``labels`` (already resolved, see :func:`_parse_label_pair`), when given,
  replaces the basenames of ``path0``/``path1`` as the shared pair of legend
  labels every compared simulation uses."""
  common = _shared_subdirs(path0, path1)
  batch_labels = labels or [
      os.path.basename(os.path.normpath(path0)),
      os.path.basename(os.path.normpath(path1))
  ]
  return {
      name:
      _compare_one(
          os.path.join(path0, name),
          os.path.join(path1, name),
          out_dir=out_dir if pdf is not None else os.path.join(out_dir, name),
          num_plots=num_plots,
          must_include=must_include,
          exclude=exclude,
          labels=batch_labels,
          z0=z0,
          z1=z1,
          z2=z2,
          z3=z3,
          z4=z4,
          z5=z5,
          format=format,
          frame=frame,
          pdf=pdf,
          title_prefix=f"{name}: ",
          fig_per_quantity=fig_per_quantity,
          diff=diff,
          plan=plans[name] if plans is not None else None,
      )
      for name in common
  }


def _title_page_figure(path0: str,
                       path1: str,
                       *,
                       batch: bool,
                       sim_names: list[str] | None = None,
                       page_ranges: dict[str, tuple[int, int]] | None = None):
  """A plain text-only title page naming what a combined-PDF comparison
  covers, so the PDF is self-contained without its filename or the command
  that produced it."""
  fig = plt.figure(figsize=(8.5, 11))
  lines = ["Gkeyll simulation comparison", ""]

  def page_label(name: str) -> str:
    if page_ranges is None or name not in page_ranges:
      return ""
    first, last = page_ranges[name]
    if first > last:
      return " — no plot pages"
    if first == last:
      return f" — page {first}"
    return f" — pages {first}-{last}"

  if batch:
    lines += [
        f"Baseline directory: {path0}", f"Candidate directory: {path1}", "",
        f"Simulations compared ({len(sim_names or [])}):"
    ]
    lines += [f"  - {name}{page_label(name)}" for name in sim_names or []]
  else:
    name = os.path.basename(os.path.normpath(path0))
    lines += [
        f"Baseline: {path0}", f"Candidate: {path1}", "",
        f"Comparison: {name}{page_label(name)}"
    ]
  fig.text(0.1,
           0.9,
           "\n".join(lines),
           ha="left",
           va="top",
           fontsize=12,
           family="monospace",
           wrap=True)
  return fig


def sims(
    path0: str,
    path1: str,
    *,
    batch: bool = False,
    out_dir: str | None = None,
    file_per_fig: bool = False,
    fig_per_quantity: bool = False,
    pdf_name: str | None = None,
    num_plots: int = 5,
    must_include: list[str] | None = None,
    exclude: list[str] | None = None,
    labels: str | None = None,
    z0: str | None = None,
    z1: str | None = None,
    z2: str | None = None,
    z3: str | None = None,
    z4: str | None = None,
    z5: str | None = None,
    format: str = "jpg",
    frame: int | None = None,
    diff: bool = False,
    verbose: bool = False,
) -> dict[str, CompareSimsReport]:
  """Compare two Gkeyll simulations' output and plot their biggest differences.

  Finds every output quantity both sims wrote, ranks the shared ones by
  :func:`postgkyl.numerics.max_abs_rel_diff` (largest relative difference
  first), and plots ``num_plots`` of them -- always including every
  ``must_include`` quantity. Use ``exclude`` to skip specific quantities or
  prefixes.

  A 2-D field (or a 3-D+ field once ``z0``..``z5`` have reduced it to 2-D)
  is drawn with the two sims side by side on a shared color scale. Any axis
  left unpinned by ``z0``..``z5`` beyond the 2 ``plot`` can draw is pinned
  automatically: at the point where the two sims differ most (by relative
  difference), the axes along which that difference itself changes fastest
  (a normalized, 2nd-order finite-difference estimate) are kept, and every
  other unpinned axis is fixed at its own index there -- so a 3-D, 4-D, 5-D,
  or 6-D quantity is always plottable with no selectors at all, picking
  whichever cross-section is most informative for that specific pair of
  sims rather than an arbitrary fixed coordinate.

  ``diff`` additionally plots ``baseline - candidate``: one more
  overlay-adjacent panel per component for a dynvector/1-D field, or one
  more side-by-side panel per component (with its own independent color
  scale, since a difference is usually much smaller than the original
  quantity) for a 2-D field. Labeled ``"<label0> - <label1>"`` for a
  single-component quantity, or ``"<label0>_c<i> - <label1>_c<i>"`` per
  component for a multi-component one.

  Each figure contains at most six panels by default. Complete component
  groups continue on numbered pages when necessary: a 1-D component needs
  one panel (two with ``diff``), and a 2-D component needs two (three with
  ``diff``). ``fig_per_quantity`` restores the unlimited, single-figure
  layout for each quantity or multiblock patch.

  By default, every plotted quantity is collected into one combined,
  PDF, whose path is ``<out_dir>/<pdf_name>`` if ``out_dir`` was given, else
  ``<cwd>/<pdf_name>``. Its title page lists the PDF page range for each
  simulation. Alternatively, passing ``file_per_fig`` yields individual
  image files.

  ``batch`` reinterprets ``path0``/``path1`` as directories with multiple
  simulations each (rather than having one simulation each): each holds one
  subdirectory per simulation. Subdirectories or simulations present in both
  are compared.

  Args:
    path0: the first simulation's output directory (w/o --batch) or a
      directory with one subdirectory per simulation (w/ --batch) to compare
      against --path1.
    path1: Path to data from the second simulation(s).
    batch: Compare every simulation two directories-of-sims share.
    pdf_name: Filename for the combined PDF; defaults to
      ``"postgkyl_compare_sims.pdf"``.
    out_dir: The path to the directory where the combined PDF is written to
      (defaults to ``<cwd>/``). With --file_per_fig this is the directory
      individual figures are written to (defaults to
      ``<cwd>/postgkyl_compare_sims``).
    file_per_fig: Write individual image files instead of one combined PDF.
      A quantity that spans figures gets numbered ``_p1``, ``_p2``, ... files.
    fig_per_quantity: Keep every component of a quantity on one figure,
      without the default six-panel limit.
    num_plots: Number of quantities to plot, including every
      ``must_include`` entry.
    must_include: Names of quantities (e.g. ``"field_energy"``) that must
      beplotted.
    exclude: Quantity-name prefixes to skip (e.g. ``"geo_"``, ``"bc_"``).
    labels: The two legend labels for ``path0``/``path1`` as one
      comma-separated string, e.g. ``"baseline,candidate"``; defaults to
      each path's shortest distinguishing path segment
    z0: Selector for coordinate direction 0.
    z1: Selector for coordinate direction 1.
    z2: Selector for coordinate direction 2.
    z3: Selector for coordinate direction 3.
    z4: Selector for coordinate direction 4.
    z5: Selector for coordinate direction 5.
    format: Output image file extension (e.g. ``"jpg"``, ``"png"``,
      ``"pdf"``, ``"svg"``) if ``file_per_fig`` is set.
    frame: Frame to compare for framed (per-timestep) quantities; defaults
      to the latest frame both sims wrote. Quantities with no frame index
      (dynvectors, single-snapshot fields) ignore this.
    diff: Also plot ``baseline - candidate``, appended after the
      baseline/candidate panel(s).
    verbose: Print the full per-quantity report at the CLI (the ``repr`` of
      the returned value). Default: CLI prints a one-line-per-sim summary.

  Returns:
    A dict mapping each compared simulation's name to its
    :class:`CompareSimsReport`. Without ``batch``, this has exactly one
    entry, keyed by ``path0``'s basename. With ``batch``, one entry per
    simulation subdirectory shared by ``path0`` and ``path1``.

  Raises:
    ValueError: a ``must_include`` quantity is missing from either sim, two
      compared files are structurally incompatible (block count or array
      shape mismatch), ``labels`` is not exactly two comma-separated,
      non-empty labels, ``pdf_name`` was given together with
      ``file_per_fig``, or (``batch`` only) ``path0`` and ``path1`` share no
      simulation subdirectory.
  """
  if pdf_name is not None and file_per_fig:
    raise ValueError("pdf_name cannot be combined with file_per_fig")
  resolved_labels = _parse_label_pair(labels)

  def _run(
      effective_out_dir: str,
      pdf: PdfPages | None,
      plans: dict[str, _ComparisonPlan] | None = None
  ) -> dict[str, CompareSimsReport]:
    if batch:
      return _batch(path0,
                    path1,
                    out_dir=effective_out_dir,
                    num_plots=num_plots,
                    must_include=must_include,
                    exclude=exclude,
                    labels=resolved_labels,
                    z0=z0,
                    z1=z1,
                    z2=z2,
                    z3=z3,
                    z4=z4,
                    z5=z5,
                    format=format,
                    frame=frame,
                    pdf=pdf,
                    fig_per_quantity=fig_per_quantity,
                    diff=diff,
                    plans=plans)
    report = _compare_one(path0,
                          path1,
                          out_dir=effective_out_dir,
                          num_plots=num_plots,
                          must_include=must_include,
                          exclude=exclude,
                          labels=resolved_labels,
                          z0=z0,
                          z1=z1,
                          z2=z2,
                          z3=z3,
                          z4=z4,
                          z5=z5,
                          format=format,
                          frame=frame,
                          pdf=pdf,
                          fig_per_quantity=fig_per_quantity,
                          diff=diff,
                          plan=plans[os.path.basename(os.path.normpath(path0))]
                          if plans is not None else None)
    return {os.path.basename(os.path.normpath(path0)): report}

  if not file_per_fig:
    pdf_dir = out_dir if out_dir is not None else os.getcwd()
    pdf_path = os.path.join(pdf_dir, pdf_name or _DEFAULT_PDF_NAME)
    sim_names = (_shared_subdirs(path0, path1)
                 if batch else [os.path.basename(os.path.normpath(path0))])
    plans: dict[str, _ComparisonPlan] = {}
    page_ranges: dict[str, tuple[int, int]] = {}
    next_page = 2  # The title page occupies page 1.
    zs = (z0, z1, z2, z3, z4, z5)
    for name in sim_names:
      sim_path0 = os.path.join(path0, name) if batch else path0
      sim_path1 = os.path.join(path1, name) if batch else path1
      plans[name] = _prepare_one(sim_path0,
                                 sim_path1,
                                 out_dir=pdf_dir,
                                 num_plots=num_plots,
                                 must_include=must_include,
                                 exclude=exclude,
                                 frame=frame)
      num_pages = _page_count_for_plan(sim_path0, sim_path1, plans[name], zs,
                                       fig_per_quantity, diff)
      page_ranges[name] = (next_page, next_page + num_pages - 1)
      next_page += num_pages
    os.makedirs(pdf_dir, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
      title = _title_page_figure(path0,
                                 path1,
                                 batch=batch,
                                 sim_names=sim_names if batch else None,
                                 page_ranges=page_ranges)
      pdf.savefig(title)
      plt.close(title)
      return _ReportDict(_run(pdf_dir, pdf, plans), verbose=verbose)

  effective_out_dir = (out_dir if out_dir is not None else os.path.join(
      os.getcwd(), _DEFAULT_SUBDIR))
  return _ReportDict(_run(effective_out_dir, None), verbose=verbose)


__all__ = ["CompareSimsReport", "QuantityDiff", "sims"]
