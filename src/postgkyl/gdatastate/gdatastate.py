"""``GDataState`` -- the verb-less data container (the CONTAINER layer).

Holds a Gkeyll dataset: a nodal ``grid`` (list of 1-D edge arrays) plus values
in one of **two backends** -- the two-domain lifecycle of REFACTOR_GKEYLL_FFI.md:

- ``backend == "gkyl"``: modal DG coefficients held as a native
  :class:`~postgkyl.gpython.array.GkylArray`. Gkeyll owns the memory and all math
  on it (weak ops, coefficient lin-combs, integrate). ``values`` exposes a
  read-only NumPy *view* for inspection; ``__array__`` refuses (interpolate first).
- ``backend == "numpy"``: post-``interpolate`` (or never-modal) values as a plain
  ``np.ndarray`` -- the field domain, where all NumPy math applies.

It constructs itself by delegating to the :mod:`postgkyl.io` leaf and exposes
only *state*. Crucially it imports **nothing upward** (no ``operations``/``render``/
``api``). The fluent verb methods and the computing operators live on the
:class:`postgkyl.gdata.gdata.GData` subclass, one layer up. That is what keeps
the dependency graph a strict, cycle-free DAG -- see HIERARCHY_2.md / HIERARCHY_3.md.
"""

from __future__ import annotations

import numbers
import warnings
from typing import Tuple

import numpy as np

from postgkyl import io  # leaf layer (below); top-level import -- never a cycle
from postgkyl import gpython  # foreign floor (below): GkylArray backend type


class GDataState:
  """Storage + metadata for one dataset. No verbs; no upward imports."""

  def __init__(self,
               file_name: str = "",
               *,
               ctx: dict | None = None,
               tag: str = "default",
               label: str = "",
               value_form: str | None = None,
               basis_type: str | None = None,
               poly_order: int | None = None,
               **read_kwargs):
    self._grid: list | None = None
    self._values: np.ndarray | gpython.GkylArray | None = None
    self.ctx: dict = {}
    if ctx:
      self.ctx.update(ctx)
    self._tag = tag
    self._label = ""
    self._custom_label = label
    self._file_name = str(file_name)
    self.color = None

    if self._file_name:
      self._grid, self._values = io.read(self._file_name,
                                         self.ctx,
                                         value_form=value_form,
                                         basis_type=basis_type,
                                         poly_order=poly_order,
                                         **read_kwargs)
      self._stamp_output_name()
      # A dynvector/diagnostic file (no "cells" in ctx: no reader ever stamps
      # one without a spatial grid, e.g. a dynvector time series) has no DG
      # basis to speak of -- basis_type/poly_order/value_form genuinely don't
      # apply, so no defaulting is needed here.
      if self.ctx.get("cells") is not None:
        defaulted = []
        if self.ctx.get("basis_type") is None:
          self.ctx["basis_type"] = "serendipity"
          defaulted.append("basis_type")
        if self.ctx.get("poly_order") is None:
          self.ctx["poly_order"] = 0
          defaulted.append("poly_order")
        if "value_form" not in self.ctx:
          self.ctx["value_form"] = "nodal"
          defaulted.append("value_form")
        if defaulted:
          warnings.warn(
              f"{self._file_name}:\n"
              f"{', '.join(defaulted)} not resolvable (not present in the "
              "file header, and not given explicitly); defaulting to "
              "basis_type='serendipity', poly_order=0, value_form='nodal' "
              "(p0 -- one point per cell, at the cell center). Pass "
              "basis_type=/poly_order=/value_form=... explicitly if this "
              "is wrong.",
              stacklevel=2)

  # -------------------------------------------------------------- identity
  def _stamp_output_name(self) -> None:
    """Record the file's Gkeyll *identity* (sim, block, quantity, frame) in
    ``ctx``, parsed once from its path by :mod:`postgkyl.io.naming`.

    Header metadata wins: ``setdefault`` never overwrites a ``frame`` (or
    anything else) a reader already read out of the file itself. Because
    ``clone`` copies ``ctx``, the identity survives every verb, so a
    multiblock family is still recognizable after ``interpolate``/``gk_rz``
    -- which is what lets terminal verbs draw one field's blocks together
    (see ``gdatastate.collection.group_blocks``).
    """
    name = io.parse_output_name(self._file_name)
    if name is None:
      return
    self.ctx.setdefault("sim", name.sim)
    self.ctx.setdefault("block", name.block)
    self.ctx.setdefault("quantity", name.quantity)
    if name.frame is not None:
      self.ctx.setdefault("frame", name.frame)

  @property
  def output_name(self):
    """This dataset's parsed source-file identity (:class:`postgkyl.io.OutputName`),
    or ``None`` when it was never read from disk."""
    return io.parse_output_name(self._file_name)

  # ------------------------------------------------------------------ tags
  def get_tag(self) -> str:
    """Return the short identifier used to select this dataset."""
    return self._tag

  def set_tag(self, tag: str = "") -> None:
    """Replace the dataset tag when ``tag`` is nonempty."""
    if tag:
      self._tag = tag

  tag = property(get_tag, set_tag)

  def get_label(self) -> str:
    """Return the custom label, falling back to the generated label."""
    return self._custom_label or self._label

  def set_label(self, label: str) -> None:
    """Set the generated display label."""
    self._label = label

  label = property(get_label, set_label)

  @property
  def file_name(self) -> str:
    """Source file path this dataset was loaded from ("" if it was never
    read from disk, e.g. a verb's freshly-computed result)."""
    return self._file_name

  # ------------------------------------------------------------- shape info
  def get_num_cells(self) -> np.ndarray:
    """Return the cell count in each spatial dimension."""
    if self.ctx.get("cells") is not None:
      return np.asarray(self.ctx["cells"])
    if isinstance(self._values, np.ndarray):
      return np.array(self._values.shape[:-1], dtype=np.int64)
    return np.array([], dtype=np.int64)

  num_cells = property(get_num_cells)

  def get_num_comps(self) -> int:
    """Return the number of physical components per cell."""
    if self.ctx.get("num_comps"):
      return int(self.ctx["num_comps"])
    if isinstance(self._values, gpython.GkylArray):
      return self._values.ncomp
    if self._values is not None:
      return int(self._values.shape[-1])
    return 0

  num_comps = property(get_num_comps)

  def get_num_dims(self) -> int:
    """Return the number of spatial dimensions."""
    if self.ctx.get("cells") is not None:
      return len(self.ctx["cells"])
    if isinstance(self._values, np.ndarray):
      return int(self._values.ndim - 1)
    return 0

  num_dims = property(get_num_dims)

  def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays containing the lower and upper spatial bounds."""
    if "lower" in self.ctx and "upper" in self.ctx:
      return np.asarray(self.ctx["lower"]), np.asarray(self.ctx["upper"])
    if self._grid is not None:
      num_dims = self.get_num_dims()
      lo = np.array([self._grid[d].min() for d in range(num_dims)])
      up = np.array([self._grid[d].max() for d in range(num_dims)])
      return lo, up
    return None, None

  bounds = property(get_bounds)

  def get_grid_type(self) -> str:
    """Return the grid classification, defaulting to ``"uniform"``."""
    return self.ctx.get("grid_type", "uniform")

  # --------------------------------------------------------- grid / values
  def get_grid(self) -> list:
    """Return the coordinate array for each spatial dimension."""
    return self._grid

  def set_grid(self, grid: list) -> None:
    """Replace the coordinate arrays and update their stored bounds."""
    self._grid = grid
    # ``len(grid)`` (not ``get_num_dims()``) on purpose: for a gkyl-backed
    # dataset, num_dims reads ctx["cells"], which a dimension-reducing verb
    # (e.g. ``average``) updates via ``_result``'s ctx_updates -- AFTER
    # ``push`` calls this method. Deriving straight from the just-given grid
    # avoids depending on that update having landed yet.
    num_dims = len(grid)
    self.ctx["lower"] = np.array([grid[d].min() for d in range(num_dims)])
    self.ctx["upper"] = np.array([grid[d].max() for d in range(num_dims)])

  grid = property(get_grid, set_grid)

  @property
  def backend(self) -> str:
    """``"gkyl"`` (native modal storage) or ``"numpy"`` (field domain)."""
    return "gkyl" if isinstance(self._values, gpython.GkylArray) else "numpy"

  @property
  def native(self) -> gpython.GkylArray | None:
    """The native ``GkylArray`` when gkyl-backed; None otherwise. This is the
    handle the modal verbs pass to the Gkeyll kernels."""
    return self._values if isinstance(self._values, gpython.GkylArray) else None

  def get_values(self) -> np.ndarray:
    """Values for *reading*: gkyl-backed data yields a read-only NumPy view of
    the C buffer (valid while this dataset is alive); numpy-backed data yields
    the array itself. Mutation of modal data must go through the kernels."""
    if isinstance(self._values, gpython.GkylArray):
      return self._values.view(self.ctx.get("cells"))
    return self._values

  def set_values(self, values) -> None:
    """Replace stored values and update cell/component metadata."""
    self._values = values
    if isinstance(values, gpython.GkylArray):
      # Cell layout is not derivable from the flat native array; it comes from
      # ctx (set by the reader, and carried through metadata-only copies).
      self.ctx["num_comps"] = values.ncomp
    else:
      self.ctx["cells"] = np.array(values.shape[:-1], dtype=np.int64)
      self.ctx["num_comps"] = int(values.shape[-1])

  values = property(get_values, set_values)

  def __getitem__(self, index):
    """Index values using their ordinary NumPy axis order.

    The value layout is ``(*spatial_axes, component)``, so, for example, a
    one-dimensional four-component dataset supports ``data[:, 2:4]``.  This
    deliberately mirrors indexing ``data.values`` rather than treating every
    subscript as a component-only selector.  A component is selected explicitly with ``data[..., 1]``.
    """
    if self._values is None:
      raise ValueError("GData values are not loaded; cannot subscript.")
    return self.get_values()[index]

  def __setitem__(self, index, value) -> None:
    """Assign through NumPy-style indexing on NumPy-backed data.

    Native Gkeyll storage is intentionally read-only from Python; use the
    appropriate operation/kernel, or interpolate first, before mutating it.
    """
    if self._values is None:
      raise ValueError(
          "GData values are not loaded; cannot assign by subscript.")
    if isinstance(self._values, gpython.GkylArray):
      raise ValueError(
          "Cannot assign through indexing to native Gkeyll storage; call "
          ".interpolate() first to obtain mutable NumPy-backed values.")
    self._values[index] = value

  def push(self, grid, values):
    """Set values (updating cell/comp ctx) then the grid (updating bounds)."""
    self.set_values(values)
    self.set_grid(grid)
    return self

  # ------------------------------------------------------------- duplication
  def clone(self, metadata_only: bool = False) -> "GDataState":
    """Deep-copy without re-reading. Builds ``type(self)`` so subclasses
    (e.g. the fluent ``GData``) propagate through every verb result.

    Set ``metadata_only=True`` to omit the grid and values from the copy.
    """
    new = type(self)(tag=self._tag, label=self._custom_label, ctx=self.ctx)
    new.set_label(self._label)
    new._file_name = self._file_name
    new.color = self.color
    if not metadata_only and self._values is not None:
      dup = (self._values.clone() if isinstance(self._values, gpython.GkylArray)
             else np.array(self._values, copy=True))
      new.push([np.array(g, copy=True) for g in self._grid], dup)
    return new

  def _result(self,
              grid,
              values,
              *,
              inplace: bool = False,
              tag: str | None = None,
              label: str | None = None,
              **ctx_updates):
    """The single 'mutate self vs. emit a new dataset' decision point.

    Every verb funnels its computed ``(grid, values)`` through here. Because
    ``copy`` uses ``type(self)``, the result is the *same* (sub)class as the
    input -- so ``operations`` can be typed on ``GDataState`` yet return a fluent
    ``GData`` at runtime.
    """
    target = self if inplace else self.clone(metadata_only=True)
    target.push(grid, values)
    if tag is not None:
      target.set_tag(tag)
    if label is not None:
      target._custom_label = label
    if ctx_updates:
      target.ctx.update(ctx_updates)
    return target

  # ---------------------------------------------------------- operability
  @property
  def is_interpolated(self) -> bool:
    """True when values are safe for element-wise math: data with no DG
    structure at all (no ``basis_type`` -- plain point values by
    construction), never-modal DG data (``value_form`` is ``nodal``/
    ``quad``), or modal data already run through ``interpolate``
    (``ctx['interpolated']``)."""
    if not self.ctx.get("basis_type"):
      return True
    return (self.ctx.get("value_form", "modal") != "modal"
            or self.ctx.get("interpolated", False))

  def _require_operable(self) -> None:
    """Pointwise math is allowed exactly where the data are point values:
    the NumPy field domain, or the nodal/quad value forms. Modal
    coefficients refuse -- a pointwise operation has no basis-space meaning.
    ``value_form`` applies uniformly regardless of ``backend`` -- it is the
    one fact for "what do these values mean", set once at load time."""
    if self._values is None:
      raise ValueError("GData has no values to operate on.")
    if not self.is_interpolated:
      raise ValueError(
          "Cannot do NumPy math on modal DG coefficients. Convert explicitly: "
          ".to_nodal()/.to_quad() (pointwise, stays native), .apply(fn) "
          "(pointwise via quadrature, projects back to modal), or .interpolate() "
          "(leave for the NumPy field domain).")

  # ----------------------------------------------------- numpy interop (read)
  _HANDLED_TYPES = (numbers.Number, np.ndarray, np.generic)

  def __array__(self, dtype=None):
    """Expose values so ``np.asarray(data)`` / matplotlib accept the dataset.

    This is a pure *reader* (no ``operations``), so it lives on the container; the
    computing operators (``__add__``, ``__array_ufunc__``) live on the fluent
    subclass -- see HIERARCHY_3.md. Nodal/quad data expose their point values;
    native *modal* data refuses: silently handing out DG coefficients as if
    they were point values is a correctness trap."""
    if isinstance(self._values, gpython.GkylArray):
      if self.ctx.get("value_form", "modal") != "modal":
        return np.asarray(self.get_values(), dtype=dtype)
      raise ValueError(
          "This dataset holds modal DG coefficients in native Gkeyll storage; "
          ".to_nodal()/.to_quad() for point values, or .interpolate() for NumPy."
      )
    return np.asarray(self._values, dtype=dtype)

  # -------------------------------------------------------------- reporting
  def info(self, index: int = 0, no_header: bool = False) -> str:
    """Build and print a summary; optionally omit its descriptive heading."""
    values, num_comps = self.get_values(), self.num_comps
    num_dims, num_cells = self.num_dims, self.num_cells
    lo, up = self.bounds
    out = ""
    if not no_header:
      lbl = self.get_label()
      out += f"{lbl}{' ' if lbl else ''}({self.get_tag()}#{index})\n"
    if "time" in self.ctx:
      out += f"├─ Time: {self.ctx['time']:e}\n"
    if "frame" in self.ctx:
      out += f"├─ Frame: {self.ctx['frame']:d}\n"
    if self.ctx.get("block") is not None:
      out += f"├─ Block: {self.ctx['block']:d} (sim '{self.ctx.get('sim', '')}')\n"
    out += f"├─ Number of components: {num_comps:d}\n"
    out += f"├─ Number of dimensions: {num_dims:d}\n"
    if lo is not None:
      out += f"├─ Grid: ({self.get_grid_type()})\n"
      for d in range(num_dims):
        branch = "└" if d == num_dims - 1 else "├"
        out += (f"│  {branch}─ Dim {d}: Num. cells: {int(num_cells[d]):d}; "
                f"Lower: {lo[d]:e}; Upper: {up[d]:e}\n")
    if values is not None:
      vmax = np.nanmax(values)
      vmin = np.nanmin(values)
      max_idx = np.unravel_index(np.nanargmax(values), values.shape)
      min_idx = np.unravel_index(np.nanargmin(values), values.shape)
      max_pos = tuple(int(i) for i in max_idx[:num_dims])
      min_pos = tuple(int(i) for i in min_idx[:num_dims])
      out += f"├─ Maximum: {vmax:e} at {max_pos}"
      out += f" component {int(max_idx[-1]):d}\n" if num_comps > 1 else "\n"
      out += f"├─ Minimum: {vmin:e} at {min_pos}"
      out += f" component {int(min_idx[-1]):d}\n" if num_comps > 1 else "\n"
    if self.ctx.get("basis_type"):
      form = self.ctx.get("value_form", "modal")
      if self.ctx.get("interpolated"):
        form = "interpolated"
      elif form == "quad" and self.ctx.get("num_quad"):
        form = f"quad, num_quad={self.ctx['num_quad']}"
      out += f"├─ DG: {self.ctx['basis_type']} p{self.ctx.get('poly_order', '?')} ({form})\n"
    if "changeset" in self.ctx or "builddate" in self.ctx:
      out += "├─ Created with Gkeyll:\n"
      if "changeset" in self.ctx:
        out += f"│  ├─ Changeset: {self.ctx['changeset']}\n"
      if "builddate" in self.ctx:
        out += f"│  └─ Build Date: {self.ctx['builddate']}\n"
    if "geometry_type" in self.ctx or "geqdsk_sign_convention" in self.ctx:
      out += "├─ Geometry info:\n"
      if "geometry_type" in self.ctx:
        out += f"│  ├─ Type: {self.ctx['geometry_type']}\n"
      if "geqdsk_sign_convention" in self.ctx:
        out += f"│  ├─ GEQDSK sign convention: {self.ctx['geqdsk_sign_convention']:d}\n"
    if any(k in self.ctx for k in ("mass", "charge", "gas_gamma", "vdim")):
      out += "├─ Species properties:\n"
      if "mass" in self.ctx:
        out += f"│  ├─ Mass: {self.ctx['mass']:e}\n"
      if "charge" in self.ctx:
        out += f"│  ├─ Charge: {self.ctx['charge']:e}\n"
      if "gas_gamma" in self.ctx:
        out += f"│  ├─ Adiabatic index: {self.ctx['gas_gamma']:e}\n"
      if "vdim" in self.ctx:
        out += f"│  ├─ Velocity dimensions: {self.ctx['vdim']:d}\n"
    for key, val in self.ctx.items():
      if key not in self._INFO_HANDLED_CTX_KEYS:
        out += f"├─ {key}: {val}\n"
    out += "└─ File: " + (self._file_name or "<no file>") + "\n"
    print(out)
    return out

  # Keys already rendered by a dedicated branch above; anything else in ctx
  # is file/reader-native metadata (e.g. a .gkyl file's msgpack meta) that
  # still deserves to surface, so it falls through to the generic dump.
  _INFO_HANDLED_CTX_KEYS = frozenset({
      "time",
      "frame",
      "sim",
      "block",
      "quantity",
      "lower",
      "upper",
      "cells",
      "grid_type",
      "poly_order",
      "basis_type",
      "num_comps",
      "value_form",
      "num_quad",
      "interpolated",
      "changeset",
      "builddate",
      "geometry_type",
      "geqdsk_sign_convention",
      "mass",
      "charge",
      "gas_gamma",
      "vdim",
  })

  # --------------------------------------------------------------- summary
  def _summary(self) -> str:
    if self._values is None:
      return f"<{type(self).__name__} empty | tag '{self._tag}'>"
    cells = tuple(int(c) for c in self.get_num_cells())
    parts = [f"<{type(self).__name__} {cells}", f"{self.num_comps:d} comp"]
    lo, up = self.bounds
    if lo is not None:
      parts.append(" ".join(f"[{lo[d]:g},{up[d]:g}]"
                            for d in range(self.num_dims)))
    value_form = self.ctx.get("value_form", "modal")
    if self.ctx.get("basis_type"):
      dg = str(self.ctx["basis_type"])
      if self.ctx.get("poly_order") is not None:
        dg += f" p{self.ctx['poly_order']}"
      if self.ctx.get("interpolated"):
        dg += " interpolate"
      elif value_form == "modal":
        dg += " modal"
      parts.append(dg)
    if self.backend == "gkyl":
      parts.append("gkyl-native" if value_form ==
                   "modal" else f"gkyl-native ({value_form})")
    parts.append(f"tag '{self._tag}'")
    return " | ".join(parts) + ">"

  def __repr__(self) -> str:
    return self._summary()

  def __str__(self) -> str:
    if self._values is None:
      return self._summary()
    return (f"{self._summary()}\n"
            f"{np.array2string(self.get_values(), threshold=20, edgeitems=2)}")
