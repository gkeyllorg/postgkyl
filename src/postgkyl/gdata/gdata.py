"""``GData`` -- the fluent surface (the FLUENT API layer).

A thin subclass of the verb-less :class:`~postgkyl.gdatastate.gdatastate.GDataState`
container that adds the fluent verb methods and the computing operators. Because
this module sits *above* ``operations``/``render``/``io``, it imports them with plain
top-level imports -- there is **no import cycle and no lazy import anywhere**.

Inherited from the container (pure state readers): ``info``, ``__array__``,
``__repr__``/``__str__``, all shape properties, ``copy``/``_result``.
"""

from __future__ import annotations

import operator

import numpy as np

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl import operations, io

from .gdatagroup import GDataGroup


class GData(GDataState):
  """Fluent dataset: ``pg.load(...).interpolate().select(z0=0.0).plot()``."""

  # ---------------------------------------------------------- fluent verbs
  def interpolate(self, *, num_interp: int | None = None, inplace: bool = False,
      tag: str | None = None, label: str | None = None) -> "GData":
    """Interpolate DG coefficients onto a uniform mesh (see ``operations.interpolate``).

    Basis, polynomial order, and value_form are properties of this dataset,
    fixed at load time -- this method never re-specifies them.
    """
    return operations.interpolate(self, num_interp=num_interp,
        inplace=inplace, tag=tag, label=label)
  # end

  def local_poly(self, *, npoints: int = 2, inplace: bool = False,
      tag: str | None = None, label: str | None = None) -> "GData":
    """Evaluate the DG polynomial cell-by-cell onto a discontinuity-preserving
    plotting mesh (see ``operations.local_poly``). Basis/order/value_form are
    properties of this dataset, fixed at load time."""
    return operations.local_poly(self, npoints=npoints,
        inplace=inplace, tag=tag, label=label)
  # end

  def select(self, *, comp=None, z0=None, z1=None, z2=None, z3=None, z4=None,
      z5=None, inplace: bool = False, tag: str | None = None,
      label: str | None = None) -> "GData":
    """Subselect coordinates/components (see ``operations.select``)."""
    return operations.select(self, comp=comp, z0=z0, z1=z1, z2=z2, z3=z3, z4=z4,
        z5=z5, inplace=inplace, tag=tag, label=label)
  # end

  def plot(self, **kwargs):
    """Render this dataset and return the Matplotlib figure.

    Pass ``save=True`` for an auto-named PNG or ``saveas=...`` for a PNG/PDF
    output path; no CLI glue is required.
    """
    return operations.plot(self, **kwargs)
  # end

  def plotly(self, **kwargs):
    """Render this dataset with Plotly (terminal verb; see ``operations.plotly``).

    ``d.plotly()`` alone just builds and returns the figure; pass
    ``show=True`` to open an auto-rotating browser preview, or
    ``save=True``/``saveas=...`` to write it instead -- no CLI glue needed
    either way (see ``render.plotly``'s docstring).
    """
    return operations.plotly(self, **kwargs)
  # end

  def save(self, out_name: str = "", extension: str = "gkyl") -> str:
    """Write this dataset to disk (see ``io.save``)."""
    return io.save(self, out_name=out_name, extension=extension)
  # end

  # ``info`` is inherited from GDataState (a pure state reader).

  # ----------------------------------------------------------- modal verbs
  # Explicit spellings of the weak algebra (the * and / operators dispatch to
  # the same Gkeyll kernels when both operands are modal).
  def mul(self, other) -> "GData":
    """Weak (DG) multiply -- runs inside Gkeyll on modal data."""
    return operations.arithmetic.binary(operator.mul, self, other)
  # end

  def div(self, other) -> "GData":
    """Weak (DG) divide -- runs inside Gkeyll on modal data."""
    return operations.arithmetic.binary(operator.truediv, self, other)
  # end

  def integrate(self, *, op: str = "none"):
    """Grid integral of modal data via ``gkyl_array_integrate`` (terminal).

    ``op`` is ``"none"``, ``"abs"``, or ``"sq"``; returns a float (one field)
    or a NumPy array (one value per field)."""
    return operations.integrate(self, op=op)
  # end

  def integrate_axis(self, axis: int | tuple | str | None = None, *,
      inplace: bool = False, tag: str | None = None,
      label: str | None = None) -> "GData":
    """Trapezoidal integral over one or more axes of point-value data
    (see ``operations.integrate_axis``); a new (reduced) dataset, like ``.select()``.

    Works on already-interpolated (NumPy) data or a native nodal/quad
    value_form; raw modal DG coefficients raise -- convert explicitly
    first (``.interpolate()``/``.to_nodal()``/``.to_quad()``).
    """
    return operations.integrate_axis(self, axis, inplace=inplace, tag=tag, label=label)
  # end

  def average(self, dims, *, weight: "GData | None" = None,
      inplace: bool = False, tag: str | None = None,
      label: str | None = None) -> "GData":
    """Weighted (or plain) average of modal data over ``dims`` via
    ``gkyl_array_average`` (see ``operations.average``).

    Runs inside Gkeyll on native modal (pre-``interpolate()``) data, unlike
    ``integrate_axis``; produces a new, lower-dimensional dataset, still
    modal and gkyl-native (so it composes with further ``.average()``/
    ``.to_nodal()``/``.interpolate()`` calls, unlike the terminal
    ``.integrate()``).
    """
    return operations.average(self, dims, weight=weight, inplace=inplace, tag=tag,
        label=label)
  # end

  def eval_at_coord_proj(self, eval_dirs, eval_coords, *,
      inplace: bool = False, tag: str | None = None,
      label: str | None = None) -> "GData":
    """Evaluate modal data at physical coordinates in ``eval_dirs`` and
    project onto the surviving directions' target basis, via
    ``gkyl_dg_eval_at_coord_proj`` (see ``operations.eval_at_coord_proj``).

    Runs inside Gkeyll on native modal (pre-``interpolate()``) data, like
    ``.average()``; produces a new, lower-dimensional dataset, still modal
    and gkyl-native.
    """
    return operations.eval_at_coord_proj(self, eval_dirs, eval_coords,
        inplace=inplace, tag=tag, label=label)
  # end

  # --------------------------------------------- value_form changes (explicit)
  # Conversions never happen implicitly -- these verbs are the only doorway
  # between the modal / nodal / quadrature value_forms (all gkyl-native).
  def to_modal(self, **kwargs) -> "GData":
    """Convert to modal coefficients (exact from nodal; projection from quad)."""
    return operations.represent(self, to="modal", **kwargs)
  # end

  def to_nodal(self, **kwargs) -> "GData":
    """Convert to values at the basis nodes (exact, invertible)."""
    return operations.represent(self, to="nodal", **kwargs)
  # end

  def to_quad(self, num_quad: int | None = None, **kwargs) -> "GData":
    """Convert to values at Gauss–Legendre points (default ``p+1`` per dim)."""
    return operations.represent(self, to="quad", num_quad=num_quad, **kwargs)
  # end

  def apply(self, fn, *, num_quad: int | None = None, **kwargs) -> "GData":
    """Pointwise ``fn`` via quadrature (modal -> quad -> fn -> modal), e.g.
    ``d.apply(np.sqrt)``. The explicit spelling of nonlinear pointwise math
    on DG data; raise ``num_quad`` to de-alias."""
    return operations.apply(self, fn, num_quad=num_quad, **kwargs)
  # end

  # ------------------------------------------------- field-domain analysis
  # Equation-blind core verbs from layers 07-09 (``operations/__init__.py``), each a
  # one-line delegation to its matching ``operations`` function.
  def fft(self, *, psd: bool = False, iso: bool = False, inplace: bool = False,
      tag: str | None = None, label: str | None = None) -> "GData":
    """Fourier transform / power spectral density (see ``operations.fft``)."""
    return operations.fft(self, psd=psd, iso=iso, inplace=inplace, tag=tag, label=label)
  # end

  def magsq(self, *, coords: str = "0:3", inplace: bool = False,
      tag: str | None = None, label: str | None = None) -> "GData":
    """Magnitude squared of a vector field (see ``operations.magsq``)."""
    return operations.magsq(self, coords=coords, inplace=inplace, tag=tag, label=label)
  # end

  def mask(self, mask_data: "GData | None" = None, *, lower: float | None = None,
      upper: float | None = None, inplace: bool = False, tag: str | None = None,
      label: str | None = None) -> "GData":
    """Mask values by a mask dataset or numeric thresholds (see ``operations.mask``)."""
    return operations.mask(self, mask_data, lower=lower, upper=upper, inplace=inplace,
        tag=tag, label=label)
  # end

  def val2coord(self, *, x: str, y: str, periodic: bool = False,
      tag: str | None = None, label: str | None = None) -> "GDataGroup":
    """Build new (x, y) datasets from DynVector columns (see ``operations.val2coord``).

    Wraps the ``operations`` verb's (verb-less) ``core.GDataStateGroup`` result in a
    fluent :class:`~postgkyl.gdata.gdatagroup.GDataGroup` so the chain keeps going,
    e.g. ``d.val2coord(x='0', y='1:3')[0].plot()``.
    """
    return GDataGroup(operations.val2coord(self, x=x, y=y, periodic=periodic,
        tag=tag, label=label))
  # end

  def extract_input(self) -> str:
    """Decode the input file embedded in ``ctx`` (see ``operations.extract_input``);
    a terminal verb returning a plain ``str`` (``""`` if none is embedded)."""
    return operations.extract_input(self)
  # end

  def fit(self, fit_type: str, *, guess=None, window: bool = False,
      min_n: int | None = None, inplace: bool = False, tag: str | None = None,
      label: str | None = None) -> "GData":
    """Fit a model to this dataset (see ``operations.fit``).

    ``window=True`` fits only the best-scoring leading window of a 1D
    series -- the growth-rate use case, e.g. ``d.fit('exp2', window=True)``.
    """
    return operations.fit(self, fit_type, guess=guess, window=window, min_n=min_n,
        inplace=inplace, tag=tag, label=label)
  # end

  def differentiate(self, *, direction: int | None = None, inplace: bool = False,
      tag: str | None = None, label: str | None = None) -> "GData":
    """Numerical gradient of field-domain data (see ``operations.differentiate``)."""
    return operations.differentiate(self, direction=direction, inplace=inplace, tag=tag,
        label=label)
  # end

  def map(self, mapping: "str | GData", *, space: str = "conf",
      basis_type: str | None = None, poly_order: int | None = None,
      inplace: bool = False, tag: str | None = None,
      label: str | None = None) -> "GData":
    """Deform this dataset's grid by evaluating a coordinate map (see ``operations.map``)."""
    return operations.map(self, mapping, space=space, basis_type=basis_type,
        poly_order=poly_order, inplace=inplace, tag=tag, label=label)
  # end

  # Note: no fluent ``grid`` method. ``GData.grid`` (inherited from
  # GDataState) is the axis-edge-array property that most of ``operations`` reads
  # via plain attribute access (``data.grid``); a same-named verb method
  # would shadow it for every GData instance and silently break every other
  # verb. ``operations.grid`` (the "turn a dataset's grid into a dataset of
  # coordinates" verb) is reachable as ``postgkyl.operations.grid(data, ...)`` --
  # src_bak's GData carried the identical exception with the identical
  # reasoning (src_bak/postgkyl/data/gdata.py:1258-1259).

  # ------------------------------------------------------ binary operators
  def __add__(self, o):      return operations.arithmetic.binary(operator.add, self, o)
  def __sub__(self, o):      return operations.arithmetic.binary(operator.sub, self, o)
  def __mul__(self, o):      return operations.arithmetic.binary(operator.mul, self, o)
  def __truediv__(self, o):  return operations.arithmetic.binary(operator.truediv, self, o)
  def __pow__(self, o):      return operations.arithmetic.binary(operator.pow, self, o)

  def __radd__(self, o):     return operations.arithmetic.binary(operator.add, o, self)
  def __rsub__(self, o):     return operations.arithmetic.binary(operator.sub, o, self)
  def __rmul__(self, o):     return operations.arithmetic.binary(operator.mul, o, self)
  def __rtruediv__(self, o): return operations.arithmetic.binary(operator.truediv, o, self)
  def __rpow__(self, o):     return operations.arithmetic.binary(operator.pow, o, self)

  # ----------------------------------------------------------------- unary
  def __neg__(self): return operations.arithmetic.binary(operator.mul, self, -1.0)
  def __abs__(self): return operations.arithmetic.apply_ufunc(np.absolute, "__call__", self)
  def __pos__(self): return self.clone()

  # --------------------------------------------------------- NumPy interop
  __array_priority__ = 100  # ndarray defers to us in mixed ndarray·GData ops

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """Apply NumPy ufuncs while preserving pointwise dataset metadata.

    Pointwise calls such as ``np.sqrt``/``np.add`` return a GData carrying the
    grid/ctx; reductions such as ``np.max``/``np.sum`` return NumPy results.
    """
    return operations.arithmetic.apply_ufunc(ufunc, method, *inputs, **kwargs)
  # end
# end
