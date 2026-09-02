"""``GData`` -- the fluent surface (the FLUENT API layer).

A thin subclass of the verb-less :class:`~postgkyl.gdatastate.gdatastate.GDataState`
container that adds the fluent verb methods and the computing operators. Because
this module sits *above* ``operations``/``render``/``io``, it imports them with plain
top-level imports -- there is **no import cycle and no lazy import anywhere**.

Inherited from the container (pure state readers): ``info``, ``__array__``,
``__repr__``/``__str__``, all shape properties, ``copy``/``_result``.
"""

from __future__ import annotations

from glob import has_magic
import operator

import numpy as np

from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl import operations, io
from postgkyl.command_spec import hidden

from .gdatagroup import GDataGroup


class GData(GDataState):
  """Fluent dataset: ``pg.load(...).interpolate().select(z0=0.0).plot()``."""

  # ------------------------------------------------------- data lifecycle
  def load(self, file_name: str, *, tag: str | None = None,
      label: str | None = None, ctx: dict | None = None,
      value_form: str | None = None, basis_type: str | None = None,
      poly_order: int | None = None, **read_kwargs) -> "GData":
    """Load one file into this dataset in place and return ``self``.

    This is the two-step counterpart of constructing ``GData(file_name)``::

        data = GData()
        data.load(file_name).local_poly().plot()

    The read is atomic with respect to this object: if it fails, the current
    grid, values, context, and filename are left unchanged.  A pristine empty
    dataset's existing ``ctx`` seeds the read unless ``ctx`` is supplied
    explicitly; reloading an already populated dataset starts from a fresh
    context so metadata from the previous file cannot leak into the new one.
    The dataset's existing tag and custom label are preserved unless ``tag``
    or ``label`` is passed.

    This method accepts one literal filename.  Use :func:`postgkyl.load` for
    shell-style glob patterns, which produce a ``GDataGroup`` rather than one
    dataset.
    """
    file_name = str(file_name)
    if not file_name:
      raise ValueError("GData.load() requires a non-empty filename.")
    # end
    if has_magic(file_name):
      raise ValueError(
          "GData.load() accepts one literal filename; use pg.load(pattern) "
          "to load a glob as a GDataGroup.")
    # end

    if ctx is None and self._grid is None and self._values is None:
      load_ctx = self.ctx
    # end
    else:
      load_ctx = ctx
    # end

    # Construct through GDataState so this follows exactly the same reader and
    # metadata-defaulting path as GData(file_name).  Nothing on ``self`` is
    # changed until construction succeeds.
    loaded = GDataState(file_name, ctx=load_ctx, value_form=value_form,
        basis_type=basis_type, poly_order=poly_order, **read_kwargs)
    self._grid = loaded._grid
    self._values = loaded._values
    self.ctx = loaded.ctx
    self._file_name = loaded._file_name
    self._label = loaded._label
    if tag is not None:
      self._tag = tag
    # end
    if label is not None:
      self._custom_label = label
    # end
    return self
  # end

  # ---------------------------------------------------------- fluent verbs
  # These are ordinary class-body aliases, not wrappers or runtime setattr
  # calls. Python binds the leading dataset argument as ``self``. The alias
  # keeps the signature, annotations, docstring, command metadata, and
  # implementation in one canonical function while remaining discoverable by
  # static language servers such as VS Code/Pylance.
  interpolate = operations.interpolate
  local_poly = operations.local_poly
  gk_rz = operations.gyrokinetics.gk_rz
  gk_fluxsurf = operations.gyrokinetics.gk_fluxsurf
  select = operations.select
  integrate = operations.integrate
  integrate_axis = operations.integrate_axis
  average = operations.average
  eval_at_coord_proj = operations.eval_at_coord_proj
  fft = operations.fft
  magsq = operations.magsq
  mask = operations.mask
  extract_input = operations.extract_input
  fit = operations.fit
  growth = operations.growth
  differentiate = operations.differentiate
  map = operations.map
  apply = operations.apply
  save = io.save
  plot = operations.plot
  plotly = operations.plotly

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

  # ------------------------------------------------- field-domain analysis
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


for _name, _reason in {
    "load": "the canonical loader is postgkyl.load",
    "mul": "Python operators are not stringly exposed as commands",
    "div": "Python operators are not stringly exposed as commands",
    "to_modal": "representation shortcuts remain Python-only",
    "to_nodal": "representation shortcuts remain Python-only",
    "to_quad": "representation shortcuts remain Python-only",
    "val2coord": "the functional operation owns this exceptional group result",
}.items():
  hidden(_reason)(GData.__dict__[_name])
# end
