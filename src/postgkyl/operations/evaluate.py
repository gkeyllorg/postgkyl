"""The ``evaluate`` verb -- evaluate RPN math expressions over datasets.

The numeric operators live in :mod:`postgkyl.numerics.ev_ops` (pure
``(grid, values)`` functions, keyed by token in ``numerics.ev_cmds``); this
module is the stack machine that drives them and the glue that resolves
``f``/``fN`` tokens against an explicit list of datasets.

Expressions use Reverse Polish Notation, e.g. ``"f0 f1 +"`` adds two datasets
and ``"f 2 *"`` doubles one. Data tokens are:

- ``f`` / ``fN``  -- the ``N``-th provided dataset (``f`` == ``f0``),
- ``fN[c]``       -- component ``c`` of that dataset (slices like ``0:3`` work),
- ``fN.key``      -- the scalar ``ctx[key]`` of that dataset.

Anything else is parsed as a numeric/axis literal (a float, a ``"0,1"`` /
``"0:3"`` axis spec, or a Python literal in brackets/parens). Every operator
in ``numerics.ev_cmds`` is a plain array function -- none needed a
``NotImplementedError`` GData-only placeholder (see the numerics module
docstring), so there is nothing left to resolve here.

A data token referencing native (gkyl-backed) data is kept native, not
forced through ``select()``'s point-value guard, regardless of
value_form -- see ``_native_kernel``:

- **modal** (raw DG coefficients): ``+ - * /`` and integer ``pow``/``sq``
  route through Gkeyll's own weak DG kernels, the same math
  ``operations.arithmetic`` uses for the ``GData`` operators. An operator
  with no weak-kernel meaning (``sqrt``, ``sin``, reductions, ...) -- or one
  Gkeyll's kernel itself refuses for this basis/order -- warns and falls
  back to plain NumPy math on the raw coefficient view, rather than
  hard-blocking: value_form/basis metadata is sometimes simply wrong (a
  diagnostic file mistagged "modal" by its writer; see the load-time
  ``--value_form`` override), and the raw view is exact whenever
  coefficient 0 already *is* the point value (e.g. p0 data).
- **nodal/quad** (point values): every operator in ``_POINTWISE_TOKENS``
  (``+ - * / pow sq sqrt sin cos tan abs log log10 exp max2 min2
  scale_comp scale_zi_axis``) is exact regardless of packing, so it is
  computed with plain NumPy on the view and the result is wrapped back into
  a native array -- computed on the view, wrapped back native, staying
  in-value_form, mirroring ``operations.arithmetic``'s ufunc dispatch.
  Anything else (``dot``, ``avg``, ``max``, ``min``, ``mean``, ``len``,
  ``grad``, ``grad2``, ``int``, ``div``, ``curl``) is a genuine reduction or
  finite-difference derivative -- not a per-point transform -- so it leaves
  the native domain for plain NumPy math on the raw view, same as before;
  ``apply_operator`` then strips the now-stale ``value_form`` tag and
  marks the result ``interpolated`` (mirroring ``.interpolate()``) so
  ``info()`` doesn't keep claiming a value_form the data no longer has.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING

import numpy as np

from postgkyl import dg
from postgkyl.numerics import ev_cmds
from postgkyl.operations.select import select

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState
# end

# RPN tokens with an exact Gkeyll weak-kernel meaning on modal data.
_MODAL_BINARY_OPS = {"+", "-", "*", "/", "pow"}

# RPN tokens that are exact, shape-preserving pointwise math on nodal/quad
# point values (elementwise, no cross-cell/cross-node access, no reduction)
# -- safe to compute on the raw view and wrap back into a native array.
# Everything else in numerics.ev_cmds (dot, avg, max, min, mean, len, grad,
# grad2, int, div, curl) is a reduction or a finite-difference derivative
# and must leave the native domain instead.
_POINTWISE_TOKENS = frozenset({
    "+", "-", "*", "/", "pow", "sq", "sqrt", "sin", "cos", "tan", "abs",
    "log", "log10", "exp", "max2", "min2", "scale_comp", "scale_zi_axis",
})

# f, f0, f12 ... with optional [comp] selection and optional .ctxkey suffix.
_DATA_TOKEN = re.compile(r"^f(\d*)(?:\[([^\]]*)\])?(?:\.(\w+))?$")


def _rep_of(ctx: dict) -> str:
  return ctx.get("value_form", "modal")
# end


def _compare(a, b) -> bool:
  """Equality that also handles NumPy arrays (used when merging ctx dicts)."""
  if isinstance(a, np.ndarray):
    return np.array_equal(a, b)
  # end
  return a == b
# end


def _modal_view(value, ctx: dict):
  """Read-only NumPy view of a native modal operand, for the (warned)
  pointwise fallback; anything else passes through unchanged."""
  if dg.modal.is_native(value):
    return value.view(ctx.get("cells"))
  # end
  return value
# end


def _basis_of(ctx: dict):
  basis_type, poly_order = ctx.get("basis_type"), ctx.get("poly_order")
  if basis_type is None or poly_order is None:
    raise ValueError("modal operand has no basis_type/poly_order metadata")
  # end
  return str(basis_type), int(poly_order)
# end


def _as_scalar(value):
  """A Python float if ``value`` is scalar-shaped, else None."""
  if isinstance(value, (int, float, np.integer, np.floating)):
    return float(value)
  # end
  if isinstance(value, np.ndarray) and value.ndim == 0:
    return float(value)
  # end
  return None
# end


def _modal_kernel(token: str, tmp_grid, tmp_values, tmp_ctx):
  """Try to compute ``token`` via Gkeyll's own weak DG kernels when a modal
  (native, raw-DG-coefficient) operand is present.

  Returns ``(out_grid, out_values)`` when the operator has an exact modal
  meaning and Gkeyll's kernel accepts this basis/order (``+ - * /`` and
  integer ``pow``/``sq``). Returns ``None`` when no operand is modal
  (nothing to do here -- the caller runs the plain NumPy ``func`` as usual).

  Deliberately never raises: basis/value_form metadata can be wrong
  (a diagnostic file mistagged "modal" by its writer), so an operator with
  no weak-kernel form, a basis/kernel Gkeyll itself refuses, or a
  non-scalar second operand all warn and return ``None`` too -- the caller
  then falls back to plain NumPy math on the raw coefficient view (exact
  whenever coefficient 0 already *is* the point value, e.g. p0 data).
  """
  is_modal = [dg.modal.is_native(v) for v in tmp_values]
  if not any(is_modal):
    return None
  # end

  try:
    if len(tmp_values) == 1:
      if token != "sq":
        raise ValueError(f"'{token}' has no weak-kernel form")
      # end
      basis_type, poly_order = _basis_of(tmp_ctx[0])
      out = dg.modal.power(basis_type, len(tmp_grid[0]), poly_order, tmp_values[0], 2)
      return [tmp_grid[0]], [out]
    # end

    if len(tmp_values) == 2:
      if token not in _MODAL_BINARY_OPS:
        raise ValueError(f"'{token}' has no weak-kernel form")
      # end
      # RPN order: tmp_values[0] is "b" (top of stack), tmp_values[1] is "a".
      a, b = tmp_values[1], tmp_values[0]
      a_modal, b_modal = is_modal[1], is_modal[0]

      if a_modal and b_modal:
        grid = tmp_grid[1] if tmp_grid[1] is not None else tmp_grid[0]
        basis_a, basis_b = _basis_of(tmp_ctx[1]), _basis_of(tmp_ctx[0])
        if basis_a != basis_b:
          raise ValueError(f"operands have different DG bases ({basis_a} vs {basis_b})")
        # end
        basis_type, poly_order = basis_a
        ndim = len(grid)
        if token == "+":
          out = dg.modal.lincomb(1.0, a, 1.0, b)
        # end
        elif token == "-":
          out = dg.modal.lincomb(1.0, a, -1.0, b)
        # end
        elif token in ("*", "/"):
          fn = dg.modal.weak_mul if token == "*" else dg.modal.weak_div
          out = fn(basis_type, ndim, poly_order, a, b)
        # end
        else:
          raise ValueError("'pow' is not defined between two modal datasets")
        # end
        return [grid], [out]
      # end

      # Exactly one operand is modal; the other must be a plain scalar.
      modal_arr, modal_ctx, modal_grid = (a, tmp_ctx[1], tmp_grid[1]) if a_modal \
          else (b, tmp_ctx[0], tmp_grid[0])
      other = b if a_modal else a
      scalar = _as_scalar(other)
      if scalar is None:
        raise ValueError("cannot mix native modal data with a plain array")
      # end
      basis_type, poly_order = _basis_of(modal_ctx)
      ndim = len(modal_grid)
      scalar_first = not a_modal  # the scalar came first in the expression

      if token == "*":
        out = dg.modal.scale(modal_arr, scalar)
      # end
      elif token == "/":
        out = (dg.modal.scale(dg.modal.weak_inv(basis_type, ndim, poly_order, modal_arr), scalar)
               if scalar_first else dg.modal.scale(modal_arr, 1.0 / scalar))
      # end
      elif token == "+":
        out = dg.modal.shift_mean(basis_type, ndim, poly_order, modal_arr, scalar)
      # end
      elif token == "-":
        out = (dg.modal.shift_mean(basis_type, ndim, poly_order,
                   dg.modal.scale(modal_arr, -1.0), scalar) if scalar_first
               else dg.modal.shift_mean(basis_type, ndim, poly_order, modal_arr, -scalar))
      # end
      else:  # pow
        if scalar_first or not float(scalar).is_integer() or scalar < 1:
          raise ValueError(
              f"modal 'pow' needs a modal base and a positive integer "
              f"exponent, got exponent {scalar!r} (scalar_first={scalar_first})")
        # end
        out = dg.modal.power(basis_type, ndim, poly_order, modal_arr, int(scalar))
      # end
      return [modal_grid], [out]
    # end

    raise ValueError(f"'{token}' has no weak-kernel form for {len(tmp_values)} operands")
  # end
  except Exception as err:
    warnings.warn(
        f"evaluate: '{token}' on native modal (raw DG coefficient) data: {err}; "
        "falling back to plain math on the raw coefficient view -- exact only "
        "if coefficient 0 already IS the point value (e.g. p0 data, or a file "
        "whose 'modal' tag is wrong; see --value_form).", stacklevel=3)
    return None
  # end
# end


def _native_kernel(token: str, tmp_grid, tmp_values, tmp_ctx, func):
  """Dispatch a native (gkyl-backed) operand to the value_form-correct math.

  Returns ``(out_grid, out_values)`` -- with ``out_values`` wrapped back into
  native arrays whenever the result stays a per-point/per-coefficient field
  -- or ``None`` when nothing here applies (the caller runs the plain NumPy
  ``func`` on the raw view as usual, e.g. for reductions/derivatives).

  - Every native operand modal: delegates to :func:`_modal_kernel` (weak
    DG kernels), unchanged.
  - Every native operand the *same* nodal/quad value_form, and ``token``
    in :data:`_POINTWISE_TOKENS`: exact NumPy math on the raw view, wrapped
    back native -- mirrors ``operations.arithmetic``'s "compute on the view,
    wrap back native, stay in-value_form" pointwise dispatch.
  - Native operands in *different* value_forms: warns and falls back
    (the caller then runs ``func`` on plain views, same as a value_form
    mismatch anywhere else in this module).
  - Any other token (reductions, finite-difference derivatives): returns
    ``None`` so the caller's plain-NumPy path runs -- the result then
    genuinely leaves the native/value_form domain.
  """
  is_native = [dg.modal.is_native(v) for v in tmp_values]
  if not any(is_native):
    return None
  # end

  reps = {_rep_of(c) for v, c, native in zip(tmp_values, tmp_ctx, is_native) if native}
  if reps == {"modal"}:
    return _modal_kernel(token, tmp_grid, tmp_values, tmp_ctx)
  # end

  if len(reps) > 1:
    warnings.warn(
        f"evaluate: '{token}' mixes native operands in different "
        f"value_forms ({sorted(reps)}); falling back to plain math on "
        "the raw views.", stacklevel=3)
    return None
  # end

  if token not in _POINTWISE_TOKENS:
    return None
  # end

  view_values = [_modal_view(v, c) for v, c in zip(tmp_values, tmp_ctx)]
  out_grid, out_values = func(tmp_grid, view_values)
  return out_grid, [dg.rep.wrap(v) for v in out_values]
# end


def apply_operator(grid_stack, value_stack, ctx_stack, token: str) -> bool:
  """Reduce the RPN stacks in place by applying ``token`` if it is an operator.

  Each stack entry is a list of "sets" (grids/values/ctx dicts); an operator
  pops ``num_in`` entries, applies its pure function from
  :data:`postgkyl.numerics.ev_cmds` over every set (broadcasting shorter
  inputs), and pushes ``num_out`` results. The ctx of the output is the merge
  of the inputs' ctx, dropping any key whose value disagrees between inputs.

  Args:
    grid_stack, value_stack, ctx_stack: the parallel RPN stacks, mutated in
      place.
    token: the candidate operator token (e.g. ``'+'``, ``'sqrt'``, ``'int'``).

  Returns:
    True if ``token`` was a known operator and the stacks were reduced;
    False if ``token`` is not an operator (the stacks are untouched).

  Raises:
    ValueError: if the operator's function raises while evaluating.
  """
  if token not in ev_cmds:
    return False
  # end
  num_in = ev_cmds[token]["num_in"]
  num_out = ev_cmds[token]["num_out"]
  func = ev_cmds[token]["func"]

  in_grid, in_values, in_ctx, num_sets = [], [], [], []
  for _ in range(num_in):
    in_grid.append(grid_stack.pop())
    in_values.append(value_stack.pop())
    in_ctx.append(ctx_stack.pop())
    num_sets.append(len(in_values[-1]))
  # end
  for _ in range(num_out):
    grid_stack.append([])
    value_stack.append([])
    ctx_stack.append([])
  # end

  for set_idx in range(max(num_sets)):
    tmp_grid, tmp_values, tmp_ctx = [], [], []
    for i in range(num_in):
      tmp_grid.append(in_grid[i][min(set_idx, num_sets[i] - 1)])
      tmp_values.append(in_values[i][min(set_idx, num_sets[i] - 1)])
      tmp_ctx.append(in_ctx[i][min(set_idx, num_sets[i] - 1)])
    # end
    try:
      native_out = _native_kernel(token, tmp_grid, tmp_values, tmp_ctx, func)
      if native_out is not None:
        out_grid, out_values = native_out
      # end
      else:
        view_values = [_modal_view(v, c) for v, c in zip(tmp_values, tmp_ctx)]
        out_grid, out_values = func(tmp_grid, view_values)
      # end
    # end
    except Exception as err:
      raise ValueError(str(err)) from err
    # end

    # Merge ctx of all inputs; drop keys that disagree between inputs.
    out_ctx: dict = {}
    remove_list = []
    for i in range(num_in):
      for key in tmp_ctx[i]:
        if key in out_ctx and _compare(tmp_ctx[i][key], out_ctx[key]):
          pass  # already copied and matches; nothing to do
        # end
        elif key in out_ctx:
          remove_list.append(key)  # discrepancy; mark for removal
        # end
        else:
          out_ctx[key] = tmp_ctx[i][key]
        # end
      # end
    # end
    for key in dict.fromkeys(remove_list):
      out_ctx.pop(key)
    # end

    # A native nodal/quad operand whose result did *not* come back wrapped
    # native (a genuine reduction/derivative, per _native_kernel) has left
    # the per-point field domain: the merged ctx's 'value_form' is now
    # stale (it still names a value_form this output no longer has), so
    # drop it and mark the result the same way .interpolate() does -- no
    # longer gkyl-native -- rather than let info() keep describing it as a
    # value_form it left behind.
    was_native_nonmodal = any(
        dg.modal.is_native(v) and _rep_of(c) != "modal"
        for v, c in zip(tmp_values, tmp_ctx))

    for i in range(num_out):
      grid_stack[-num_out + i].append(out_grid[i])
      value_stack[-num_out + i].append(out_values[i])
      this_ctx = dict(out_ctx)
      if was_native_nonmodal and not dg.modal.is_native(out_values[i]):
        this_ctx.pop("value_form", None)
        this_ctx["interpolated"] = True
      # end
      ctx_stack[-num_out + i].append(this_ctx)
    # end
  # end
  return True
# end


def _push_token(token: str, datasets, grid_stack, value_stack, ctx_stack) -> bool:
  """Push a single non-operator ``token`` (data reference or literal).

  Returns False only if the token cannot be interpreted at all.
  """
  match = _DATA_TOKEN.match(token)
  if match:
    idx = int(match.group(1)) if match.group(1) else 0
    comp = match.group(2)
    ctx_key = match.group(3)
    dat = datasets[idx]
    if ctx_key is not None:
      if ctx_key not in dat.ctx:
        raise ValueError(f"evaluate: unknown ctx key '{ctx_key}' on dataset f{idx}")
      # end
      grid, values = None, np.array(dat.ctx[ctx_key])
    # end
    elif comp is None and dat.backend == "gkyl":
      # Keep native data on the stack (rather than forcing it through
      # select()'s point-value guard), regardless of value_form: RPN
      # math routes through Gkeyll's own weak kernels for modal data, or
      # exact NumPy math wrapped back native for nodal/quad point values,
      # when the operator supports it, or warns/falls back to the raw view
      # otherwise -- see _native_kernel.
      grid, values = dat.grid, dat.native
    # end
    else:
      # select() carries the shared operability guard (raw modal coefficients
      # refuse; nodal/quad value_forms, already point values, pass) for
      # a comp-sliced modal token (still genuinely unsafe -- slicing raw DG
      # coefficients by component can mix basis functions) and every
      # already-point-value token. select() itself now keeps a gkyl-backed
      # nodal/quad result native, so keep pushing the native array here too
      # (not its plain-view .values) so it stays eligible for _native_kernel.
      selected = select(dat, comp=comp)
      grid = selected.grid
      values = selected.native if selected.backend == "gkyl" else selected.values
    # end
    grid_stack.append([grid])
    value_stack.append([values])
    ctx_stack.append([dat.ctx])
    return True
  # end

  # Numeric / axis literal fallback (mirrors the CLI token parser).
  if "(" in token or "[" in token:
    value_stack.append([eval(token)])  # noqa: S307 -- trusted expression source
  # end
  elif ":" in token or "," in token:
    value_stack.append([str(token)])
  # end
  else:
    try:
      value_stack.append([np.array(float(token))])
    # end
    except ValueError:
      return False
    # end
  # end
  grid_stack.append([None])
  ctx_stack.append([{}])
  return True
# end


def available_operators() -> list[str]:
  """The RPN operator tokens ``evaluate`` recognizes (e.g. ``'+'``, ``'sqrt'``)."""
  return sorted(ev_cmds)
# end


def evaluate(chain: str, *datasets: "GDataState", tag: str | None = None,
    label: str | None = None) -> "GDataState":
  """Evaluate an RPN expression over an explicit list of datasets.

  ``f``/``fN`` tokens in ``chain`` refer to ``datasets[N]`` (``f`` == ``f0``);
  see the module docstring for the token grammar. The result is built via
  ``datasets[0]._result(...)`` (so it stays the caller's concrete dataset
  class) and holds the single value left on top of the stack.

  Args:
    chain: the RPN expression, e.g. ``"f0 f1 +"`` or ``"f sq 2 *"``.
    *datasets: the datasets referenced positionally by the ``f``/``fN``
      tokens. At least one is required (it anchors the result's class).
    tag: optional tag for the returned dataset (defaults to ``'default'``).
    label: optional label for the returned dataset (defaults to ``chain``).

  Returns:
    A dataset holding the evaluated grid/values and the merged ctx.

  Raises:
    ValueError: if ``datasets`` is empty, the expression is empty, a token
      is unrecognized, or an operator fails.
  """
  if not datasets:
    raise ValueError("evaluate: at least one dataset is required.")
  # end

  grid_stack, value_stack, ctx_stack = [], [], []
  for token in filter(None, chain.split(" ")):
    if apply_operator(grid_stack, value_stack, ctx_stack, token):
      continue
    # end
    if not _push_token(token, datasets, grid_stack, value_stack, ctx_stack):
      raise ValueError(f"evaluate: token '{token}' is neither data nor an operator")
    # end
  # end

  if not value_stack:
    raise ValueError("evaluate: expression produced no result")
  # end

  final_grid = grid_stack[-1][0]
  final_values = value_stack[-1][0]
  final_ctx = dict(ctx_stack[-1][0])
  out_grid = final_grid if final_grid is not None else datasets[0].grid
  result = datasets[0]._result(out_grid, final_values,
      tag=(tag or "default"), label=(label if label is not None else chain))
  # The result's ctx is the RPN merge (apply_operator already resolved every
  # conflict), not datasets[0]'s ctx that '_result' copied as a starting
  # point -- a key apply_operator dropped as conflicting must not survive
  # just because it happened to be on datasets[0]. 'cells'/'num_comps'/
  # 'lower'/'upper' are the shape/grid-derived facts '_result's push() just
  # recomputed from the actual final_grid/final_values; keep those.
  derived = {"cells", "num_comps", "lower", "upper"}
  kept = {k: result.ctx[k] for k in derived if k in result.ctx}
  result.ctx = final_ctx
  result.ctx.update(kept)
  return result
# end
