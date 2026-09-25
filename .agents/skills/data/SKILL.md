---
name: data
description: Preserve DG representation and backend semantics when changing loading, arithmetic, conversions, integration, or terminal data consumers.
---

# Preserve data meaning

Storage (`backend`: `gkyl` or `numpy`) and representation
(`ctx["value_form"]`: `modal`, `nodal`, or `quad`) are separate facts. Do not
add an `is_modal` flag or infer point-value capabilities from storage alone.

- Modal coefficients use Gkeyll DG operations: weak multiply/divide, coefficient
  linear combinations, scalar scaling/mean shifts, repeated weak multiplication
  for integer powers, and native integration. Ufuncs, array conversion, selection,
  and plotting must not treat coefficients as field values.
- Nodal and quadrature values are fields at points. Pointwise NumPy operations
  are allowed; native results wrap back into native storage in the same value_form.
  Plot true point locations; non-tensor node sets use explicit
  `.represent(to='quad')`.
- Reject mixed backends or mixed value_forms when combining datasets.
- `.interpolate()` is the one-way bridge to a new, by-value NumPy field array.
  Interpolation matrices come from Gkeyll basis functions, applied per cell;
  nodal input first uses the exact nodal-to-modal transform.
- Only explicit `.represent(to=...)` performs representation conversions
  between `modal`, `nodal`, and `quad`. `.apply(fn, num_quad=...)` spells
  modal → quad → fn → projection, equivalent to
  `fn(d.represent(to='quad')).represent(to='modal')`. Respect quadrature
  exactness limits.
- Full modal integration is terminal and native; partial integration uses native
  averaging with physical-volume scaling and returns lower-dimensional modal data.
  `.average()` and `.eval_at_coord_proj()` likewise stay modal/native and compose.
  `.local_poly()` builds a discontinuity-preserving plotting mesh.

Resolve `basis_type`, `poly_order`, and `value_form` once from file metadata or
explicit load options. Downstream verbs read ctx and raise if required metadata
is missing; never add basis/order override parameters to them.

For spatial data with unresolved basis metadata, loading warns and defaults to
serendipity p0 nodal data. When value_form is defaulted, express the grid as cell
centers, matching one point per cell. Dynvectors have no spatial DG basis and
are exempt. The readers' narrower case—known basis but no value_form tag—assumes
modal silently; keep these two defaults distinct.

Route results through `_result` and terminal native point-value consumers through
`gdatastate.materialize_point_values`. Preserve native ownership and read-only
coefficient views; use the existing representation tests to verify semantics.

## Arithmetic operators

Prefer `GData`'s ordinary arithmetic operators when composing diagnostics and
physical formulas. They dispatch through `operations.arithmetic`; on native
modal inputs, results stay native and modal:

| Expression | Modal meaning |
| --- | --- |
| `f + g`, `f - g` | Linear combinations of the two fields' coefficients. |
| `f * g` | Gkeyll weak multiplication, projected into the retained DG basis. |
| `f / g` | Gkeyll weak division, using a per-cell solve. |
| `f * s`, `f / s` | Scale all coefficients by a scalar `s` or its reciprocal. |
| `f + s`, `f - s` | Shift the constant mode by the correctly normalized scalar; higher modes are unchanged. |
| `1.0 / f` | Gkeyll weak inverse. |
| `f ** n` for positive integer `n` | Repeated weak multiplication. |
| `f ** 0.5` (equivalently `f ** (1 / 2)`) | Gkeyll quadrature projection of the square root onto the modal basis, retaining the native negative-value floor. |

Here `f` and `g` are compatible datasets and `s` is a scalar. A projected square
root is not generally an exact analytic square root. Zero, negative, and other
fractional powers also use the projected-power path; use `1.0 / f` when a weak
inverse is required, rather than substituting `f ** -1`.

Preserve the intended sequence of weak operations. Weak products are generally
not associative: `(f * g) * h` and `f * (g * h)` can differ. Likewise, `f / g`
and `f * (1.0 / g)` invoke different kernels; retain inverse-then-multiply when
that is the prescribed calculation, as in the GK quantity definitions. Do not
silently replace unsupported modal operations with pointwise calculations.

Use `density * temperature` and `(temperature / mass) ** 0.5` directly on the
fields. Do not interpolate first, multiply `.values` coefficient arrays, or use
`np.sqrt` on modal coefficients. Already nodal, quadrature, or interpolated
inputs follow their point-value semantics instead. Modal arithmetic is the most
exact manipulation on data. Prefer modal arithmetic.

The operator interface belongs to `GData`, not the inert `GDataState`. Formula
functions using operators should declare and receive `GData` (as the GK loader
returns). Lower layers that accept `GDataState` should keep using their existing
operation functions; do not import `GData` into those lower layers or duplicate
the arithmetic dispatch in diagnostic helpers.
