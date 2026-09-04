"""Weak (DG) arithmetic and NumPy interop.

Two backends, two sets of rules (see ``REFACTOR_GKEYLL_FFI.md`` section 3 /
the "two-domain lifecycle" in the project's ``CLAUDE.md``):

* **modal** (straight off disk): ``+``/``-``/``*``/``/`` on two ``GData``
  run *inside Gkeyll* (coefficient lin-combs / weak DG kernels), and
  ``.integrate()`` runs a grid integral there too. Plain NumPy math
  (``np.sqrt``, ``np.asarray``, ...) refuses -- coefficients aren't values.
* **numpy** (after ``.interpolate()``): the field is plain point values, so
  every NumPy ufunc and ``+``/``-``/``*``/``/`` "just work", and the result
  keeps carrying its grid/``ctx`` like a ``GData`` should.

Run directly:
    MPLBACKEND=Agg PYTHONPATH=src python examples/scripts/02_arithmetic_and_numpy.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

import postgkyl as pg

from _example_paths import TEST_DATA

# A coordinate-map field: strictly positive everywhere, so weak division
# never divides near zero (see the note on ``back`` below).
DATA = TEST_DATA / "generated" / "2d_c2p_stretch_ms_p1.gkyl"

# ----------------------------------------------------------- modal algebra
a = pg.load(DATA)
b = pg.load(DATA)

prod = a * b  # gkyl_dg_mul_op -- weak multiply
back = prod / b  # gkyl_dg_div_op -- weak divide
summ = a + b  # gkyl_array_accumulate -- coefficient sum

# ``a + a == 2*a`` is an exact linear identity -- coefficient lin-combs never
# alias. Weak multiply/divide, by contrast, is a *nonlinear* operation: the
# product of two degree-p polynomials is degree 2p, and projecting it back
# onto the degree-p basis is lossy in general. It only round-trips exactly
# when nothing aliases -- e.g. this field, or dividing by a field that never
# crosses zero. Compare via NumPy, which means interpolating first.
print("(a*b)/b == a:  ",
      np.allclose(back.interpolate().values,
                  a.interpolate().values))
print("a+a == 2*a:    ",
      np.allclose(summ.interpolate().values, (2.0 * a).interpolate().values))

total = a.integrate()  # gkyl_array_integrate -- a terminal verb
print("integrate(a) = ", total)  # one value per component (this field has 2)

# A general NumPy ufunc has no meaning on raw DG coefficients -- the
# capability boundary is enforced, not just "usually correct".
try:
  np.sqrt(a)
except ValueError as exc:
  print("np.sqrt(modal) refuses:", exc)

# ----------------------------------------------------- field-domain (NumPy)
fa = a.interpolate()
fb = b.interpolate()

field_sum = fa + fb
mag = np.sqrt(fa**2 + fb**2)  # ufunc -> still a GData, carrying fa's grid
print("sqrt(a^2+b^2) == sqrt(2)*a:",
      np.allclose(np.asarray(mag),
                  np.sqrt(2) * np.asarray(fa)))

as_array = np.asarray(fa)  # plain ndarray -- the escape hatch out of GData
print("np.asarray(interpolated) ->", as_array.shape, as_array.dtype)

print("02_arithmetic_and_numpy: OK")
