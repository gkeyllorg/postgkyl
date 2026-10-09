"""The weak reciprocal ``dg.modal.weak_inv`` must not depend on the scale of
the field it inverts.

``gkyl_dg_inv_op`` raises coefficients to high powers, so in 3-D p1 a
constant field of 1e-39 or 1e40 overflows inside the kernel. A constant
field ``f = v`` has the exact weak reciprocal ``1/v`` (a constant lies in the
basis), so its coefficients are known in closed form at any magnitude: the
constant mode of ``v`` in the orthonormal basis is ``v * 2**(ndim/2)``.
"""

import numpy as np
import pytest

from postgkyl import dg, gpython
from postgkyl.gpython import GkylArray

pytestmark = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

# The reciprocal of a constant is exact up to double round-off in the
# kernel's few operations.
ROUND_OFF = 1e-13


def _constant(values, ndim):
  """One cell per entry of ``values``, each a constant field of that value."""
  values = np.asarray(values, dtype=float)
  coefficients = np.zeros((values.size, 2**ndim))
  coefficients[:, 0] = values * 2.0**(ndim / 2.0)
  return coefficients


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("value", [1e-150, 1e-39, 3.7, 1e40, 1e150, -2e-45])
def test_constant_field_inverts_exactly_at_any_magnitude(ndim, value):
  out = dg.modal.weak_inv("serendipity", ndim, 1,
                          GkylArray.from_numpy(_constant([value], ndim)))
  expected = _constant([1.0 / value], ndim)
  np.testing.assert_allclose(out.view(), expected, rtol=ROUND_OFF, atol=0.0)


def test_cells_and_fields_are_scaled_independently():
  """Cells of wildly different size, and two fields of different size in
  the same cell, each come back as their own exact reciprocal."""
  ndim = 3
  cell_values = np.array([1e-39, 1.0, 1e40, 5e-20])
  two_fields = np.concatenate(
      [_constant(cell_values, ndim),
       _constant(1e30 * cell_values[::-1], ndim)],
      axis=-1)

  out = dg.modal.weak_inv("serendipity", ndim, 1,
                          GkylArray.from_numpy(two_fields)).view()

  np.testing.assert_allclose(out[:, :8],
                             _constant(1.0 / cell_values, ndim),
                             rtol=ROUND_OFF,
                             atol=0.0)
  np.testing.assert_allclose(out[:, 8:],
                             _constant(1.0 / (1e30 * cell_values[::-1]), ndim),
                             rtol=ROUND_OFF,
                             atol=0.0)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_order_one_fields_match_the_unscaled_kernel_bit_for_bit(ndim):
  """Power-of-two scaling is exact, so a field the kernel already handles
  gives the identical result (here random, nonconstant coefficients around
  a positive mean)."""
  rng = np.random.default_rng(0)
  coefficients = 0.1 * rng.standard_normal((6, 2 * 2**ndim))
  coefficients[:, 0] += 3.0
  coefficients[:, 2**ndim] += 0.4
  field = GkylArray.from_numpy(coefficients)

  scaled = dg.modal.weak_inv("serendipity", ndim, 1, field).view()
  raw = gpython.kernels.weak_inv("serendipity", ndim, 1, field).view()

  np.testing.assert_array_equal(scaled, raw)


def test_zero_cells_keep_the_kernel_behaviour():
  """A zero field has no power of two to scale by; it is passed through
  unscaled, so the kernel's own result (non-finite) is returned."""
  coefficients = _constant([0.0, 2.0], 1)
  out = dg.modal.weak_inv("serendipity", 1, 1,
                          GkylArray.from_numpy(coefficients)).view()
  raw = gpython.kernels.weak_inv("serendipity", 1, 1,
                                 GkylArray.from_numpy(coefficients)).view()
  np.testing.assert_array_equal(out, raw)
