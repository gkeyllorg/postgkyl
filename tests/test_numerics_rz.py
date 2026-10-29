"""Analytic checks of toroidal-angle location and wedge sampling."""
import numpy as np
import pytest

from postgkyl.numerics.rz import binormal_fraction, wedge_surface


@pytest.mark.parametrize("span", [2.4, -2.4, 2 * np.pi / 3])
def test_angles_inside_the_binormal_wedge_are_located_exactly(span):
  # The domain covers phi_lo + f * span for f in [0, 1), at any 2*pi turn.
  rng = np.random.default_rng(3)
  phi_lo = rng.uniform(-20, 20, size=(4, 5))
  f = rng.uniform(0, 1, size=(4, 5))
  for turn in (-2, 0, 3):
    angle = phi_lo + f * span + 2 * np.pi * turn
    np.testing.assert_allclose(binormal_fraction(angle, phi_lo, span),
                               f,
                               rtol=0,
                               atol=1e-12)


@pytest.mark.parametrize("span", [2.4, -2.4])
def test_angles_beyond_the_wedge_repeat_the_binormal_domain(span):
  # 2*pi / 2.4 = 2.62 copies: past the wedge the domain repeats with period
  # |span| until the turn closes, where the copies meet.
  for f in (0.1, 0.55, 0.9):
    np.testing.assert_allclose(binormal_fraction((1 + f) * span, 0.0, span),
                               f,
                               atol=1e-12)
  np.testing.assert_allclose(binormal_fraction(2.6 * span, 0.0, span),
                             0.6,
                             atol=1e-12)
  turn = np.sign(span) * 2 * np.pi
  np.testing.assert_allclose(binormal_fraction(turn - 1e-9 * span, 0.0, span),
                             turn / span % 1.0,
                             atol=1e-8)


def test_whole_fraction_of_the_torus_repeats_seamlessly():
  # With 2*pi = 3 * span, every copy agrees with plain binormal periodicity.
  span = 2 * np.pi / 3
  angle = np.linspace(-15, 15, 301)
  np.testing.assert_allclose(binormal_fraction(angle, 0.7, span),
                             np.mod((angle - 0.7) / span, 1.0),
                             rtol=0,
                             atol=1e-12)


def test_wedge_surface_walks_and_closes_the_boundary():
  rows = np.array([2, 3, 4, 5])
  angles = np.array([0.0, 0.5, 1.0])
  nz = 3

  def sample(rows, angles):
    # Encode (row, angle, z) and a trailing component axis.
    z = np.arange(nz)
    row_angle = rows[:, None] * 100.0 + angles * 10.0
    return np.stack(np.broadcast_arrays(row_angle + z, -row_angle), axis=-1)

  surface = wedge_surface(sample, rows, angles)

  inner, outer = rows[0], rows[-1]
  path = ([(r, angles[0]) for r in rows] + [(outer, a) for a in angles[1:]] +
          [(r, angles[-1]) for r in rows[-2::-1]] + [(inner, a)
                                                     for a in angles[-2::-1]])
  assert path[-1] == path[0]
  expected = np.array([[(100.0 * r + 10.0 * a + z, -(100.0 * r + 10.0 * a))
                        for z in [*range(nz), 0]] for r, a in path])
  assert surface.shape == (2 * 4 + 2 * 3 - 3, nz + 1, 2)
  np.testing.assert_array_equal(surface, expected)
