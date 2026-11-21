"""Tests for ``postgkyl.diagnostics.gk.regions`` on the synthetic run of
``gk_tokamak_fixture``.

The density ``n = a + b x`` and the Jacobian ``J = 1 + x`` are linear in
each block, so the region's volume average ``int n J / int J`` integrates
in closed form: over ``x`` in ``[x0, x1]`` and a theta extent ``Ly``,
``int n J = Ly (a [x + x^2/2] + b [x^2/2 + x^3/3])``. It differs from the
volume-weighted mean of cell means (the product of the two cell means
drops the ``b h^2 / 3`` of each cell), which the test also checks. A linear
field's cell mean is its value at the cell centre, which gives the median
relative change and the non-positive cells.
"""

from __future__ import annotations

import numpy as np
import pytest

import gk_tokamak_fixture as fx
from postgkyl import gpython
from postgkyl.diagnostics.gk import regions

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

REGION_BLOCKS = {"core": (0, ), "sol": (1, 2), "all": (0, 1, 2), "pf": ()}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
  return str(fx.write_run(tmp_path_factory.mktemp("run")))


def _extent(block):
  xe, ye = fx.cell_edges(block)
  return xe[0], xe[-1], ye[-1] - ye[0]


def _content(a, b, x0, x1):
  """int_{x0}^{x1} (a + b x)(1 + x) dx."""
  F = lambda x: a * (x + x**2 / 2) + b * (x**2 / 2 + x**3 / 3)  # noqa: E731
  return F(x1) - F(x0)


def _expected_average(region, frame):
  num = den = 0.0
  for b in REGION_BLOCKS[region]:
    x0, x1, ly = _extent(b)
    a, slope, _, _ = fx.density(b, frame)
    num += ly * _content(a, slope, x0, x1)
    den += ly * _content(1.0, 0.0, x0, x1)
  return num / den if den else np.nan


def _cell_centre_values(block, frame):
  xe, ye = fx.cell_edges(block)
  a, slope, _, _ = fx.density(block, frame)
  xc = 0.5 * (xe[1:] + xe[:-1])
  return np.repeat(a + slope * xc, ye.size - 1)


@needs_gkeyll
@pytest.mark.parametrize("region", ["core", "sol", "all"])
def test_region_average_is_the_volume_integral(run, region):
  got = regions.region_average("M0",
                               "elc",
                               fx.NAME,
                               "0:3",
                               path=run,
                               region=region)
  expected = [_expected_average(region, k) for k in range(3)]
  np.testing.assert_allclose(got.values, expected, rtol=1e-13)
  np.testing.assert_array_equal(got.frames, [0, 1, 2])
  np.testing.assert_array_equal(got.time, fx.FRAME_TIMES)
  assert got.statistic == "volume average" and got.region == region


@needs_gkeyll
def test_region_average_is_not_the_mean_of_cell_means(run):
  # The volume-weighted mean of cell means (J's and n's) for the core.
  xe, _ = fx.cell_edges(0)
  xc = 0.5 * (xe[1:] + xe[:-1])
  a, slope, _, _ = fx.density(0, 1)
  products = (a + slope * xc) * (1 + xc)
  of_means = products.sum() / (1 + xc).sum()
  got = regions.region_average("M0",
                               "elc",
                               fx.NAME,
                               "1",
                               path=run,
                               region="core")
  # Each cell's exact content exceeds the product of means by b h^2 / 3.
  h = 0.25
  exact = (products + slope * h**2 / 3).sum() / (1 + xc).sum()
  np.testing.assert_allclose(got.values, [exact], rtol=1e-13)
  assert abs(got.values[0] - of_means) > 1e-3


@needs_gkeyll
def test_region_average_of_an_empty_region_is_nan(run):
  got = regions.region_average("M0", "elc", fx.NAME, "0", path=run, region="pf")
  assert np.isnan(got.values).all()


@needs_gkeyll
@pytest.mark.parametrize("region", ["core", "sol", "all"])
def test_region_median_change(run, region):
  got = regions.region_median_change("M0",
                                     "elc",
                                     fx.NAME,
                                     "0:3",
                                     path=run,
                                     region=region)
  expected = []
  for k in range(3):
    change = np.concatenate([
        _cell_centre_values(b, k) / _cell_centre_values(b, 0) - 1
        for b in REGION_BLOCKS[region]
    ])
    expected.append(np.median(change))
  np.testing.assert_allclose(got.values, expected, rtol=1e-13, atol=1e-15)
  # An explicit reference frame.
  ref = regions.region_median_change("M0",
                                     "elc",
                                     fx.NAME,
                                     "2",
                                     path=run,
                                     region=region,
                                     reference=1)
  change = np.concatenate([
      _cell_centre_values(b, 2) / _cell_centre_values(b, 1) - 1
      for b in REGION_BLOCKS[region]
  ])
  np.testing.assert_allclose(ref.values, [np.median(change)], rtol=1e-13)


@needs_gkeyll
def test_nonpositive_cells(run):
  # Frame 2: block 2's density -4 + 2.5 x is -0.875 at x = 1.25, in both
  # of its theta cells.
  for region, count in (("all", 2), ("sol", 2), ("core", 0)):
    got = regions.nonpositive_cells("M0",
                                    "elc",
                                    fx.NAME,
                                    ":",
                                    path=run,
                                    region=region)
    np.testing.assert_array_equal(got.values, [0, 0, count])
  assert "non-positive cells of M0 (elc) over region 'core'" in str(got)
